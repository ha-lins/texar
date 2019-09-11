#!/usr/bin/env python3
"""
Text Content Manipulation
3-gated copy net.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import importlib
import os
import numpy as np
import tensorflow as tf
import texar as tx
print('path is {}'.format(tx.__path__))
import pickle
from texar.core import get_train_op
from utils import *
from ie import get_precrec
from texar.modules import TransformerEncoder, TransformerDecoder, TransformerCopyDecoder
from texar.utils import transformer_utils


flags = tf.flags
flags.DEFINE_string("config_data", "config_data_nba", "The data config.")
flags.DEFINE_string("config_model", "config_model_transformer", "The model config.")
flags.DEFINE_string("config_train", "config_train", "The training config.")
flags.DEFINE_float("rec_w", 0.8, "Weight of reconstruction loss.")
flags.DEFINE_float("rec_w_rate", 0., "Increasing rate of rec_w.")
flags.DEFINE_boolean("add_bleu_weight", False, "Whether to multiply BLEU weight"
                     " onto the first loss.")
flags.DEFINE_string("expr_name", "nba", "The experiment name. "
                    "Used as the directory name of run.")
flags.DEFINE_string("restore_from", "", "The specific checkpoint path to "
                    "restore from. If not specified, the latest checkpoint in "
                    "expr_name is used.")
flags.DEFINE_boolean("copy_x", False, "Whether to copy from x.")
flags.DEFINE_boolean("copy_y_", False, "Whether to copy from y'.")
flags.DEFINE_boolean("coverage", False, "Whether to add coverage onto the copynets.")
flags.DEFINE_float("exact_cover_w", 0., "Weight of exact coverage loss.")
flags.DEFINE_float("eps", 1e-10, "epsilon used to avoid log(0).")
flags.DEFINE_integer("disabled_vocab_size", 0, "Disabled vocab size.")
flags.DEFINE_boolean("attn_x", False, "Whether to attend x.")
flags.DEFINE_boolean("attn_y_", False, "Whether to attend y'.")
flags.DEFINE_boolean("sd_path", False, "Whether to add structured data path.")
flags.DEFINE_float("sd_path_multiplicator", 1., "Structured data path multiplicator.")
flags.DEFINE_float("sd_path_addend", 0., "Structured data path addend.")
flags.DEFINE_boolean("align", False, "Whether it is to get alignment.")
flags.DEFINE_boolean("output_align", False, "Whether to output alignment.")
flags.DEFINE_boolean("verbose", False, "verbose.")
flags.DEFINE_boolean("eval_ie", False, "Whether evaluate IE.")
flags.DEFINE_integer("eval_ie_gpuid", 3, "ID of GPU on which IE runs.")
FLAGS = flags.FLAGS

copy_flag = FLAGS.copy_x or FLAGS.copy_y_
attn_flag = FLAGS.attn_x or FLAGS.attn_y_

if FLAGS.output_align:
    FLAGS.align = True

config_data = importlib.import_module(FLAGS.config_data)
config_model = importlib.import_module(FLAGS.config_model)
config_train = importlib.import_module(FLAGS.config_train)
expr_name = FLAGS.expr_name
restore_from = FLAGS.restore_from

dir_summary = os.path.join(expr_name, 'log')
dir_model = os.path.join(expr_name, 'ckpt')
dir_best = os.path.join(expr_name, 'ckpt-best')
ckpt_model = os.path.join(dir_model, 'model.ckpt')
ckpt_best = os.path.join(dir_best, 'model.ckpt')


def get_optimistic_restore_variables(ckpt_path, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(ckpt_path)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([
        (var.name, var.name.split(':')[0]) for var in tf.global_variables()
        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        var = graph.get_tensor_by_name(var_name)
        var_shape = var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(var)
    return restore_vars


def get_optimistic_saver(ckpt_path, graph=tf.get_default_graph()):
    return tf.train.Saver(
        get_optimistic_restore_variables(ckpt_path, graph=graph))


def build_model(data_batch, data, step):
    batch_size, num_steps = [
        tf.shape(data_batch["value_text_ids"])[d] for d in range(2)]
    vocab = data.vocab('sent')

    id2str = '<{}>'.format
    bos_str, eos_str = map(id2str, (vocab.bos_token_id, vocab.eos_token_id))

    def single_bleu(ref, hypo):
        ref = [id2str(u if u != vocab.unk_token_id else -1) for u in ref]
        hypo = [id2str(u) for u in hypo]

        ref = tx.utils.strip_special_tokens(
            ' '.join(ref), strip_bos=bos_str, strip_eos=eos_str)
        hypo = tx.utils.strip_special_tokens(
            ' '.join(hypo), strip_eos=eos_str)

        return 0.01 * tx.evals.sentence_bleu(references=[ref], hypothesis=hypo)

    def batch_bleu(refs, hypos):
        return np.array(
            [single_bleu(ref, hypo) for ref, hypo in zip(refs, hypos)],
            dtype=np.float32)


    # losses
    losses = {}

    # embedders
    embedders = {
        name: tx.modules.WordEmbedder(
            vocab_size=data.vocab(name).size, hparams=hparams)
        for name, hparams in config_model.embedders.items()}

    # encoders
    sent_encoder = tx.modules.TransformerEncoder(
        hparams=config_model.sent_encoder)
    sd_encoder = tx.modules.TransformerEncoder(
        hparams=config_model.sd_encoder)


    def concat_encoder_outputs(outputs):
        return tf.concat(outputs, -1)


    def encode(ref_str):
        sent_ids = data_batch['sent{}_text_ids'.format(ref_str)]
        sent_embeds = embedders['sent'](sent_ids)
        # print('sent embeds are :{}'.format(sent_embeds))
        sent_sequence_length = data_batch['sent{}_length'.format(ref_str)]
        sent_enc_outputs = sent_encoder(
            sent_embeds, sequence_length=sent_sequence_length)
        sent_enc_outputs = concat_encoder_outputs(sent_enc_outputs)

        sd_ids = {
            field: data_batch['{}{}_text_ids'.format(field, ref_str)][:, 1:-1]
            for field in sd_fields}
        sd_embeds = tf.concat(
            [embedders[field](sd_ids[field]) for field in sd_fields],
            axis=-1)
        sd_sequence_length = data_batch[
            '{}{}_length'.format(sd_fields[0], ref_str)] - 2
        sd_enc_outputs = sd_encoder(
            sd_embeds, sequence_length=sd_sequence_length)
        sd_enc_outputs = concat_encoder_outputs(sd_enc_outputs)

        return sent_ids, sent_embeds, sent_enc_outputs, sent_sequence_length, \
            sd_ids, sd_embeds, sd_enc_outputs, sd_sequence_length


    encode_results = [encode(ref_str) for ref_str in ref_strs]
    sent_ids, sent_embeds, sent_enc_outputs, sent_sequence_length, \
            sd_ids, sd_embeds, sd_enc_outputs, sd_sequence_length = \
        zip(*encode_results)

    # get rnn cell
    # rnn_cell = tx.core.layers.get_rnn_cell(config_model.rnn_cell)


    def get_decoder( y__ref_flag, x_ref_flag, tgt_ref_flag,
                    beam_width=None):
        output_layer_params = \
             {'output_layer': tf.identity} if copy_flag else {'vocab_size': vocab.size}


        if attn_flag: # attention
            memory = tf.concat(
                [sent_enc_outputs[y__ref_flag],
                 sd_enc_outputs[x_ref_flag]],
                axis=1)
            memory_sequence_length = None
            copy_memory_sequence_length = None

            tgt_embedding = tf.concat(
                [tf.zeros(shape=[1, embedders['sent'].dim]), embedders['sent'].embedding[1:, :]], axis=0)
            decoder = tx.modules.TransformerDecoder(
                embedding=tgt_embedding,
                hparams=config_model.decoder)

        return decoder

    def get_decoder_and_outputs(
            y__ref_flag, x_ref_flag, tgt_ref_flag, params,
            beam_width=None):
        decoder = get_decoder(
            y__ref_flag, x_ref_flag, tgt_ref_flag,
            beam_width=beam_width)
        if beam_width is None:
            ret = decoder(**params)
        else:
            ret = decoder(
                beam_width=beam_width,
                **params)
        return decoder, ret

    get_decoder_and_outputs = tf.make_template(
        'get_decoder_and_outputs', get_decoder_and_outputs)

    def teacher_forcing(y__ref_flag, x_ref_flag, loss_name):
        tgt_ref_flag = x_ref_flag
        tgt_str = 'sent{}'.format(ref_strs[tgt_ref_flag])
        memory_sequence_length = tf.add(sent_sequence_length[y__ref_flag] - 1, sd_sequence_length[x_ref_flag])
        sequence_length = data_batch['{}_length'.format(tgt_str)] - 1

        memory = tf.concat(
            [sent_enc_outputs[y__ref_flag],
             sd_enc_outputs[x_ref_flag]],
            axis=1)
        source_ids = tf.concat(
            [sent_ids[y__ref_flag],
             sd_ids[x_ref_flag]['entry']],
            axis=1)

        decoder, tf_outputs = get_decoder_and_outputs(
            y__ref_flag, x_ref_flag, tgt_ref_flag,
            {
             'memory': memory,
             'memory_sequence_length': memory_sequence_length,
             #'copy_memory': memory, #sd_enc_outputs[x_ref_flag],
             #'source_ids': source_ids,         #sd_ids[x_ref_flag]['entry'],
             'decoding_strategy': 'train_greedy',
             'inputs': sent_embeds[tgt_ref_flag][:, 1:, :], #target sentence embeds (ignore <BOS>)
             'alpha': config_model.alpha,
             'sequence_length': sequence_length,
             'mode': tf.estimator.ModeKeys.TRAIN})

        tgt_sent_ids = data_batch['{}_text_ids'.format(tgt_str)][:, 1:]  # ground_truth ids (ignore <BOS>)
        loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=tgt_sent_ids,
            logits=tf_outputs.logits,
            sequence_length=data_batch['{}_length'.format(tgt_str)] - 1,
            average_across_batch=False)

        loss = tf.reduce_mean(loss, 0)

        # if copy_flag and FLAGS.exact_cover_w != 0:
        #     sum_copy_probs = list(map(lambda t: tf.cast(t, tf.float32), final_state.sum_copy_probs))
        #     memory_lengths = [lengths for _, _, lengths in decoder.cell.memory_ids_states_lengths]
        #     exact_coverage_losses = [
        #         tf.reduce_mean(tf.reduce_sum(
        #             tx.utils.mask_sequences(
        #                 tf.square(sum_copy_prob - 1.), memory_length),
        #             1))
        #         for sum_copy_prob, memory_length in zip(sum_copy_probs, memory_lengths)]
        #     print_xe_loss_op = tf.print(loss_name, 'xe loss:', loss)
        #     with tf.control_dependencies([print_xe_loss_op]):
        #         for i, exact_coverage_loss in enumerate(exact_coverage_losses):
        #             print_op = tf.print(loss_name, 'exact coverage loss {:d}:'.format(i), exact_coverage_loss)
        #             with tf.control_dependencies([print_op]):
        #                 loss += FLAGS.exact_cover_w * exact_coverage_loss

        losses[loss_name] = loss

        return decoder, tf_outputs, loss, tgt_sent_ids


    def beam_searching(y__ref_flag, x_ref_flag, beam_width):
        start_tokens = tf.ones_like(data_batch['sent_length']) * \
            vocab.bos_token_id
        end_token = vocab.eos_token_id
        memory_sequence_length = tf.add(sent_sequence_length[y__ref_flag] - 1, sd_sequence_length[x_ref_flag])
        sequence_length = data_batch['sent{}_length'.format(ref_strs[y__ref_flag])] - 1

        memory = tf.concat(
            [sent_enc_outputs[y__ref_flag],
             sd_enc_outputs[x_ref_flag]],
            axis=1)
        source_ids = tf.concat(
            [sent_ids[y__ref_flag],
             sd_ids[x_ref_flag]['entry']],
            axis=1)
        tgt_ref_flag = x_ref_flag
        tgt_str = 'sent{}'.format(ref_strs[tgt_ref_flag])
        copy_memory_sequence_length = None

        decoder, bs_outputs = get_decoder_and_outputs(
            y__ref_flag, x_ref_flag, None,
            {
             'memory': memory,
             # 'embedding': embedders['sent'],
             'memory_sequence_length': memory_sequence_length,
             #'copy_memory': memory,     #sd_enc_outputs[x_ref_flag],
             #'source_ids': source_ids,     #sd_ids[x_ref_flag]['entry'],        #[ batch_size, source_length]
             'decoding_strategy': 'infer_greedy',
             'alpha': config_model.alpha,
             'start_tokens': start_tokens,
             'end_token': end_token,
             'max_decoding_length': config_train.infer_max_decoding_length},
            beam_width=config_train.infer_beam_width)

        return decoder, bs_outputs, sequence_length, start_tokens



    decoder, tf_outputs, loss, tgt_sent_ids = teacher_forcing(1, 0, 'MLE')
    rec_decoder, _, rec_loss, _ = teacher_forcing(1, 1, 'REC')
    rec_weight = FLAGS.rec_w + FLAGS.rec_w_rate * tf.cast(step, tf.float32)
    # step_stage = tf.cast(step, tf.float32) / tf.constant(982.0)
    # rec_weight = tf.case([(tf.less_equal(step_stage, tf.constant(1.5)), lambda: tf.constant(1.0)), \
    #                       (tf.greater(step_stage, tf.constant(1.5)), lambda: FLAGS.rec_w)], \
    #                      default=lambda: tf.constant(1.0) - (step_stage - 1) * (1 - FLAGS.rec_w))
    joint_loss = (1 - rec_weight) * loss + rec_weight * rec_loss
    losses['joint'] = joint_loss

    tiled_decoder, bs_outputs, sequence_length, start_tokens = beam_searching(
        1, 0, config_train.infer_beam_width)

    train_ops = {
        name: get_train_op(losses[name], hparams=config_train.train[name])
        for name in config_train.train}

    return train_ops, bs_outputs, tf_outputs, sequence_length, tgt_sent_ids, start_tokens


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    global_step = tf.train.get_or_create_global_step()

    train_ops, bs_outputs, mle_outputs, sequence_length, tgt_sent_ids, start_tokens = build_model(data_batch, datasets['train'], global_step)

    summary_ops = {
        name: tf.summary.merge(
            tf.get_collection(
                tf.GraphKeys.SUMMARIES,
                scope=get_scope_name_of_train_op(name)),
            name=get_scope_name_of_summary_op(name))
        for name in train_ops.keys()}

    saver = tf.train.Saver(max_to_keep=None)

    global best_ever_val_bleu
    best_ever_val_bleu = 0.


    def _save_to(directory, step):
        print('saving to {} ...'.format(directory))

        saved_path = saver.save(sess, directory, global_step=step)

        print('saved to {}'.format(saved_path))


    def _restore_from_path(ckpt_path):
        print('restoring from {} ...'.format(ckpt_path))

        try:
            saver.restore(sess, ckpt_path)
        except tf.errors.NotFoundError:
            print('Some variables are missing. Try optimistically restoring.')
            (get_optimistic_saver(ckpt_path)).restore(sess, ckpt_path)

        print('done.')


    def _restore_from(directory):
        if os.path.exists(directory):
            ckpt_path = tf.train.latest_checkpoint(directory)
            _restore_from_path(ckpt_path)

        else:
            print('cannot find checkpoint directory {}'.format(directory))



    def _train_epoch(sess, summary_writer, mode, train_op, summary_op, mle_outputs):
        print('in _train_epoch')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }

        cnt = 0

        while True:
            try:
                loss, summary, tf_outputs, ground_truth = \
                    sess.run((train_op, summary_op, mle_outputs, tgt_sent_ids), feed_dict)

                step = tf.train.global_step(sess, global_step)

                print('step {:d}: loss = {:.6f}'.format(step, loss))

                summary_writer.add_summary(summary, step)

                if step % config_train.steps_per_eval == 0:
                    _eval_epoch(sess, summary_writer, 'val')
                    # _eval_epoch(sess, summary_writer, 'test')

            except tf.errors.OutOfRangeError:
                break

        print('end _train_epoch')


    def _eval_epoch(sess, summary_writer, mode):
        global best_ever_val_bleu

        print('in _eval_epoch with mode {}'.format(mode))

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode)
        }

        step = tf.train.global_step(sess, global_step)

        ref_hypo_pairs = []
        fetches = [
            [data_batch['sent_text'], data_batch['sent_ref_text']],
            bs_outputs['sample_id'],#[:, :, 0],
            sequence_length,
            tgt_sent_ids,
            start_tokens,
        ] if not FLAGS.align else [
            [data_batch['entry_text'], data_batch['entry_ref_text']],
            align_bs_outputs.predicted_ids,
        ]

        if not os.path.exists(dir_model):
            os.makedirs(dir_model)

        hypo_file_name = os.path.join(
            dir_model, "hypos.step{}.{}.txt".format(step, mode))
        hypo_file = open(hypo_file_name, "w")

        cnt = 0
        while True:
            try:
                target_texts, bs_ids, seq_length, ground_truth, _start_tokens = sess.run(fetches, feed_dict)
                target_texts = [
                    tx.utils.strip_special_tokens(
                        texts[:, 1:].tolist(), is_token_list=True)
                    for texts in target_texts]
                output_ids = bs_ids[:, :, 0]
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids.tolist(), vocab=datasets[mode].vocab('sent'),
                    join=False)

                target_texts = list(zip(*target_texts))

                for ref, hypo in zip(target_texts, output_texts):
                    if cnt < 10:
                        print('cnt = {}'.format(cnt))
                        for i, s in enumerate(ref):
                            print('ref{}: {}'.format(i, ' '.join(s)))
                        print('hypo: {}'.format(' '.join(hypo)))
                    print(' '.join(hypo), file=hypo_file)
                    cnt += 1
                print('processed {} samples'.format(cnt))

                ref_hypo_pairs.extend(zip(target_texts, output_texts))

            except tf.errors.OutOfRangeError:
                break

        hypo_file.close()

        if FLAGS.eval_ie and mode == 'test':
            gold_file_name = os.path.join(
                config_data.dst_dir, "gold.{}.txt".format(
                    config_data.mode_to_filemode[mode]))
            inter_file_name = "{}.h5".format(hypo_file_name[:-len(".txt")])
            ref_file_name = os.path.join(config_data.dst_dir, "nba.sent_ref.{}.txt".format(config_data.mode_to_filemode[mode]))
            prec, rec = get_precrec(
                gold_file_name, hypo_file_name, inter_file_name, ref_file_name,
                gpuid=3)

        refs, hypos = zip(*ref_hypo_pairs)
        bleus = []
        get_bleu_name = '{}_BLEU'.format
        print('In {} mode:'.format(mode))
        for i in range(1, 2):
            refs_ = list(map(lambda ref: ref[i:i+1], refs))
            bleu = corpus_bleu(refs_, hypos)
            print('{}: {:.2f}'.format(get_bleu_name(i), bleu))
            bleus.append(bleu)

        summary = tf.Summary()
        for i, bleu in enumerate(bleus):
            summary.value.add(
                tag='{}/{}'.format(mode, get_bleu_name(i)), simple_value=bleu)
        if FLAGS.eval_ie and mode == 'test':
            for name, value in {'precision': prec, 'recall': rec}.items():
                summary.value.add(tag='{}/{}'.format(mode, name),
                                  simple_value=value)
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

        bleu = bleus[0]
        if mode == 'val':
            if bleu > best_ever_val_bleu:
                best_ever_val_bleu = bleu
                print('updated best val bleu: {}'.format(bleu))

                _save_to(ckpt_best, step)

        print('end _eval_epoch')
        return bleu


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        if restore_from:
            _restore_from_path(restore_from)
        else:
            _restore_from(dir_model)


        summary_writer = tf.summary.FileWriter(
            dir_summary, sess.graph, flush_secs=30)

        epoch = 0
        while epoch < config_train.max_epochs:
            name = 'align' if FLAGS.align else 'joint'
            train_op = train_ops[name]
            summary_op = summary_ops[name]

            # val_bleu = _eval_epoch(sess, summary_writer, 'val')
            # step = tf.train.global_step(sess, global_step)
            #
            # print('epoch: {} ({}), step: {}, '
            #       'val BLEU: {:.2f}'.format(
            #     epoch, name, step, val_bleu))

            _train_epoch(sess, summary_writer, 'train', train_op, summary_op, mle_outputs)

            epoch += 1

            step = tf.train.global_step(sess, global_step)
            _save_to(ckpt_model, step)

        test_bleu = _eval_epoch(sess, summary_writer, 'test')
        print('epoch: {}, test BLEU: {}'.format(epoch, test_bleu))


if __name__ == '__main__':
    main()
