# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Transformer decoder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=invalid-name

import collections

import tensorflow as tf
from tensorflow.contrib.framework import nest

from texar.core import layers
from texar.module_base import ModuleBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar.modules.embedders.position_embedders import SinusoidsPositionEmbedder
from texar.modules.encoders.transformer_encoders import \
    default_transformer_poswise_net_hparams
from texar.modules.encoders.multihead_attention import \
    MultiheadAttentionEncoder
from texar.utils import beam_search
from texar.utils.shapes import shape_list, mask_sequences
from texar.utils import transformer_attentions as attn
from texar.utils.mode import is_train_mode
from texar.utils import transpose_batch_time


__all__ = [
    "TransformerDecoderOutput",
    "TransformerDecoder"
]


class TransformerDecoderOutput(
        collections.namedtuple("TransformerDecoderOutput",
                               ("logits", "sample_id"))):
    """The output of :class:`TransformerDecoder`.

    Attributes:
        logits: A float Tensor of shape
            `[batch_size, max_time, vocab_size]` containing the logits.
        sample_id: An int Tensor of shape `[batch_size, max_time]`
            containing the sampled token indexes.
    """
    pass


class TransformerDecoder(ModuleBase):
    """Transformer decoder that applies multi-head attention for
    sequence decoding.
    Stacked :class:`~texar.modules.encoders.MultiheadAttentionEncoder` for
    encoder-decoder attention and self attention,
    :class:`~texar.modules.FeedForwardNetwork` and residual connections.

    Use the passed `embedding` variable as the parameters of the
    transform layer from output to logits.

    Args:
        embedding: A Tensor of shape `[vocab_size, dim]` containing the
            word embedding. The Tensor is used as the decoder output layer.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    .. document private functions
    .. automethod:: _build
    """
    def __init__(self, embedding, hparams=None):
        ModuleBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer( \
                    layers.get_initializer(self._hparams.initializer))

            self.position_embedder = \
                SinusoidsPositionEmbedder(
                    self._hparams.position_embedder_hparams)

            self._embedding = embedding
            self._vocab_size = self._embedding.get_shape().as_list()[0]

            self.output_layer = \
                self._build_output_layer(shape_list(self._embedding)[-1])
            self.multihead_attentions = {
                'self_att': [],
                'encdec_att': []
            }
            self.poswise_networks = []
            for i in range(self._hparams.num_blocks):
                layer_name = 'layer_{}'.format(i)
                with tf.variable_scope(layer_name):
                    with tf.variable_scope("self_attention"):
                        multihead_attention = MultiheadAttentionEncoder(
                            self._hparams.multihead_attention)
                        self.multihead_attentions['self_att'].append(
                            multihead_attention)
                    # pylint: disable=protected-access
                    if self._hparams.dim != \
                        multihead_attention._hparams.output_dim:
                        raise ValueError('The output dimenstion of'
                                         'MultiheadEncoder should be equal'
                                         'to the dim of TransformerDecoder')

                    with tf.variable_scope('encdec_attention'):
                        multihead_attention = MultiheadAttentionEncoder(
                            self._hparams.multihead_attention)
                        self.multihead_attentions['encdec_att'].append(
                            multihead_attention)
                    if self._hparams.dim != \
                        multihead_attention._hparams.output_dim:
                        raise ValueError('The output dimenstion of'
                                         'MultiheadEncoder should be equal'
                                         'to the dim of TransformerDecoder')

                    poswise_network = FeedForwardNetwork(
                        hparams=self._hparams['poswise_feedforward'])
                    if self._hparams.dim != \
                        poswise_network._hparams.layers[-1]['kwargs']['units']:
                        raise ValueError('The output dimenstion of'
                                         'FeedForwardNetwork should be equal'
                                         'to the dim of TransformerDecoder')
                    self.poswise_networks.append(poswise_network)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # Same as in TransformerEncoder
                "num_blocks": 6,
                "dim": 512,
                "position_embedder_hparams": None,
                "embedding_dropout": 0.1,
                "residual_dropout": 0.1,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                "multihead_attention": {
                    "num_units": 512,
                    "num_heads": 8,
                },
                "initializer": None,
                # Additional for TransformerDecoder
                "embedding_tie": True,
                "output_layer_bias": False,
                "max_decoding_length": 1e10,
                "name": "transformer_decoder"
            }

        Here:

        "num_blocks" : int
            Number of stacked blocks.

        "dim" : int
            Hidden dimension of the encoder.

        "position_embedder_hparams" : dict, optional
            Hyperparameters of a
            :class:`~texar.modules.SinusoidsPositionEmbedder` as position
            embedder. If `None`, the
            :meth:`~texar.modules.SinusoidsPositionEmbedder.default_hparams`
            is used.

        "embedding_dropout": float
            Dropout rate of the input word and position embeddings.

        "residual_dropout" :  float
            Dropout rate of the residual connections.

        "poswise_feedforward" : dict,
            Hyperparameters for a feed-forward network used in residual
            connections.
            Make sure the dimension of the output tensor is equal to `dim`.

            See :func:`~texar.modules.default_transformer_poswise_net_hparams`
            for details.

        "multihead_attention": dict,
            Hyperparameters for the multihead attention strategy.
            Make sure the `output_dim` in this module is equal to `dim`.

            See :func:
            `~texar.modules.encoder.MultiheadAttentionEncoder.default_harams`
            for details.

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "embedding_tie" : bool
            Whether to use the word embedding matrix as the output layer
            that computes logits. If `False`, an additional dense layer
            is created.

        "output_layer_bias" : bool
            Whether to use bias to the output layer.

        "max_decoding_length" : int
            The maximum allowed number of decoding steps.
            Set to a very large number of avoid the length constraint.
            Ignored if provided in :meth:`_build` or
            "train_greedy" decoding is used.

            Length penalty coefficient. Refer to
            https://arxiv.org/abs/1609.08144 for more details.

        "name" : str
            Name of the module.
        """
        return {
            "num_blocks": 6,
            "initializer": None,
            "position_embedder_hparams": None,
            "embedding_tie": True,
            "output_layer_bias": False,
            "max_decoding_length": 1e10,
            "embedding_dropout": 0.1,
            "residual_dropout": 0.1,
            "poswise_feedforward": default_transformer_poswise_net_hparams(),
            'multihead_attention': {
                'num_units': 512,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'num_heads': 8,
            },
            "dim": 512,
            "name": "transformer_decoder",
        }

    def _prepare_tokens_to_embeds(self, tokens):
        """ a callable function to transform tokens into embeddings."""
        token_emb = tf.nn.embedding_lookup(self._embedding, tokens)
        return token_emb

    def _inputs_to_outputs_fn(self, max_length):
        """Returns a function that accepts the inputs and related
        decoding status, and returns the outputs of this step (for example,
        the predicted logits of the next token).
        """
        channels = shape_list(self._embedding)[-1]
        timing_signal = self.position_embedder(max_length, channels)

        def _impl(inputs, step, cache):
            """The function is called in dynamic decoding.
            """
            # Multiply embedding by sqrt of its dimention
            inputs *= self._embedding.shape.as_list()[-1]**0.5
            inputs += timing_signal[:, step]
            outputs = self._self_attention_stack(
                tf.expand_dims(inputs, axis=1),
                memory=cache['memory'],
                cache=cache,
            )
            outputs = self.output_layer(outputs)
            outputs = tf.squeeze(outputs, axis=[1])
            return outputs, cache

        return _impl

    def _build(self,    # pylint: disable=arguments-differ
               memory,
               memory_sequence_length=None,
               memory_attention_bias=None,
               inputs=None,
               sequence_length=None,
               decoding_strategy='train_greedy',
               beam_width=1,
               alpha=0,
               start_tokens=None,
               end_token=None,
               softmax_temperature=None,
               max_decoding_length=None,
               helper=None,
               mode=None):
        """Performs decoding.

        The decoder supports 4 decoding strategies. For the first 3 strategies,
        set :attr:`decoding_strategy` to the respective string.

        - **"train_greedy"**: decoding in teacher-forcing fashion \
          (i.e., feeding \
          ground truth to decode the next step), and for each step sample \
          is obtained by taking the `argmax` of logits. \
          Argument :attr:`inputs` is required for this strategy. \
          :attr:`sequence_length` is optional.
        - **"infer_greedy"**: decoding in inference fashion (i.e., feeding \
          `generated` sample to decode the next step), and for each
          step sample is obtained by taking the `argmax` of logits.\
          Arguments :attr:`(start_tokens, end_token)` are \
          required for this strategy, and argument \
          :attr:`max_decoding_length` is optional.
        - **"infer_sample"**: decoding in inference fashion, and for each step\
          sample is obtained by `random sampling` from the logits.
          Arguments :attr:`(start_tokens, end_token)` are \
          required for this strategy, and argument \
          :attr:`max_decoding_length` is optional.
        - **Beam Search**: set :attr:`beam_width` to > 1 to use beam search \
          decoding.\
          Arguments :attr:`(start_tokens, end_token)` are \
          required, and argument \
          :attr:`max_decoding_length` is optional.

        Args:
            memory: The memory to attend, e.g., the output of an RNN encoder.
                A Tensor of shape `[batch_size, memory_max_time, dim]`.
            memory_sequence_length (optional): A Tensor of shape `[batch_size]`
                containing the sequence lengths for the batch entries in
                memory. Used to create attention bias of
                :attr:`memory_attention_bias` is not given. Ignored if
                `memory_attention_bias` is provided.
            memory_attention_bias (optional): A Tensor of shape
                `[batch_size, num_heads, memory_max_time, dim]`.
                An attention bias typically sets the value of a padding
                position to a large negative value for masking. If not given,
                :attr:`memory_sequence_length` is used to automatically
                create an attention bias.
            inputs (optional): Input tensor for teacher forcing decoding, of
                shape `[batch_size, target_max_time, emb_dim]` containing the
                target sequence word embeddings.
                Used when :attr:`decoding_strategy` is set to "train_greedy".
            sequence_length (optional): A Tensor of shape `[batch_size]`,
                containing the sequence length of :attr:`inputs`.
                Tokens beyond the respective sequence length are masked out.
                Used when :attr:`decoding_strategy` is set to
                "train_greedy".
            decoding_strategy (str): A string specifying the decoding
                strategy, including "train_greedy", "infer_greedy",
                "infer_sample".
                Different arguments are required based on the
                strategy. See above for details. Ignored if
                :attr:`beam_width` > 1.
            beam_width (int): Set to > 1 to use beam search.
            alpha (float): Length penalty coefficient.
                Refer to https://arxiv.org/abs/1609.08144
                for more details.
            start_tokens (optional): An int Tensor of shape `[batch_size]`,
                containing the start tokens.
                Used when `decoding_strategy` = "infer_greedy" or
                "infer_sample", or `beam_width` > 1.
            end_token (optional): An int 0D Tensor, the token that marks end
                of decoding.
                Used when `decoding_strategy` = "infer_greedy" or
                "infer_sample", or `beam_width` > 1.
            softmax_temperature (optional): A float 0D Tensor, value to divide
                the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples. Must > 0. If `None`,
                1.0 is used.
                Used when `decoding_strategy="infer_sample"`.
            max_decoding_length (optional): An int scalar Tensor indicating
                the maximum allowed number of decoding steps.
                If `None` (default), use "max_decoding_length" defined in
                :attr:`hparams`. Ignored in "train_greedy" decoding.
            helper (optional): An instance of
                :tf_main:`Helper <contrib/seq2seq/Helper>` that defines the
                decoding strategy. If given, `decoding_strategy` is ignored.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. Controls dropout mode.
                If `None` (default), :func:`texar.global_mode`
                is used.

        Returns:

            - For **"train_greedy"** decoding, returns an instance of \
            :class:`~texar.modules.TransformerDecoderOutput` which contains\
            `sample_id` and `logits`.

            - For **"infer_greedy"** and **"infer_sample"** decoding, returns\
            a tuple `(outputs, sequence_lengths)`, where `outputs` is an \
            instance of :class:`~texar.modules.TransformerDecoderOutput` as\
            in "train_greedy", and `sequence_lengths` is a Tensor of shape\
            `[batch_size]` containing the length of each sample.

            - For **beam_search** decoding, returns a `dict` containing keys\
            "sample_id" and "log_prob".

                - **"sample_id"** is an int Tensor of shape \
                `[batch_size, max_time, beam_width]` containing generated\
                token indexes. `sample_id[:,:,0]` is the highest-probable \
                sample.
                - **"log_porb"** is a float Tensor of shape \
                `[batch_size, beam_width]` containing the log probability \
                of each sequence sample.
        """
        if memory_attention_bias is None:
            if memory_sequence_length is None:
                raise ValueError(
                    "`memory_sequence_length` is required if "
                    "`memory_attention_bias` is not given.")

            #enc_padding = 1 - mask_sequences(tf.ones_like(memory),
            #                                 memory_sequence_length,
            #                                 tensor_rank=3)[:, :, 0]
            enc_padding = 1 - tf.sequence_mask(
                memory_sequence_length, tf.shape(memory)[1], dtype=tf.float32)
            memory_attention_bias = attn.attention_bias_ignore_padding(
                enc_padding)

        if helper is not None:
            decoding_strategy = None

        if beam_width <= 1 and decoding_strategy == 'train_greedy':
            if sequence_length is not None:
                inputs = mask_sequences(inputs, sequence_length, tensor_rank=3)

            decoder_self_attention_bias = (
                attn.attention_bias_lower_triangle(
                    shape_list(inputs)[1]))
            target_inputs = inputs * self._hparams.dim**0.5

            _, lengths, channels = shape_list(target_inputs)
            pos_embeds = self.position_embedder(lengths, channels)

            inputs = target_inputs + pos_embeds

            decoder_output = self._self_attention_stack(
                inputs,
                memory,
                decoder_self_attention_bias=decoder_self_attention_bias,
                memory_attention_bias=memory_attention_bias,
                cache=None,
                mode=mode)
            logits = self.output_layer(decoder_output)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            output = TransformerDecoderOutput(
                logits=logits,
                sample_id=preds
            )
            rets = output

        else: # Inference-like decoding

            if max_decoding_length is None:
                max_decoding_length = self._hparams.max_decoding_length

            if beam_width <= 1:
                # Prepare helper
                if helper is not None:
                    pass
                elif decoding_strategy is not None:
                    if decoding_strategy == "infer_greedy":
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            self._embedding, start_tokens, end_token)
                    elif decoding_strategy == "infer_sample":
                        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                            self._embedding, start_tokens, end_token,
                            softmax_temperature)
                    else:
                        raise ValueError(
                            "Unknown decoding strategy: {}".format(
                                decoding_strategy))
                else:
                    raise ValueError(
                        "Must provide one of 'decoding_strategy' and 'helper'."
                        " We may implement helper hparams in the future.")
                self._helper = helper

                rets = self._infer_decoding(
                    decode_length=max_decoding_length,
                    memory=memory,
                    memory_attention_bias=memory_attention_bias,
                    decoding_strategy=decoding_strategy,
                )

            else:
                # The output format is different when running beam search
                sample_id, log_prob = self._beam_decode(
                    start_tokens,
                    end_token,
                    beam_width=beam_width,
                    alpha=alpha,
                    decode_length=max_decoding_length,
                    memory=memory,
                    memory_attention_bias=memory_attention_bias,
                )
                predictions = {
                    'sample_id':sample_id,
                    'log_prob': log_prob
                }
                rets = predictions

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return rets

    def _self_attention_stack(self,
                              inputs,
                              memory,
                              decoder_self_attention_bias=None,
                              memory_attention_bias=None,
                              cache=None,
                              mode=None):
        """Stacked multihead attention module.
        """
        inputs = tf.layers.dropout(inputs,
                                   rate=self._hparams.embedding_dropout,
                                   training=is_train_mode(mode))
        if cache is not None:
            memory_attention_bias = \
                cache['memory_attention_bias']
        else:
            assert decoder_self_attention_bias is not None

        x = inputs
        for i in range(self._hparams.num_blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    multihead_attention = \
                        self.multihead_attentions['self_att'][i]
                    selfatt_output = multihead_attention(
                        queries=layers.layer_normalize(x),
                        memory=None,
                        memory_attention_bias=decoder_self_attention_bias,
                        cache=layer_cache,
                        mode=mode,
                    )
                    x = x + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )
                if memory is not None:
                    with tf.variable_scope('encdec_attention'):
                        multihead_attention = \
                            self.multihead_attentions['encdec_att'][i]
                        encdec_output = multihead_attention(
                            queries=layers.layer_normalize(x),
                            memory=memory,
                            memory_attention_bias=memory_attention_bias,
                            mode=mode,
                        )
                        x = x + tf.layers.dropout(encdec_output, \
                            rate=self._hparams.residual_dropout, \
                            training=is_train_mode(mode))
                poswise_network = self.poswise_networks[i]
                with tf.variable_scope('past_poswise_ln'):
                    sub_output = tf.layers.dropout(
                        poswise_network(layers.layer_normalize(x)),
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )
                    x = x + sub_output

        return layers.layer_normalize(x)

    def _build_output_layer(self, dim):
        if self._hparams.embedding_tie:
            if self._hparams.output_layer_bias:
                with tf.variable_scope(self.variable_scope):
                    affine_bias = tf.get_variable(
                        'affine_bias', [self._vocab_size])
            else:
                affine_bias = None

            def _outputs_to_logits(outputs):
                shape = shape_list(outputs)
                outputs = tf.reshape(outputs, [-1, dim])
                logits = tf.matmul(outputs, self._embedding, transpose_b=True)
                if affine_bias is not None:
                    logits += affine_bias
                logits = tf.reshape(logits, shape[:-1] + [self._vocab_size])
                return logits

            return _outputs_to_logits
        else:
            layer = tf.layers.Dense(self._vocab_size, \
                use_bias=self._hparams.output_layer_bias)
            layer.build([None, dim])
            return layer

    def _init_cache(self, memory, memory_attention_bias):
        cache = {
            'memory': memory,
            'memory_attention_bias': memory_attention_bias,
        }
        batch_size = tf.shape(memory)[0]
        depth = self._hparams.multihead_attention.num_units
        for l in range(self._hparams.num_blocks):
            cache['layer_{}'.format(l)] = {
                'self_keys': tf.zeros([batch_size, 0, depth]),
                'self_values': tf.zeros([batch_size, 0, depth]),
                'memory_keys': tf.zeros([batch_size, 0, depth]),
                'memory_values': tf.zeros([batch_size, 0, depth]),
            }
        return cache

    def _infer_decoding(self,
                        decode_length,
                        memory,
                        memory_attention_bias,
                        decoding_strategy):
        """Performs inference-like (such as "infer_greedy" or "infer_sample")
        decoding.
        """
        batch_size = self._helper.batch_size
        output_size = TransformerDecoderOutput(
            logits=tf.TensorShape([self._vocab_size]),
            sample_id=self._helper.sample_ids_shape)
        output_dtype = TransformerDecoderOutput(
            logits=tf.float32,
            sample_id=self._helper.sample_ids_dtype)

        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, tf.TensorShape) or
                    from_shape.ndims == 0):
                return tf.TensorShape(None)
            else:
                batch_size = tf.contrib.util.constant_value(
                    tf.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tf.TensorShape([batch_size]).concatenate(from_shape)

        def _create_ta(s, d):
            return tf.TensorArray(
                dtype=d,
                size=0,
                dynamic_size=True,
                element_shape=_shape(batch_size, s))

        finished, inputs = self._helper.initialize()
        seq_length = tf.zeros_like(finished, dtype=tf.int32)
        step = tf.constant(0)
        outputs_ta = nest.map_structure(_create_ta, output_size, output_dtype)

        cache = self._init_cache(memory, memory_attention_bias)
        inputs_to_outputs_fn = self._inputs_to_outputs_fn(
            max_length=decode_length+1)

        def _body(step, finished, inputs, cache, outputs_ta, seq_length):
            outputs, cache = inputs_to_outputs_fn(inputs, step, cache)
            sample_ids = self._helper.sample(
                time=step, outputs=outputs, state=None)
            cur_finished, next_inputs, _ = self._helper.next_inputs(
                time=step,
                outputs=outputs,
                state=None,
                sample_ids=sample_ids)

            update_len = tf.logical_and(
                tf.logical_not(finished),
                cur_finished)
            seq_length = tf.where(
                update_len,
                tf.fill(tf.shape(seq_length), step+1),
                seq_length)

            finished |= cur_finished

            outputs_ta = nest.map_structure(
                lambda ta, out: ta.write(step, out),
                outputs_ta,
                TransformerDecoderOutput(
                    logits=outputs,
                    sample_id=sample_ids))

            return step+1, finished, next_inputs, cache, outputs_ta, seq_length

        def _not_finished(i, finished, *_):
            return (i < decode_length) & tf.logical_not(tf.reduce_all(finished))

        _, _, _, _, outputs_ta, seq_length = tf.while_loop(
            _not_finished,
            _body,
            loop_vars=(step, finished, inputs, cache, outputs_ta, seq_length),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                nest.map_structure(beam_search.get_state_shape_invariants,
                                   cache),
                nest.map_structure(lambda ta: tf.TensorShape(None), outputs_ta),
                tf.TensorShape([None])
            )
        )

        final_outputs = nest.map_structure(lambda ta: ta.stack(), outputs_ta)
        final_outputs = transpose_batch_time(final_outputs)

        return final_outputs, seq_length

    def _beam_decode(self,
                     start_tokens,
                     end_token,
                     memory,
                     memory_attention_bias,
                     decode_length=256,
                     beam_width=5,
                     alpha=0.6):
        cache = self._init_cache(memory, memory_attention_bias)
        inputs_to_outputs_fn = self._inputs_to_outputs_fn(decode_length+1)
        def symbols_to_logits_fn(ids, step, cache):
            return inputs_to_outputs_fn(
                self._prepare_tokens_to_embeds(ids[:, -1]), step, cache)

        outputs, log_prob = beam_search.beam_search(
            symbols_to_logits_fn,
            start_tokens,
            beam_width,
            decode_length,
            self._vocab_size,
            alpha,
            states=cache,
            eos_id=end_token)

        # Ignores <BOS>
        outputs = outputs[:, :, 1:]
        # shape = [batch_size, seq_length, beam_width]
        outputs = tf.transpose(outputs, [0, 2, 1])
        return (outputs, log_prob)
