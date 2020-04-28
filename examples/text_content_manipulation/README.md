# Text Content Manipulation

## Prerequisites

Before running the commands, you've to ensure Python3 is installed.

Run the following commands:

```bash
make
pip3 install -r requirements.txt
```

You also need to install [Texar](https://github.com/asyml/texar), a newly released NLP toolkits:

```bash
git clone https://github.com/asyml/texar.git
cd texar && pip3 install -e .
```

### For IE

If you'd like to evaluate IE after training, you have to ensure Lua Torch is installed, and download the IE models from [here](https://drive.google.com/file/d/1hV8I9tvoL3943OqqPkLFIbTYfFSqsV1e/view?usp=sharing) then unzip the files directly under the directory `data2text/`.

## Run

The following command illustrates how to run an experiment:

```bash
python3 manip.py --attn_x --attn_y_ --copy_x --rec_w 0.8 --expr_name ${EXPR_NAME}
```

Where `${EXPR_NAME}` is the directory you'd like to store all the files related to your experiment, e.g. `my_expr`.

Note that the code will automatically restore from the previously saved latest checkpoint if it exists.

You can start Tensorboard in your working directory and watch the curves and $\textrm{BLEU}(\hat{y}, y')$.

## evaluate IE scores

After trained your model, you may want to evaluate IE (Information Retrieval) scores. The following command illustrates how to do it:

```bash
CUDA_VISIBLE_DEVICES=${GPUIDS}$ python3 ie.py --gold_file nba_data/gold.${STAGE}.txt --ref_file nba_data/nba.sent_ref.${STAGE}.txt ${EXPR_NAME}/ckpt/hypo*.test.txt
```

which needs about 5 GPUS to run IE models for all `${EXPR_NAME}/ckpt/hypo*.test.txt`. `${STAGE}` can be val or test depending on which stage you want to evaluate. The result will be appended to `${EXPR_NAME}/ckpt/ie_results.${STAGE}.txt`, in which the columns represents training steps, $\textrm{BLEU}(\hat{y}, y')$, IE precision, IE recall, simple precision and simple recall (you don't have to know what simple precision/recall is), respectively.

## evaluate Content scores

After trained your model, you may want to evaluate two content scores via Bert classifier. This simplified model is devised from [the Texar implementation of BERT](https://github.com/asyml/texar/tree/master/examples/bert#use-other-datasetstasks). To evaluate the content fidelity, we simply concatenate each record of `x` or `x'` with `y` and classify whether `y` express the record. In this way, we construct the data in `../bert/E2E` to train the Bert classifier. 

### Prepare data
Run the following cmd to prepare data for evaluation:

```bash
python3 prepare_data.py --expr_name ${EXPR_NAME} --step ${step}
[--max_seq_length=128]
[--vocab_file=bert_config/all.vocab.txt]
[--tfrecord_output_dir=bert/E2E] 
```
which processes the previous `${EXPR_NAME}/ckpt/hypos${step}.valid.txt` into the above mentioned `x | y` fomat in TFRecord data files. Here:

* `max_seq_length`: The maxium length of sequence. This includes BERT special tokens that will be automatically added. Longer sequence will be trimmed.
* `vocab_file`: Path to a vocabary file used for tokenization. 
* `tfrecord_output_dir`: The output path where the resulting TFRecord files will be put in. Be default, it is set to `bert/E2E`.


### Restore and evaluate
We provide a pretrained transformer classifier model [link](https://drive.google.com/drive/folders/1jNaJ_R_f89G8xbAC8iwe49Yx_Z-LXr0i), which achieves 92% accuracy on the test set. Make sure that the pretrained model is put into the `bert/classifier_ckpt/ckpt` directory. Before the evaluation for content, remember to modify the file name of `config_data.py` manually. Then, run the following command to restore and compute the two content scores:

```bash
cd bert/
python3 bert_classifier_main.py  --do_pred --config_data=config_data --checkpoint=classifier_ckpt/ckpt/model.ckpt-13625
[--output_dir=output_dir/]
```
The cmd prints the two scores and the output is by default saved in `output/results_*.tsv`, where each line contains the predicted label for each instance.
