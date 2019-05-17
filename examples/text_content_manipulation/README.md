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
python3 ie.py --gold_file nba_data/gold.${STAGE}.txt --ref_file nba_data/nba.sent_ref.${STAGE}.txt ${EXPR_NAME}/ckpt/hypo*.test.txt --gpuid 0
```

which uses GPU 0 to run IE models for all `${EXPR_NAME}/ckpt/hypo*.test.txt`. `${STAGE}` can be val or test depending on which stage you want to evaluate. The result will be appended to `${EXPR_NAME}/ckpt/ie_results.${STAGE}.txt`, in which the columns represents training steps, $\textrm{BLEU}(\hat{y}, y')$, IE precision, IE recall, simple precision and simple recall (you don't have to know what simple precision/recall is), respectively.

## evaluate Content scores

After trained your model, you may want to evaluate two content scores via Bert classifier. This part is devised from [the Texar implementation of BERT](https://github.com/asyml/texar/tree/master/examples/bert#use-other-datasetstasks). To evaluate the content fidelity, we simply concatenate each record of `x/x'` with `y` and classify whether `y` express the record. In this way, we construct the data in `../bert/E2E` to train the Bert classifier. 

Run the following cmd to prepare data for evaluation:

```bash
python3 prepare_data.py --expr_name ${EXPR_NAME} --step ${step}
```
which process the previous `${EXPR_NAME}/ckpt/hypos${step}.valid.txt` into the above `x | y` fomat in TFRecord data files.

Then, run the following command to compute the two content scores:

```bash
python bert_classifier_main.py  --do_test --config_data=config_data
```
