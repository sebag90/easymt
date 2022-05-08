# EASYMT

## Description
Easymt is a small library for neural machine translation and language modeling created to use as little dependencies as possible. The core library to train models (```easymt.py```) only needs pytorch and text files. This choice was made to reduce as much as possible the risk of libraries changing their APIs and break code. Moreover, features like batched data sets and gradient accumulation were implemented to train models on low resource machines with as little as 4GB of ram (tested on a Nvidia Jetson Nano).

For instructions on how to use easymt please refer to the [wiki](https://github.com/sebag90/easymt/wiki)


## Quickstart


### 1) Download data
After [installing easymt](https://github.com/sebag90/easymt/wiki#install), you can now download some [data](https://www.kaggle.com/datasets/sebag90/mt-ita-en), create the directory ```easymt/data``` and copy the downloaded text files there.


### 2) Prepare a configuration file
At this point you need to define your model in the configuration file according to the [available options](https://github.com/sebag90/easymt/wiki/EASYMT#configuration-file).  
This configuration, for example, will train an english to italian transformer model with 6 layers with size 512 on both the encoder and decoder side. Feel free to play around with the configuration file to test new models.

```
[DATA]
src_train = data/train.en
tgt_train = data/train.it
src_eval = data/eval.en
tgt_eval = data/eval.it
src_vocab = data/vocab.en
tgt_vocab = data/vocab.it

[MODEL]
task = translation
type = transformer
source = en
target = it
encoder_layers = 6
decoder_layers = 6
max_length = 256
uniform_init = 0.1
shared_embedding = False

[RNN]
hidden_size = 300
word_vec_size = 256
attention = general
bidirectional = True
rnn_dropout = 0.3
attn_dropout = 0.1
input_feed = True

[TRANSFORMER]
d_model = 512
heads = 16
dim_feedforward = 2048
attn_dropout = 0.1
residual_dropout = 0.1

[TRAINING]
print_every = 10
steps = 100000
batch_size = 4
step_size = 32
optimizer = noam
learning_rate = 0.001
lr_reducing_factor = 0.5
noam_factor = 2
warm_up = 6000
patience = 2
save_every = 10000
teacher = True
teacher_ratio = 0.5
valid_steps = 20000
label_smoothing = 0.1
gradient_clipping = 5
```

### 3) train your model

```
python easymt.py train config.ini
```

During training easymt will save the trained models at the specified steps in the directory ```easymt/checkpoints```.  
Depending on your hardware, this step can take A VERY long time, you might consider running it as:
```
nohup python easymt.py train config.ini &
```
this way the training will run in the background and the output will be written to ```nohup.out```.
To check on how the model is training, use: ```tail -f nohup.out```

### 4) test your model

Once you trained a model you can test this on the newssyscomb evaluation files.
First translate a document:
```
python easymt.py translate data/newssyscomb.2009.processed.en checkpoints/transformer_en-it_100000.pt
```

This step will generate the file ```data/newssyscomb.2009.processed.translated.it``` which is still encoded or tokenized according to the preprocessing steps we used during data preparation. To undo this, use ```texter.py```:

```
python texter.py normalize --SP 35000 data/newssyscomb.2009.processed.translated.it
```

which will produce ```data/newssyscomb.2009.processed.translated.normalized.it```.  
Now we can take a look at the translations of our model:
```
source: Both countries invested millions of dollars into surveying.
translation: Entrambi i paesi hanno investito milioni di dollari nell’indagine.
gold: Tutti e due i paesi hanno investito nella ricerca milioni di dollari.
```

```
source: In a dramatic campaign, supporters had made desperate attempts to convince the critics of the 700 billion dollar plan's merits.
translation: In una campagna strana, i sostenitori hanno fatto dei tentativi di riflettere le critiche del piano di bilancio 700 miliardi di dollari.
gold: In una azione drammatica i sostenitori hanno tentato disperatamente, di persuadere i critici dal pesante pacco di 700 mld di dollari.
```
