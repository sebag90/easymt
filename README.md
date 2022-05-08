# EASYMT

## Description
Easymt is a small library for neural machine translation and language modeling created to use as little dependencies as possible. The core library to train models (```easymt.py```) only needs pytorch and text files. This choice was made to reduce as much as possible the risk of libraries changing their APIs and break code. Moreover, features like batched data sets and gradient accumulation were implemented to train models on low resource machines with as little as 4GB of ram (tested on a Nvidia Jetson Nano).

For instructions on how to use easymt please refer to the ![wiki](https://github.com/sebag90/easymt/wiki)
