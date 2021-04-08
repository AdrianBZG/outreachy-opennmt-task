# Task for Outreachy project "Migration of natural language query translation code to OpenNMT 2.0"

## 1. Objective

To become more familiar with OpenNMT, the library that we will be using in the project ["Migration of natural language query translation code to OpenNMT 2.0"](https://www.outreachy.org/outreachy-may-2021-internship-round/communities/intermine/#migration-of-natural-language-query-translation-co), we encourage the applicants to develop a small project outlined below. This project will be taken into account towards contribution. The procedure to share it with mentors is also explained below.

## 2. Task specification

### 2.1 Goal

In this task, we will be creating a small Python application that makes use of [OpenNMT](https://opennmt.net/OpenNMT-py/index.html) to translate from a source language to a target language. For this, the applicant is encouraged to choose one of the datasets from the EMNLP 2017 conference available in the [following link](http://www.statmt.org/wmt17/translation-task.html#download). Since we need parallel data (source->target), the dataset chosen would need to be from the "Parallel data" category. The applicant is encouraged to use a small dataset so that it is easier to handle. A good example could be the "Wiki Headlines" [(link)](http://www.statmt.org/wmt15/wiki-titles.tgz) datasets, which provides headlines from Wikipedia in Finnish and Russian (source language - preferably use the Finnish one to avoid having to deal with Cyrillic characters) and their translation to English (target language).

To create the application the applicant needs to read through the [Get started](https://opennmt.net/OpenNMT-py/examples/Translation.html) documentation from OpenNMT and understand the main three components that make up an OpenNMT-based application: data pre-processing, model training and translation. The applicant is encouraged to follow an object-oriented programming approach and a sensible choice of software design patterns [(link)](https://www.oodesign.com/) to develop the application, so that it is as modular and flexible as possible. As an example, it should be easy enough to modify the different parameters that are used throughout the application: data directory, Transformer model parameters (number of layers, number of heads), usage of Beam search or other mechanisms and their related parameters, etc.

The output of this task is an application that 1) can pre-process the selected datasets (either from the EMNLP 2017 or from somewhere else), 2) trains a Transformer translation model, 3) translates a given input sentence into its corresponding target language sentence and 4) evaluates how well the model performs by having a hold-out test dataset.

Also, to allow to query the model from external scripts, the model should be deployed as a REST server using the [OpenNMT Server script](https://opennmt.net/OpenNMT-py/options/server.html).

### 2.2 Delivery and evaluation

The applicant should fork this repository and develop the application, then create a pull request explaining the approach and considerations that were taken to fulfil the task.

## 3. Resources

Some resources to get started with OpenNMT-py (PyTorch version):

1. Introduction to the "general theory" (Neural Machine Translation or sequence-to-sequence models):

   1.1 From Google AI blog: [https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html](https://ai.googleblog.com/2016/09/a-neural-network-for-machine.html)

   1.2 From Google AI blog x2: [https://ai.googleblog.com/2017/04/introducing-tf-seq2seq-open-source.html](https://ai.googleblog.com/2017/04/introducing-tf-seq2seq-open-source.html)

   1.3 Another simple introduction: [https://www.analyticsvidhya.com/blog/2020/08/a-simple-introduction-to-sequence-to-sequence-models/](https://www.analyticsvidhya.com/blog/2020/08/a-simple-introduction-to-sequence-to-sequence-models/)

2. The library documentation and "get started" walkthrough: [https://opennmt.net/OpenNMT-py/](https://opennmt.net/OpenNMT-py/)

3. The OpenNMT-py GitHub repo, that contains some examples in the "/examples" directory: [https://github.com/OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

4. Tutorials available in the library forum: [https://forum.opennmt.net/c/tutorials/10](https://forum.opennmt.net/c/tutorials/10)

## 4. Questions

If you have any questions, you can contact the main mentors (please CC all in your email):

- Adrián Rodríguez-Bazaga (ar989@cam.ac.uk)
- Rachel Lyne (rachel@intermine.org)
