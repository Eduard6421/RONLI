# RoNLI: A Romanian Natural Language Inference Corpus

Natural language inference (NLI), the task of recognizing the entailment relationship in sentence pairs, is an actively studied topic serving as proxy for natural language understanding. Despite the relevance of the task in building conversational agents and improving text classification, machine translation and other NLP tasks, to the best of our knowledge, there is no publicly available NLI corpus for the Romanian language.
We introduce the first Romanian NLI corpus which we release alongside the code employed to gather the dataset and the developed baselines. We release our code and dataset under the MIT license.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

## 📝 Table of Contents <a name = "tabel_of_contents"></a>

- [RoNLI: A Romanian Natural Language Inference Corpus](#ronli-a-romanian-natural-language-inference-corpus)
  - [📝 Table of Contents ](#-table-of-contents-)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Installing Prerequisites ](#installing-prerequisites-)
      - [Attention!](#attention)
  - [Usage ](#usage-)
  - [⛏️ Developed with ](#️-developed-with-)
  - [Citation ](#citation-)
  - [🎉 Acknowledgements ](#-acknowledgements-)

## About <a name = "about"></a>

This repository holds the dataset and baselines introduced in the paper "RoNLI: A Romanian Natural Language Inference Corpus" while also hosting our fork of the data-cartography method introduced by (Swayamdipta et al., 2020).


The project is structured as following:

    \RONLI
        \ bert - folder which contains the bert training pipeline
        \ bert_checkpoints - folder which hosts checkpoints during bert model training
        \ bert_stats - folder which hosts training dynamics saved during bert model training
        \ bert_subset - folder which contains the training pipeline for Bert on the easy/ambiguous/hard/easyambiguous subsets
        \ bert-cart-cl - folder which contains the trainig pipeline for bert-cart-cl
        \ bert-cart-cl++ - folder which contains the training pipeline for bert-cart-cl++
        \ bert-cart-stra-cl++ - folder which contains the training pipeline for bert-cart-stra-cl++
        \ bert-spurious - folder which contains the training pipeline for bert-spurious
        \ cartography - folder which contains our private fork of the data cartography by Swayadipta et al. 2020
        \ cbow - folder which contains the training pipeline for CBOW based models
        \ corpus - folder which contains scripts used to parse RO-WIKIPEDIA
        \ dataset - dataset holding various training subset as described in our paper
        \ gpt - folder which contains the training pipeline for RO-GPT2 model
        \ gpt_checkpoints - folder which hosts checkpoitns during RO-GPT2 model training
        \ nli_learning - folder which contains the implementation of the suggested baselines


## Getting Started <a name = "getting_started"></a>

### Installing Prerequisites <a name = "prerequisites"></a>

The environment dependencies can be found insitde the env.yml file. An dedicated env can be created automatically with the command
```
conda env create -f my_env.yml
```

#### Attention! 

CBOW have a dependency on the fastText libray. Due to no available pywheel for Windows we recommend building it locally. This can be easily done by running inside the conda environment the command:

```
bash prereq_install.sh
```


## Usage <a name="usage"></a>

The training and testing of the models is done solely one one command:  
```
    python -m main_script --model [model_name]  -subtype [model_subtype]  --subset [data_subset]
```

The available options for model name are:
'''
    cbow, gpt, bert, bert-spurious, bert-cartography, bert-cart-cl, bert-cart-cl++, bert-cart-stra-cl++
'''

The flag subtype is required for CBOW based models and the available options are:
'''
    svc, xgb
'''

The flag subset is required for bert-cartography model and the available options are:
'''
    easy, ambiguous, hard, easyambiguous
'''



## ⛏️ Developed with <a name = "developed_with"></a>
- [Pytorch](https://pytorch.org/) - Deep Learning Library.
- [PytorchLightning](https://www.pytorchlightning.ai/index.html) - Pytorch Framework
- [HuggingfaceTransformers](https://huggingface.co/)- AI Model Repository 

## Citation <a name="citation"></a>

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

We thank all researchers and data annotators behind the datasets and the developed baselines!
#   R O N L I  
 