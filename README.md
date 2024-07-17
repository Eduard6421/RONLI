# A Novel Cartography-Based Curriculum Learning Method Applied on RoNLI: The First Romanian Natural Language Inference Corpus (ACL 2024) - Official Repository

Natural language inference (NLI), the task of recognizing the entailment relationship in sentence pairs, is an actively studied topic serving as proxy for natural language understanding. Despite the relevance of the task in building conversational agents and improving text classification, machine translation and other NLP tasks, to the best of our knowledge, there is no publicly available NLI corpus for the Romanian language. We introduce the first Romanian NLI corpus which we release alongside the code employed to gather the dataset and the developed baselines. We release our code and dataset under the CC-NA-SA 4.0 license.

We also propose a curriculum learning method based on data cartography that uses a novel difficulty scoring function that jointly takes confidence and variability into account. By design, our novel scoring function assigns low scores to items characterized by low variability and high correctness, medium scores for items perceived as ambiguous, and high scores to difficult examples. Our approach employs stratified easy-to-hard batches. This ensures the diversity of class labels right from the beginning of the training process, avoiding to bias the model towards certain classes. 

# License
The dataset and code are released under: [Creative Commons Attribution Non Commercial Share Alike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

## 📝 Table of Contents <a name = "tabel_of_contents"></a>

  - [Citation ](#citation-)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
  - [Usage ](#usage-)
  - [Developed with ](#️-developed-with-)

## Citation <a name = "citation"></a>

Please cite our work if you use any material released in this repository.
```
@InProceedings{Poesina-ACL-2024,
  author    = {Poesina, Eduard Gabriel and Caragea, Cornelia and Ionescu, Radu Tudor},
  title     = "{A Novel Cartography-Based Curriculum Learning Method Applied on RoNLI: The First Romanian Natural Language Inference Corpus}",
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  publisher = {Association for Computational Linguistics},
  year      = {2024},
  }
```

## About <a name = "about"></a>

This repository holds the dataset and baselines introduced in the paper "A Novel Cartography-Based Curriculum Learning Method Applied on RoNLI: The First Romanian Natural Language Inference Corpus" accepted at ACL 2024. The repository also hosts our fork of the data-cartography method introduced by (Swayamdipta et al., 2020).

The project is structured as follows:

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

The environment dependencies can be found insitde the env.yml file. An dedicated env can be created automatically with the command:
```
conda env create -f my_env.yml
```

#### Attention! 

CBOW have a dependency on the fastText library. Due to no available pywheel for Windows we recommend building it locally. This can be easily done by running inside the conda environment the command:

```
bash prereq_install.sh
```


## Usage <a name="usage"></a>

The training and testing of the models is performed with one command:  
```
    python -m main_script --model [model_name]  -subtype [model_subtype]  --subset [data_subset]
```

The available options for model name are:

```
    cbow, gpt, bert, bert-spurious, bert-cartography, bert-cart-cl, bert-cart-cl++, bert-cart-stra-cl++
```

The flag subtype is required for CBOW based models and the available options are:
```
    svc, logreg
```

The flag subset is required for bert-cartography model and the available options are:
```
    easy, ambiguous, hard, easyambiguous
```



## ⛏️ Developed with <a name = "developed_with"></a>
- [Pytorch](https://pytorch.org/) - Deep Learning Library.
- [PytorchLightning](https://www.pytorchlightning.ai/index.html) - Pytorch Framework
- [HuggingfaceTransformers](https://huggingface.co/)- AI Model Repository 


