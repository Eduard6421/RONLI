# RoNLI: A Romanian Natural Language Inference Corpus

Natural language inference (NLI), the task of recognizing the entailment relationship in sentence pairs, is an actively studied topic serving as proxy for natural language understanding. Despite the relevance of the task in building conversational agents and improving text classification, machine translation and other NLP tasks, to the best of our knowledge, there is no publicly available NLI corpus for the Romanian language.
We introduce the first Romanian NLI corpus which we release alongside the code employed to gather the dataset and the developed baselines. We release our code and dataset under the MIT license.

# License
Creative Commons Attribution Non Commercial Share Alike 4.0


## 📝 Table of Contents <a name = "tabel_of_contents"></a>

- [RoNLI: A Romanian Natural Language Inference Corpus](#ronli-a-romanian-natural-language-inference-corpus)
  - [📝 Table of Contents ](#-table-of-contents-)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Installing Prerequisites ](#installing-prerequisites-)
      - [Attention!](#attention)
  - [Usage ](#usage-)
  - [⛏️ Developed with ](#️-developed-with-)

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
    svc, logreg
'''

The flag subset is required for bert-cartography model and the available options are:
'''
    easy, ambiguous, hard, easyambiguous
'''



## ⛏️ Developed with <a name = "developed_with"></a>
- [Pytorch](https://pytorch.org/) - Deep Learning Library.
- [PytorchLightning](https://www.pytorchlightning.ai/index.html) - Pytorch Framework
- [HuggingfaceTransformers](https://huggingface.co/)- AI Model Repository 


|English |Romanian|Appearances|
|:----|:----|:----|
|Thus|Astfel|16245|
|As such|Prin urmare|5202|
|As a consequence|Ca urmare |4433|
|On the other hand|Pe de altă parte|1981|
|In consequence|În consecință|1010|
|Thus|Așadar|948|
|In contrast|În contrast|616|
|As a result|Drept urmare|601|
|In this way|În acest fel|574|
|In other words|Cu alte cuvinte|553|
|As a result|Ca rezultat|528|
|Because of this|Din această cauză|520|
|Meaning that|Adică|296|
|In spite of this fact|În ciuda acestui fapt|267|
|Such that|Astfel că|230|
|In conclusion|În concluzie|197|
|In essence|În esență|155|
|Said differently|Altfel spus|149|
|The result is|Rezultatul este|105|
|Meaning that|Asta înseamnă că|92|
|After all|În fond|53|
|In opposition|În opoziție|49|
|In contradiction|În contradicție|40|
|In result|În rezultat|36|
|Despite this|În ciuda acestui lucru|33|
|This leading to|Aceasta duce la |30|
|In spite of these facts|În ciuda acestor fapte|23|
|Because of this|Din această cauză|17|
|Concluding|Concluzionând|14|
|In summary|Sintetizând|13|
|Summarizing|Rezumând|12|
|On the other side|Pe de cealaltă parte|11|
|Despise those|În ciuda acestor lucruri|11|
|In contrary|În mod contrar|11|
|In summary|În rezumat|10|
|Nevertheless|Cu toate acestea însă|9|
|In simple terms|În termeni simpli|7|
|To finalize|Pentru a finaliza|7|
|In free translation|În traducere liberă|6|
|In short|Mai pe scurt|6|
|In more words|Pe larg|5|
|In contrast|Constrastând |5|
|Simplifying|Simplificând|5|
|In other terms|În alți termeni|5|
|In dissacord|În dezacord|4|
|As a consequence of this|Ca o consecință a acestui fapt|4|
|In antithesis|În antiteză|3|
|In a conclusion|Într-o concluzie|3|
|In an opposite sense|În sens opus|3|
|Simply said|Simplus spus|3|
|Leading to|Ducând la|2|
|In a contrast|Într-un contrast|2|
|More concise|Mai concis|2|
|Which lead to|Ceea ce a dus la |2|
|In contradiction|În contradictoriu|2|
|With this in mind|Ținând cont de acestea|1|
|Contrary to the beliefs|Contrar convingerilor|1|
|In another formulation|Într-o alta formulare|1|
|In popular terms|În termeni populari|1|
|Leading to|Conducând la|1|
|It can be concluded that|Se poate concluziona că|1|
|In spite of these|În pofida acestor lucruri|1|
|Leading to|Provocând astfel|1|

