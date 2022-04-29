# AI_Center_Project

Project repo containing the code developed for the course " AI Center Projects in Machine Learning Research" held at ETH Zurich in Spring 2022

## Table of Contents
- [How to run](#How-to-run)
- [Supported datasets](#Supported-datasets)
- [Supported KG models](#Supported-models)
- [Supported Language models](#Supported-models)

## How to run

**Step 1:** Clone the repository:
```console
git clone https://github.com/LucaMalagutti/AI_Center_Project
```

**Step 2:** Copy new .env file and modify it by adding your environment variables, you can obtain your wandb key at https://wandb.ai/settings:
```console
cp .env.tmp .env 
vim .env 
```

Example of ``.env`` file:

```console
WANDB_API_KEY = Your Key
```

**Step 3:** Create virtual environment called ``venv`` and install packages: 
```console
python -m venv venv
pip install -r requirements.txt
```

**Step 4:** Download Emdeddings file and save it in ``./word_vectors/``.
TODO explain better this step.

**Step 5:** Train the model with your embeddings. Example of how to train:
```console
python3 train.py --dataset=wn18rr --model=tucker --init=w2v
```

## Supported datasets

| Name                               | Documentation                                                                                                       | Citation                                                                                                                |   Entities |   Relations |   Triples |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|------------|-------------|-----------|
| WN18RR                    | [`pykeen.datasets.WN18RR`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.WN18RR.html)                 | [Toutanova *et al*., 2015](https://www.aclweb.org/anthology/W15-4007/)                                                  |      40559 |          11 |     92583 |
| FB15k237                          | [`pykeen.datasets.FB15k237`](https://pykeen.readthedocs.io/en/latest/api/pykeen.datasets.FB15k237.html)             | [Toutanova *et al*., 2015](https://www.aclweb.org/anthology/W15-4007/)                                                  |      14505 |         237 |    310079 |

## Supported KG models

| Name                           | Model                                                                                                                         | Interaction                                                                                                                                | Citation                                                                                                                |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| TuckER                         | [`pykeen.models.TuckER`](https://pykeen.readthedocs.io/en/latest/api/pykeen.models.TuckER.html)                               | [`pykeen.nn.TuckerInteraction`](https://pykeen.readthedocs.io/en/latest/api/pykeen.nn.TuckerInteraction.html)                              | [Balažević *et al.*, 2019](https://arxiv.org/abs/1901.09590)                                                            |                                                    |

## Supported Language models

| Name                           | Citation                                                                                                                         |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Word2Vec                         | [Mikolov *et al.*, 2013](https://arxiv.org/pdf/1301.3781.pdf)                               |
| GloVe                         | [Pennington *et al.*, 2014](https://aclanthology.org/D14-1162.pdf)                               |

