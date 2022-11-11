# swit-debias
Switching [MASK] Tokens for Gender Debiasing in Pre-trained Language Models

## Installation
This repository is available in Ubuntu 20.04 LTS, and it is not tested in other OS.
```
git clone https://github.com/squiduu/swit-debias.git
cd swit-debias

conda create -n switdebias python=3.7.10
conda activate switdebias

pip install -r requirements.txt
```

## Bias mitigation
Fine-tune a pre-trained BERT to debias
```
mkdir ./out/
sh 
```
