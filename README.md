# Alpaca-LoRA-SageMath

This repository contains the code to fine-tune the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) model with [low rank adaptation](https://github.com/tloen/alpaca-lora) on a dataset made of SageMath instructions. [SageMath](https://www.sagemath.org/) is a popular free open-source mathematical software, whose mission is ''Creating a viable free open source alternative to Magma, Maple, Mathematica and Matlab''. Following their mission, it would be helpful to have a language assistant specialized in the SageMath language for different purposes, like research, education, or as a free and open source plugin for mathematical computations.

## Generate the dataset
The dataset is made of couples instruction-output taken from the SageMath documentation, in the same format as the Stanford Alpaca dataset. The dataset is generated automatically scanning each page of the documentation, so some elements in the dataset might be inaccurate. To generate the dataset simply run 

```
python3 build_sage_dataset.py --dataset_name chosen_name_for_dataset
```
It takes approximately 30 minutes to generate the whole dataset, which has ~122311 elements. The full dataset `dataset.json` is already provided in the repository.

