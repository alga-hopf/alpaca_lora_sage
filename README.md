# Alpaca-LoRA-SageMath

This repository contains the code to fine-tune the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) model with [low rank adaptation](https://github.com/tloen/alpaca-lora) on a dataset made of SageMath instructions. [SageMath](https://www.sagemath.org/) is a popular free open-source mathematical software, whose mission is ''Creating a viable free open source alternative to Magma, Maple, Mathematica and Matlab''. Following their mission, it would be helpful to have a language assistant specialized in the SageMath language for different purposes, like research, education, or as a free and open source plugin for mathematical computations.

## Generate the dataset
The dataset is made of couples instruction-output taken from the SageMath documentation, in the same format as the Stanford Alpaca dataset. The dataset is generated automatically scanning each page of the documentation, so some elements in the dataset might be inaccurate. To generate the dataset simply run 

```
python3 build_sage_dataset.py --dataset_name chosen_name_for_dataset
```
It takes approximately 30 minutes to generate the whole dataset, which has ~122311 elements. The full dataset `dataset.json` is already provided in the repository.

## Fine-tune

The fine-tuning procedure mimics the one of the 8-bit low rank adaptation, and it is contained in the colab notebook `sage_finetuning_github.ipynb`.

## Generate new output
There are two ways to generate new output.

## To do
There are many things to improve:
- Improve the dataset: the dataset is taken from the SageMath documentation, but it lacks instructions and outputs from users that use SageMath. Including such examples could dramatically improve the generating performance of the model. If you feel comfortable, please include some of your code in the format
```
{"instruction": "your instruction",
"input": "",
"output": "the output you get"}
```
in the text file `new_examples_from_users.txt`.
- Try other quantization, like 4-bit, and different LLaMA models.

## Requirements



