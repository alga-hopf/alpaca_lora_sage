# Alpaca-LoRA-SageMath

This repository contains the code to fine-tune the [LLama model](https://huggingface.co/docs/transformers/main/en/model_doc/llama) [1] with [low rank adaptation and 8-bit quantization](https://github.com/tloen/alpaca-lora) [2, 3] on a dataset made of SageMath instructions. The dataset is generated following the [Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) model. [SageMath](https://www.sagemath.org/) is a popular free open-source mathematical software, whose mission is ''Creating a viable free open source alternative to Magma, Maple, Mathematica and Matlab''. Following their mission, it would be helpful to have a language assistant specialized in the SageMath language for different purposes, like research, education, or as a free and open source plugin for mathematical computations.

Both [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) are licensed under the Apache-2.0 license.

## Requirements
The requirements for fine-tuning the LLaMA model and generating new output can be found in the `requirements.txt` file. Note that Hugging Face `transformers` and `peft` should be installed from the github repo as follows:
```
pip3 install git+https://github.com/huggingface/transformers
pip3 install git+https://github.com/huggingface/peft
```

To execute the SageMath code you need to have [SageMath](https://www.sagemath.org/) installed.

## Generate the dataset
The dataset is made of couples instruction-output taken from the SageMath documentation, in the same format as the Stanford Alpaca dataset. The dataset is generated automatically scanning each page of the documentation, so some elements in the dataset might be inaccurate. To generate the dataset simply run 

```
python3 build_sage_dataset.py --dataset_name chosen_name_for_dataset
```
It takes approximately 30 minutes to generate the whole dataset, which has ~122311 elements. The full dataset `dataset.json` is already provided in the repository.

## Fine-tune
The fine-tuning procedure mimics the one of the 8-bit low rank adaptation from [this repo](https://github.com/tloen/alpaca-lora).
There are two ways for fine-tuning the LLaMA model on the SageMath dataset. One is through the colab notebook `sage_finetuning_github.ipynb`. The other is by running 
```
python3 sage_finetuning.py
```
with the following arguments:
- `--path_dataset`: the path to the SageMath dataset. Default is "~/alpaca_lora_sage/dataset.json".
- `--wandb_project`: the name of your wandb project. Default is "".
- `--model_max_length`: max length accepted by the model. Default is $512$.
-	`--batch_size`: model batch size. Default is $128$.
-	`--micro_batch_size`: the batch size per GPU for training. Default is $8$. 
-	`--lora_r`: LoRA rank. Default is $8$.
-	`--lora_alpha`: LoRA $\alpha$ scaling factor. Default is $16$.
-	`--lora_dropout`: dropout for LoRA. Default is $0.05$.
-	`--num_epochs`: number of training epochs. Default is $3$.
- `--learning_rate`: learning rate. Default is $3e-4$.
-	`--group_by_length`: whether to group input by length. Default is `False`.
-	`--wandb_run_name`: the name of the wandb run. Default is "".
-	`--resume_from_checkpoint`: whether to resume training from checkpoint. Default is `None`, otherwise is the path to checkpoint.
-	`--use_wandb`: whether to report the experiment to wandb. Default is `True`.
-	`--out_dir`: chosen output directory. Default is "~/alpaca_lora_sage/"	
-	`--logging_steps`: number of logging steps. Default is $1$.
-	`--model_name`: how to save the pretrained model. Default is "alpaca_lora_sage".

## Generate new output
There are two ways to generate new output. One is through the notebook `generate_sage_github.ipynb`, and the other one is through the script `generate_sage.py`. In the first case it suffices to execute each cell until the user prompt is required. Write your prompt and keep executing the cells until the end. In the notebook there is also the option to produce a sage script with the output of the model. 
The second way of generating new output is by running
```
python3 generate_sage.py
```
with the following arguments:
- `--path_model`: path to the fine-tuned model. Default is "~/alpaca_lora_sage/alpaca_lora_sage".
- `--out_dir`: chosen output directory. Default is "~/alpaca_lora_sage/".

The user will be required to write their own output. After that the model will output the new generated result. To interrupt just type `exit()` in your prompt.

## Generate and execute new output
**Note**: if the output code is not correct this will not work.

A way to generating and executing new SageMath code is by running
```
bash generate_sage_results.sh
```
The user will be asked to write the checkpoint name and the input prompt. After that the code will produce and run a sage script with the output and print the result in the shell. 

## To do
There are many things to improve:
- Improve the dataset: the dataset is taken from the SageMath documentation, but it lacks instructions and outputs from users that use SageMath. Including such examples could dramatically improve the generating performance of the model. If you feel comfortable, please include some of your code in the following format in the text file `new_examples_from_users.txt.
```
{"instruction": "your instruction",
"input": "",
"output": "the output you get"}
```
- Try other quantization, like 4-bit, and different LLaMA models.
- Improve the execution of the SageMath code, e.g., append multiple outputs to the same `.sage` file and execute the final script.

## References
[1] [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971v1). Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. 
[2] [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
[3] [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer

