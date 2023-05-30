import json
import numpy as np
import timeit
import copy
import torch
import sys
from transformers import Trainer, TrainingArguments, LlamaForCausalLM, LlamaTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq, get_scheduler, AdamW
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
import wandb
import os
from tqdm.auto import tqdm
import random
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training, set_peft_model_state_dict
from pynvml import *
import argparse


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")



if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--path_dataset', default="~/alpaca_lora_sage/dataset.json", type=str, help="dataset path")
	parser.add_argument('--wandb_project', default="sage finetuning", type=str, help="wandb project name")
	parser.add_argument('--model_max_length', default=512, type=int, help="max length accepted by the model")
	parser.add_argument('--batch_size', default=128, type=int, help="model batch size")
	parser.add_argument('--perc_dataset_size', default=1., type=float, help="percentage of dataset for training")
	parser.add_argument('--micro_batch_size', default=8, type=int, help="batch size per GPU for training")
	parser.add_argument('--lora_r', default=8, type=int, help="LoRA rank")
	parser.add_argument('--lora_alpha', default=16, type=int, help="LoRA alpha scaling factor")
	parser.add_argument('--lora_dropout', default=0.05, type=float, help="Dropout for LoRA")
	parser.add_argument('--num_epochs', default=3, type=int, help="Number of epochs")
	parser.add_argument('--learning_rate', default=3e-4, type=float, help="Learning rate")
	parser.add_argument('--group_by_length', default=False, type=bool, help="Whether to group by length")
	parser.add_argument('--wandb_run_name', default="", type=str, help="wandb run name")
	parser.add_argument('--wandb_entity_name', default="", type=str, help="wandb entity name")
	parser.add_argument('--wandb_key', default="", type=str, help="wandb key name")
	parser.add_argument('--resume_from_checkpoint', default=None, type=None, help="Whether to resume from checkpoint")
	parser.add_argument('--use_wandb', default=True, type=bool, help="Whether to log on wandb")
	parser.add_argument('--out_dir', default="~/alpaca_lora_sage/", type=str, help="Directory where to save the model")
	parser.add_argument('--logging_steps', default=1, type=int, help="Number of steps for logging")
	parser.add_argument('--model_name', default="alpaca_lora_sage", type=str, help="How to save the pretrained model")
	
	args = parser.parse_args()
	print(args)
	
	path_dataset = os.path.expanduser(args.path_dataset)
	wandb_project = args.wandb_project
	model_max_length = args.model_max_length 
	batch_size = args.batch_size 
	micro_batch_size = args.micro_batch_size   
	lora_r = args.lora_r 
	lora_alpha = args.lora_alpha 
	lora_dropout = args.lora_dropout 
	num_epochs = args.num_epochs 
	learning_rate = args.learning_rate 
	group_by_length = args.group_by_length 
	wandb_run_name = args.wandb_run_name 
	resume_from_checkpoint = args.resume_from_checkpoint 
	use_wandb = args.use_wandb
	out_dir = os.path.expanduser(args.out_dir)	
	logging_steps = args.logging_steps
	model_name = args.model_name
	wandb_entity = args.wandb_entity_name
	wandb_key = args.wandb_key
	perc_dataset = args.perc_dataset_size
	

	""" Load and inspect dataset"""

	with open(path_dataset, 'r') as f:
		raw_dataset = json.load(f)

	print("Number of elements in training dataset:", len(raw_dataset))
	print("")

	"""# Fine tuning"""

	if use_wandb:
		wandb.login(key=wandb_key)
		os.environ["WANDB_PROJECT"] = wandb_project
		os.environ["WANDB_ENTITY"] = wandb_entity

	world_size = int(os.environ.get("WORLD_SIZE", 1))
	lora_target_modules = ["q_proj", "v_proj"]
	ddp = world_size != 1

	tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", model_max_length=model_max_length, padding_side="right", use_fast=False)
	DEFAULT_PAD_TOKEN = "[PAD]"
	DEFAULT_EOS_TOKEN = "</s>"
	DEFAULT_BOS_TOKEN = "<s>"
	DEFAULT_UNK_TOKEN = "<unk>"
	tokenizer.pad_token = DEFAULT_PAD_TOKEN
	tokenizer.eos_token = DEFAULT_EOS_TOKEN
	tokenizer.bos_token = DEFAULT_BOS_TOKEN
	tokenizer.unk_token = DEFAULT_UNK_TOKEN
	IGNORE_INDEX = -100

	PROMPT_DICT = {
		"prompt_input": (
		    "Below is an instruction that describes a task, paired with an input that provides further context. "
		    "Write a response that appropriately completes the request.\n\n"
		    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
		),
		"prompt_no_input": (
		    "Below is an instruction that describes a task. "
		    "Write a response that appropriately completes the request.\n\n"
		    "### Instruction:\n{instruction}\n\n### Response:"
		),
	}
	list_data_dict = raw_dataset
	prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
	sources_list = [prompt_no_input.format_map(example) for example in list_data_dict]
	targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

	pop_size = int(len(raw_dataset) * perc_dataset)
	order = list(range(pop_size))
	random.shuffle(order)
	examples_list = [s + t for s, t in zip(sources_list, targets)]
	examples, sources = [], []
	
	for n in order:
	  examples.append(examples_list[n])
	  sources.append(sources_list[n])
	full_examples = {}
	all_examples, all_sources = [], []
	
	for i in range(len(examples)):
		all_examples.append(examples[i])
		all_sources.append(sources[i])
	full_examples["example"] = all_examples
	full_examples["source"] = all_sources

	full_examples_dataset = Dataset.from_dict(full_examples)

	def tokenize_function(example):
		data_dict = tokenizer(example["example"], padding="longest", max_length=model_max_length, truncation=True) 
		tokenized_source = tokenizer(example["source"], padding="longest", max_length=model_max_length, truncation=True) 
		data_dict["labels"] = [IGNORE_INDEX] * len(tokenized_source["input_ids"]) + data_dict["input_ids"][len(tokenized_source["input_ids"]):]
		return data_dict

	train_dataset = full_examples_dataset.map(tokenize_function)

	train_dataset = train_dataset.remove_columns(["example", "source"])

	data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

	gradient_accumulation_steps = batch_size // micro_batch_size
	device_map = "auto"
	
	if ddp:
		device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

	model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)

	model = prepare_model_for_kbit_training(model)
	config = LoraConfig(
		r=lora_r,
		lora_alpha=lora_alpha,
		target_modules=lora_target_modules,
		lora_dropout=lora_dropout,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, config)

	val_set_size = 0
	gradient_accumulation_steps = batch_size // micro_batch_size

	arguments = TrainingArguments(
		    per_device_train_batch_size=micro_batch_size,
		    gradient_accumulation_steps=gradient_accumulation_steps,
		    warmup_steps=100,
		    num_train_epochs=num_epochs,
		    learning_rate=learning_rate,
		    fp16=True,
		    logging_steps=logging_steps,
		    optim="adamw_torch",
		    evaluation_strategy="no",
		    save_strategy="no", # saving requires too much gpu memory
		    output_dir=out_dir,
		    save_total_limit=3,
		    load_best_model_at_end=True if val_set_size > 0 else False,
		    ddp_find_unused_parameters=False if ddp else None,
		    group_by_length=group_by_length,
		    report_to="wandb" if use_wandb else None,
		    run_name=wandb_run_name if use_wandb else None,
		)

	trainer = Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=None,
		args=arguments,
		data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
	)
	model.config.use_cache = False

	old_state_dict = model.state_dict
	model.state_dict = (
		lambda self, *_, **__: get_peft_model_state_dict(
		    self, old_state_dict()
		)
	).__get__(model, type(model))

	if torch.__version__ >= "2" and sys.platform != "win32":
		model = torch.compile(model)

	print("Begin training")
	trainer.train(resume_from_checkpoint=resume_from_checkpoint)
	print("End training")

	print_gpu_utilization()

	#model.eval()

	model.save_pretrained(out_dir+model_name)

