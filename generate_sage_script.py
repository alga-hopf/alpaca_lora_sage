from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch
import argparse
import os

def evaluate(prompt_input, model, input_instr=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128):
	instruction_format = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n ### Instruction:\n{instruction}\n\n### Response:")
	example = {"instruction": prompt_input}
	prompt  = instruction_format.format_map(example)
	inputs = tokenizer(prompt, return_tensors="pt")
	input_ids = inputs["input_ids"].to(device)
	generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams)

	with torch.no_grad():
		generation_output = model.generate(input_ids=input_ids, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=max_new_tokens)

	s = generation_output.sequences[0]
	output = tokenizer.decode(s)
	output = output.split(tokenizer.eos_token)[0][3:]
	return output

def sage_code_from_output(output):
	code = output.split("### Response:")[1]
	code = code.splitlines()
	sage_code = ""
	for s in code:
		sage_code += s.strip() + "\n"
	with open('output_code.sage', 'w') as f:
		f.write(sage_code)
    	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--prompt', default='', type=str, help="User prompt")
	parser.add_argument('--checkpoint', required=True, help='lora checkpoint', type=str)
	args = parser.parse_args()

	prompt = args.prompt
	checkpoint_lora = args.checkpoint
	device = "cuda" if torch.cuda.is_available() else "cpu"

	checkpoint_base =  "decapoda-research/llama-7b-hf"
	checkpoint_lora = os.path.expanduser('~/alpaca_lora_sage/'+checkpoint)
	model_max_length = 512
	tokenizer = LlamaTokenizer.from_pretrained(checkpoint_base, model_max_length=model_max_length, padding_side="right", use_fast=False)
	DEFAULT_PAD_TOKEN = "[PAD]"
	DEFAULT_EOS_TOKEN = "</s>"
	DEFAULT_BOS_TOKEN = "<s>"
	DEFAULT_UNK_TOKEN = "<unk>"
	tokenizer.pad_token = DEFAULT_PAD_TOKEN
	tokenizer.eos_token = DEFAULT_EOS_TOKEN
	tokenizer.bos_token = DEFAULT_BOS_TOKEN
	tokenizer.unk_token = DEFAULT_UNK_TOKEN
	model = LlamaForCausalLM.from_pretrained(checkpoint_base, torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map={"": device})
	model = PeftModel.from_pretrained(model, checkpoint_lora, torch_dtype=torch.float32, device_map={"": device})

	model.eval();
	
	output = evaluate(prompt, model)
	sage_code_from_output(output)
