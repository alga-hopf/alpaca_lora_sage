import json
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import timeit
import argparse


# # Create dataset. Here we create a dataset made of instruction-output pairs, taken from the SageMath documentation. Each instruction is a piece of documentation and the output is the corresponding SageMath code.

def get_reduced_text(text):
    flag = 0
    for i in range(len(text)):
        if "#" in text[i] and flag == 0:
            start_index = i
            flag = 1
        if (text[i] in ["Next", "Previous"])and text[i+1] == '':
            end_index = i
            break
    return text[start_index+1:end_index]
    
def remove_double_spaces(text):
    text_nospaces = []
    for i in range(len(text)-1):
        if text[i] == "" and text[i+1] == "":
            continue
        elif text[i] == "" and text[i+1] in ["OUTPUT:", "INPUT:", "EXAMPLES:"]:
            continue
        else:
            text_nospaces.append(text[i])
    return text_nospaces

def fix_other_spaces(text):
    text_nospaces = []
    for i in range(len(text)-1):
        if (text[i] == "" and text[i-1] in ["OUTPUT:", "INPUT:", "EXAMPLES:"] and text[i+1] != "") or text[i] in ["EXAMPLES:"]:
            continue
        else:
            text_nospaces.append(text[i])
    return text_nospaces

def text_to_data(text):
    flag = 0
    flag_empty = 0
    data = []
    for i in range(len(text)-1):
        if text[i] != "" and flag == 0:
            flag = 1
            flag_empty = 1
            instruction = text[i]
            output = ""
        elif text[i] != "" and flag == 1 and "sage: " not in text[i]:
            t = text[i].strip()
            instruction += (" " + t)
        elif flag in [1, 2] and ("sage: " in text[i] or "....:" in text[i]): 
            pos_dash = len(text[i]) - 1
            if "# optional" in text[i]:
                for j in range(len(text[i])-1):
                    if text[i][j] + text[i][j+1] == "  ":
                        pos_dash = j
                        break
            output += (text[i][:pos_dash+1] + "\n")
            flag = 2
        elif text[i] == "" and text[i+1] != "":
            flag = 0
        if flag == 0 and flag_empty == 1:
            if output != "":
                function = {}
                pos_dash, pos_dot = 0, 0 
                if ")#" in instruction:
                    for j in range(len(instruction)-1):
                        if pos_dash == 0 and instruction[j] + instruction[j+1] == ")#":
                            pos_dash = j
                        elif pos_dot == 0 and ((j == len(instruction)-1 and instruction[j] == ".") or instruction[j] + instruction[j+1] == ". "):
                            pos_dot = j
                            break
                if pos_dot != 0 and pos_dash != 0:
                    function["instruction"] = instruction[pos_dash+3 : ].strip()
                    function["input"] = ""
                else:
                    function["instruction"] = instruction
                    function["input"] = ""
            
                function["output"] = output
                #'''
                if ("sage:" in function["instruction"] or "sage:" in function["input"]) and len(data) > 0:
                    if "sage:" in function["instruction"]:
                        data[-1]["output"] += "\n " + function["instruction"]
                    if "sage:" in function["input"]:
                        data[-1]["output"] += "\n " + function["input"]
                    new_output = data[-1]["output"].replace("sage:", "").strip()
                    data[-1]["output"] = new_output
                elif ("sage:" in function["instruction"] or "sage:" in function["input"]) and len(data) == 0:
                    continue
                else:
                    new_output = function["output"].replace("sage:", "").strip()
                    function["output"] = new_output
                    data.append(function)
                flag_empty == 0
    return data

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--dataset_name', default='dataset', type=str, help="Name of the dataset")
	args = parser.parse_args()

	dataset_name = args.dataset_name

	url = "https://doc.sagemath.org/html/en/reference/index.html"

	page = requests.get(url)    
	data = page.text
	soup = BeautifulSoup(data, "lxml")

	reference_links = []
	i = 0
	for link in soup.find_all('a'):
		end_link = link.get('href')
		if end_link[0] not in ["#", "."] and end_link[:4] not in ["http"]:
			reference_links.append("https://doc.sagemath.org/html/en/reference/"+end_link)

	reference_links = reference_links[1:-7]

	dataset = []
	for url in reference_links:
		print(url)
		page = requests.get(url)    
		data = page.text
		soup = BeautifulSoup(data, "lxml")

		i = 0
		for link in soup.find_all('a'):
			end_link = link.get('href')
			if end_link[0] not in ["#", "."] and end_link[:4] not in ["http"]:
				sub_url = url[:-10] + end_link
				html = requests.get(sub_url).text
				soup = BeautifulSoup(html, "lxml")
				text = soup.get_text().split("\n")
				text = get_reduced_text(text)
				text = remove_double_spaces(text)
				text = fix_other_spaces(text)
				sub_data = text_to_data(text)
				dataset = dataset + sub_data

	print("Number of elements in the dataset:", len(dataset))

	with open(os.path.expanduser('~/alpaca_lora_sage/'+dataset_name+'.json'), 'w') as f:
		json.dump(dataset, f)


