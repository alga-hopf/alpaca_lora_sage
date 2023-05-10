#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import timeit


# # Create dataset
# 
# In this notebook we create a dataset made of instruction-output pairs, taken from the SageMath documentation. Each instruction is a piece of documentation and the output is the corresponding SageMath code.

# In[59]:


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
        elif flag in [1, 2] and ("sage: " in text[i] or "....:" in text[i]): # and text[i] != "" 
            pos_dash = len(text[i]) - 1
            if "# optional" in text[i]:
                for j in range(len(text[i])-1):
                    if text[i][j] + text[i][j+1] == "  ":
                        pos_dash = j
                        break
            output += (text[i][:pos_dash+1] + "\n")
            flag = 2
        #elif text[i] == "" and "sage: " in text[i+1]:
         #   print('DONE')
          #  continue
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
                #print(pos_dot, pos_dash)
                if pos_dot != 0 and pos_dash != 0:
                    function["instruction"] = instruction[pos_dash+3 : ].strip()#pos_dot+1].strip()
                    function["input"] = ""
                    #function["input"] = instruction[pos_dot:].strip()
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
                    '''
                    if "sage:" in function["instruction"]:
                        new_output = function["instruction"] + "\n" + function["output"]
                    else:
                        new_output = function["input"] + "\n" + function["output"]
                    print(text)
                    '''
                    continue
                else:
                #'''
                    new_output = function["output"].replace("sage:", "").strip()
                    function["output"] = new_output
                    data.append(function)
                flag_empty == 0
                #print(function)
        #print(len(data))
    return data


# In[60]:


url = "https://doc.sagemath.org/html/en/reference/index.html"

page = requests.get(url)    
data = page.text
soup = BeautifulSoup(data)

reference_links = []
i = 0
for link in soup.find_all('a'):
    end_link = link.get('href')
    if end_link[0] not in ["#", "."] and end_link[:4] not in ["http"]:# and i > 0 and end_link[:11] not in ["documentation"]:
        reference_links.append("https://doc.sagemath.org/html/en/reference/"+end_link)


# In[61]:


reference_links = reference_links[1:-7]


# In[62]:


reference_links[5]


# In[63]:


get_ipython().run_cell_magic('time', '', 'dataset = []\nfor url in reference_links:\n    print(url)\n    page = requests.get(url)    \n    data = page.text\n    soup = BeautifulSoup(data)\n\n    #reference_links = []\n    i = 0\n    for link in soup.find_all(\'a\'):\n        end_link = link.get(\'href\')\n        if end_link[0] not in ["#", "."] and end_link[:4] not in ["http"]:# and i > 0 and end_link[:11] not in ["documentation"]:\n            sub_url = url[:-10] + end_link\n            #print(sub_url)\n            html = requests.get(sub_url).text\n            soup = BeautifulSoup(html)\n            text = soup.get_text().split("\\n")\n            text = get_reduced_text(text)\n            text = remove_double_spaces(text)\n            text = fix_other_spaces(text)\n            #for d in text:\n             #   print(d)\n            sub_data = text_to_data(text)\n            dataset = dataset + sub_data\n')


# In[64]:


print("Number of elements in the dataset:", len(dataset))


# In[65]:


with open('/home/alice/Desktop/alpaca/dataset.json', 'w') as f:
    json.dump(dataset, f)


# # Load and inspect dataset

# In[2]:


with open('/home/alice/Desktop/alpaca/dataset.json', 'r') as f:
    raw_dataset = json.load(f)


# In[3]:


print("Number of elements in the dataset:", len(raw_dataset))


# In[ ]:




