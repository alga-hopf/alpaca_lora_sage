#!/bin/bash

# Script that takes as input a  prompt from the user and returns the sage results of the output generated with the model.

echo -n "Insert your prompt here: "
read prompt
python3 generate_sage_script.py --prompt $prompt
sage output_code.sage
