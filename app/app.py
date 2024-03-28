from flask import Flask, render_template, request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_path = '../results/final'

model = AutoModelForCausalLM.from_pretrained(
    save_path,
    device_map = 'auto')

tokenizer = AutoTokenizer.from_pretrained(
    save_path)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map = 'auto',
    pad_token_id = tokenizer.eos_token_id,
    max_new_tokens = 250,
    temperature = 1.0
)

def instruction_prompt(instruction, prompt_input=None):
	
	if prompt_input:
		return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{prompt_input}

### Response:
""".strip()
			
	else:
		return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
""".strip()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        instruction = request.form['instruction']
        prompt_input = request.form['prompt_input']
        output = text_generator(instruction_prompt(instruction, prompt_input))
        response = output[0]['generated_text'].split("### Response:\n")[-1]

        return render_template('home.html', instruction=instruction, prompt_input=prompt_input, response=response)

    else:
        return render_template('home.html', instruction="", prompt_input="", response="")

if __name__ == '__main__':
    app.run(debug=True)