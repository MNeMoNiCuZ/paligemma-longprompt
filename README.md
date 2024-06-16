# What is this model?
This is an experimental vision model that generates captions / prompts of an input image, based on a very long and complex structure.
It combines both booru-style tagging (comma separated keyword tags), and longer descriptive texts.

Example:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/644ed23467c9458c913059ff/09mIVsk_qSxazCCX5TDUN.jpeg)
> waterfall, no_humans, outdoors, scenery, tree, lake, rock, river, water, nature, plant, sky, grass, day, island, blue_sky, solo, mountain, forest, a peaceful and natural landscape image of a waterfall with a pond nestled in the woods has been created using digital art techniques. the vibrant green foliage of the trees, lush pink flowers in full bloom, and the sparkling waters ofthe lagoon create a sense of harmony and tranquility that's hard to put into words. the waterfall stands tall, its majestic beauty accentuated by the lush surroundings. towering over the serene pond, it is like a giant gift from nature itself. situated in a tranquil setting, the scene exudes peacefulness and a feeling of serenity. the image features a beautiful tropical landscape with an impressive waterfall, surrounded by rocks and trees. a few leaves can be seen floating on the water. there are also some flowers scattered about, adding color and texture to the environment. flowers are known for their bright colors and delicate petals that add beauty to any setting. placed strategically, they draw attention to where they are displayed, highlighting the natural beauty of this spectacularly designed piece

# Why though?

It's an experiment in longer and more complex descriptions. My goal is to create a mix of keyword tags and descriptions so that both can be used when prompting, and for the prompt to be of high quality.

This model in it's current state does not succeed with that. It needs further training and refinement.

# Simple usage script
`pip install git+https://github.com/huggingface/transformers`

```
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "gokaygokay/sd3-long-captioner"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to('cuda').eval()
processor = AutoProcessor.from_pretrained(model_id)

## prefix
prompt = "caption en"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda')
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
```

# Batch process input folder script with 4/8 quantization options
```
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import os
import glob
from colorama import init, Fore, Style
from datetime import datetime
import time
import re
from huggingface_hub import snapshot_download

# Initialize colorama
init(autoreset=True)

# Settings
quantization_bits = 8  # Set to None for full precision, 4 for 4-bit quantization, or 8 for 8-bit quantization
generation_token_length = 256
min_tokens = 20  # Minimum number of tokens required in the generated output
max_word_character_length = 30  # Maximum length of a word before it's considered too long
prune_end = True  # Remove any trailing chopped off end text until it reaches a . or ,
output_format = ".txt"  # Output format for the generated captions

# Clean up of poorly generated prompts
repetition_penalty = 1.15  # Control the repetition penalty (higher values discourage repetition)
retry_words = ["no_parallel"]  # If these words are encountered, the entire generation retries
max_retries = 10
remove_words = ["#", "/", "、", "@", "__", "|", "  ", ";", "~", "\"", "*", "^", ",,", "ON DISPLAY:"]  # Words or characters to be removed from the output results
strip_contents_inside = ["(", "[", "{"]  # Specify which characters to strip out along with their contents
remove_underscore_tags = True  # Option to remove words containing underscores

# Specify the model path
model_name = "mnemic/paligemma-longprompt-v1-safetensors"
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
model_path = os.path.join(models_dir, model_name.split('/')[-1])

# Ensure the local directory is correctly specified relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
local_model_path = model_path  # Use the specified model directory

# Directory paths
input_dir = os.path.join(script_dir, 'input')
output_in_input_dir = True  # Set this to False if you want to use a separate output directory
output_dir = input_dir if output_in_input_dir else os.path.join(script_dir, 'output')

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to download the model from HuggingFace using snapshot_download
def download_model(model_name, model_path):
    if not os.path.exists(model_path):
        print(Fore.YELLOW + f"Downloading model {model_name} to {model_path}...")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False, local_files_only=False)
        print(Fore.GREEN + "Model downloaded successfully.")
    else:
        print(Fore.GREEN + f"Model directory already exists: {model_path}")

# Download the model if not already present
download_model(model_name, model_path)

# Check that the required files are in the local_model_path
required_files = ["config.json", "tokenizer_config.json"]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(local_model_path, f))]
safetensor_files = [f for f in os.listdir(local_model_path) if f.endswith(".safetensors")]
if missing_files:
    raise FileNotFoundError(f"Missing required files in {local_model_path}: {', '.join(missing_files)}")
if not safetensor_files:
    raise FileNotFoundError(f"No safetensors files found in {local_model_path}")

# Load model and processor from local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(Fore.YELLOW + "Loading model and processor...")
try:
    if quantization_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            local_model_path,
            quantization_config=bnb_config,
            device_map={"": 0},
        ).eval()
    elif quantization_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            local_model_path,
            quantization_config=bnb_config,
            device_map={"": 0},
        ).eval()
    elif quantization_bits is None:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            local_model_path
        ).eval()
        model.to(device)  # Ensure the model is on the correct device
    else:
        raise ValueError("Unsupported quantization_bits value. Use None for full precision, 4 for 4-bit quantization, or 8 for 8-bit quantization.")

    processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True)
    print(Fore.GREEN + "Model and processor loaded successfully.")
except OSError as e:
    print(Fore.RED + f"Error loading model or processor: {e}")
    raise

# Process each image in the input directory recursively
image_extensions = ['jpg', 'jpeg', 'png', 'webp']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_dir, '**', f'*.{ext}'), recursive=True))

print(Fore.YELLOW + f"Found {len(image_paths)} image(s) to process.\n")

def prune_text(text):
    if not prune_end:
        return text
    # Find the last period or comma
    last_period_index = text.rfind('.')
    last_comma_index = text.rfind(',')
    prune_index = max(last_period_index, last_comma_index)
    if prune_index != -1:
        # Return text up to the last period or comma
        return text[:prune_index].strip()
    return text

def contains_retry_word(text, retry_words):
    return any(word in text for word in retry_words)

def remove_unwanted_words(text, remove_words):
    for word in remove_words:
        text = text.replace(word, ' ')
    return text

def strip_contents(text, chars):
    for char in chars:
        if char == "(":
            text = re.sub(r'\([^)]*\)', ' ', text)
        elif char == "[":
            text = re.sub(r'\[[^\]]*\]', ' ', text)
        elif char == "{":
            text = re.sub(r'\{[^}]*\}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
    text = re.sub(r'\s([,.!?;])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,.!?;])\s', r'\1 ', text)  # Add space after punctuation if missing
    return text.strip()

def remove_long_words(text, max_word_length):
    words = text.split()
    for i, word in enumerate(words):
        if len(word) > max_word_length:
            # Strip back to the previous comma or period
            last_period_index = text.rfind('.', 0, text.find(word))
            last_comma_index = text.rfind(',', 0, text.find(word))
            prune_index = max(last_period_index, last_comma_index)
            if prune_index != -1:
                return text[:prune_index].strip()
            else:
                return text[:text.find(word)].strip()
    return text

def clean_text(text):
    text = remove_unwanted_words(text, remove_words)
    text = strip_contents(text, strip_contents_inside)
    text = remove_long_words(text, max_word_character_length)
    # Remove unwanted characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    if remove_underscore_tags:
        text = ' '.join([word for word in text.split() if '_' not in word])
    return text

for image_path in image_paths:
    output_file_path = os.path.splitext(image_path)[0] + output_format if output_in_input_dir else os.path.join(output_dir, os.path.splitext(os.path.relpath(image_path, input_dir))[0] + output_format)
    
    if os.path.exists(output_file_path):
        # print(Fore.CYAN + f"Skipping {image_path}, output already exists.")
        continue

    try:
        start_time = datetime.now()
        print(Fore.CYAN + f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting processing for {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        prompt = "caption en"
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)  # Ensure inputs are on the correct device
        input_len = model_inputs["input_ids"].shape[-1]

        # Generate the caption with additional parameters to reduce repetitiveness
        retries = 0
        success = False
        while retries < max_retries:
            with torch.inference_mode():
                generation_start_time = time.time()
                generation = model.generate(
                    **model_inputs,
                    max_new_tokens=generation_token_length,
                    do_sample=True,  # Enable sampling
                    temperature=0.7,  # Control randomness of predictions
                    top_k=50,  # Consider top 50 candidates
                    top_p=0.9,  # Consider tokens that comprise the top 90% probability mass
                    no_repeat_ngram_size=2,  # Avoid repeating 2-grams
                    repetition_penalty=repetition_penalty  # Apply a penalty to repeated tokens
                )
                generation_end_time = time.time()
                generation = generation[0][input_len:]
                decoded = processor.decode(generation, skip_special_tokens=True)
                pruned_text = prune_text(decoded)
                
                if not contains_retry_word(pruned_text, retry_words) and len(pruned_text.split()) >= min_tokens:
                    success = True
                    break
                retries += 1
                print(Fore.YELLOW + f"Retrying generation for {image_path} due to retry word or insufficient tokens, attempt {retries}")
            
            if retries == max_retries:
                print(Fore.RED + f"Max retries reached for {image_path}. Saving the result with retry word or insufficient tokens.")

        # Clean the text
        cleaned_text = clean_text(pruned_text)

        # Save the output to a text file, replicating the directory structure
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
            f.write(cleaned_text)
        
        end_time = datetime.now()
        duration = generation_end_time - generation_start_time
        
        print(Fore.GREEN + f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Processed {image_path}, saved to {output_file_path}")
        print(Fore.LIGHTBLACK_EX + f"Output: {cleaned_text}")
        print(Fore.LIGHTBLACK_EX + f"Time taken for generation: {duration:.2f} seconds\n")
        
        # Clear memory
        del model_inputs
        torch.cuda.empty_cache()
    except Exception as e:
        print(Fore.RED + f"Error processing {image_path}: {e}\n")
```
