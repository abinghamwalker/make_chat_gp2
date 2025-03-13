# Install required packages
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, 
    GPT2Config, 
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from tqdm.auto import tqdm

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the Alpaca dataset
def download_and_load_file(file_path, url):
    import urllib.request
    import ssl
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

# Download Alpaca dataset
alpaca_path = "alpaca_data.json"
alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

data = download_and_load_file(alpaca_path, alpaca_url)
print("Number of entries:", len(data))

# Split data into train/val/test
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
val_data = data[train_portion:train_portion + val_portion]
test_data = data[train_portion + val_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# Format data entries
def format_example(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    response_text = f"\n\n### Response:\n{entry['output']}"
    
    return instruction_text + input_text + response_text

# Create a custom dataset with batch preparation to save memory
class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = format_example(self.data[idx])
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encodings.input_ids[0],
            "attention_mask": encodings.attention_mask[0],
            "labels": encodings.input_ids[0]  # Same as input_ids for language modeling
        }

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a padding token by default

# Load the medium GPT-2 model (355M parameters)
print("Loading GPT-2 Medium (355M parameters)...")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))  # Resize for any added tokens
model.to(device)

# Optional: Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Create datasets with a smaller subset of data to fit in memory
print("Creating datasets...")
# Use smaller subsets to avoid memory issues
train_subset_size = 5000  # Adjust based on your Colab's memory
train_dataset = AlpacaDataset(train_data[:train_subset_size], tokenizer)
val_dataset = AlpacaDataset(val_data[:500], tokenizer)

# Configure training arguments optimized for Colab with T4 GPU
training_args = TrainingArguments(
    output_dir="./gpt2-medium-alpaca",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Smaller batch size for larger model
    per_device_eval_batch_size=2,
    eval_steps=500,
    save_steps=1000,
    warmup_steps=500,
    evaluation_strategy="steps",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    fp16=True,  # Use mixed precision training
    gradient_accumulation_steps=4,  # Accumulate gradients over multiple steps
    save_total_limit=2,  # Limit the number of checkpoints to save memory
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
model_save_path = "./gpt2-medium-alpaca-final"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Test inference function
def generate_response(instruction, input_text=""):
    # Format the input
    if input_text:
        prompt = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
        )
    else:
        prompt = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{instruction}\n\n### Response:"
        )
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate a response
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the response part
    response_parts = generated_text.split("### Response:")
    if len(response_parts) > 1:
        return response_parts[1].strip()
    else:
        return generated_text
        
# Test the model with a sample instruction
test_instruction = "Write a short poem about artificial intelligence."
print("\nTesting model with instruction:", test_instruction)
response = generate_response(test_instruction)
print("\nGenerated response:")
print(response)