
import time
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Custom dataset for GPT training
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the text into overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Create a DataLoader for GPT training
def create_dataloader(txt, batch_size=2, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

# Multi-head attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

# Layer normalization
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# GELU activation
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

# Feed-forward network
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"], num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# GPT model
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# Text generation function
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Tokenization utilities
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# Training and evaluation functions
def calc_loss_batch(input_batch, target_batch, model, device):
    try:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target_batch.reshape(-1))
        return loss
    except RuntimeError as e:
        if "MPS backend out of memory" in str(e):
            print("MPS memory error, falling back to CPU for this batch")
            model_cpu = model.to("cpu")
            input_batch_cpu, target_batch_cpu = input_batch.to("cpu"), target_batch.to("cpu")
            logits = model_cpu(input_batch_cpu)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target_batch_cpu.reshape(-1))
            model.to(device)
            return loss.to(device)
        else:
            raise

def calc_loss_loader(data_loader, model, device, max_batches=None):
    total_loss = 0.0
    batch_count = 0
    max_batches = max_batches if max_batches is not None else len(data_loader)
    pbar = tqdm(total=max_batches, desc="Calculating loss", leave=False, position=2)

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        loss = calc_loss_batch(inputs, targets, model, device)
        total_loss += loss.item()
        batch_count += 1
        pbar.update(1)
    pbar.close()
    return total_loss / max(1, batch_count)

def evaluate_model(model, train_loader, val_loader, device, eval_iter, progress_callback=None):
    """
    Evaluate model on train and validation data
    Returns average loss for each dataset
    """
    model.eval()
    train_loss = 0.0
    val_loss = 0.0
    
    # Evaluate on training data
    print("Evaluating on training data...")
    train_batches = 0
    try:
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(train_loader):
                if i >= eval_iter:
                    break
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                train_loss += loss.item()
                train_batches += 1
        
        if train_batches > 0:
            train_loss /= train_batches
    except Exception as e:
        print(f"Error during training evaluation: {str(e)}")
        train_loss = float('inf')  # Set to a high value on error
    
    if progress_callback:
        progress_callback(1)
    
    # Evaluate on validation data
    print("Evaluating on validation data...")
    val_batches = 0
    try:
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(val_loader):
                if i >= eval_iter:
                    break
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                val_loss += loss.item()
                val_batches += 1
        
        if val_batches > 0:
            val_loss /= val_batches
    except Exception as e:
        print(f"Error during validation evaluation: {str(e)}")
        val_loss = float('inf')  # Set to a high value on error
    
    if progress_callback:
        progress_callback(1)
    
    model.train()
    return train_loss, val_loss

    
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        print(f"\nEpoch {epoch+1}/{num_epochs} started.")
        epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader), position=0)

        for batch_idx, (input_batch, target_batch) in enumerate(epoch_progress):
            try:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1
                epoch_progress.set_postfix({'loss': f"{loss.item():.3f}", 'tokens': tokens_seen})

                if global_step % eval_freq == 0:
                    print(f"\nEvaluating at step {global_step}...")
                    with tqdm(total=2, desc="Evaluation", position=1) as eval_progress:
                        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter, progress_callback=lambda x: eval_progress.update(x))
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Epoch {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            except RuntimeError as e:
                if "MPS" in str(e):
                    print(f"MPS memory error at Epoch {epoch+1}, Batch {batch_idx+1}. Falling back to CPU.")
                    device_cpu = torch.device("cpu")
                    input_batch = input_batch.to(device_cpu)
                    target_batch = target_batch.to(device_cpu)
                    model.to(device_cpu)
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device_cpu)
                    loss.backward()
                    optimizer.step()
                    tokens_seen += input_batch.numel()
                    global_step += 1
                    epoch_progress.set_postfix({'loss (CPU)': f"{loss.item():.3f}", 'tokens': tokens_seen})
                    model.to(device)
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)
                else:
                    raise e

        epoch_progress.close()
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1} COMPLETED - MODEL GENERATION:")
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
        full_text = token_ids_to_text(token_ids, tokenizer)
        continuation = full_text[len(start_context):]
        model.train()

    return train_losses, val_losses, track_tokens_seen

#Helper functions to load models and convert tokens
def load_model(model_path, config):
    """
    Load a previously saved GPT model from state dictionary
    
    Parameters:
    - model_path: Path to the saved state dictionary
    - config: Model configuration dictionary (default uses GPT_CONFIG_124M)
    
    Returns:
    - Loaded model ready for inference
    """
    # Create a new model instance with the same configuration
    model = GPTModel(config)
    
    # Load the state dictionary
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def generate_text(model, start_context, max_new_tokens=50):
    """
    Generate text using the loaded model
    
    Parameters:
    - model: Loaded GPT model
    - start_context: Initial text to start generation
    - max_new_tokens: Maximum number of tokens to generate
    
    Returns:
    - Generated text
    """
    # Use the tokenizer from the original script
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Convert start context to token ids
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    
    # Generate tokens
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, 
            idx=encoded, 
            max_new_tokens=max_new_tokens, 
            context_size=context_size
        )
    
    # Convert tokens back to text
    full_text = token_ids_to_text(token_ids, tokenizer)
    return full_text

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # Ensure model and idx are on the same device
    device = next(model.parameters()).device
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            neg_inf = torch.tensor(float("-inf"), device=device)
            logits = torch.where(logits < min_val, neg_inf, logits)

        if temperature > 0.0:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if eos_id is not None and (idx_next == eos_id).all():  
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

import os
import requests  
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = requests.get(url, stream=True, verify=False)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = url.split("/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        print(f"Loading block {b}")  # Debug: Track block number
        try:
            # Attention weights
            attn_c_attn = params["blocks"][b]["attn"]["c_attn"]
            q_w, k_w, v_w = np.split(attn_c_attn["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            # Attention biases
            q_b, k_b, v_b = np.split(attn_c_attn["b"], 3, axis=-1)

            # Debugging: Check if bias parameters exist
            if gpt.trf_blocks[b].att.W_query.bias is None:
                print(f"Error: gpt.trf_blocks[{b}].att.W_query.bias is None!")
            if gpt.trf_blocks[b].att.W_key.bias is None:
                print(f"Error: gpt.trf_blocks[{b}].att.W_key.bias is None!")
            if gpt.trf_blocks[b].att.W_value.bias is None:
                print(f"Error: gpt.trf_blocks[{b}].att.W_value.bias is None!")

            gpt.trf_blocks[b].att.W_query.bias = assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            # Attention output projection
            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight, 
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias, 
                params["blocks"][b]["attn"]["c_proj"]["b"])

            # Feedforward network
            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight, 
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias, 
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight, 
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias, 
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            # Layer norms
            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale, 
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift, 
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale, 
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift, 
                params["blocks"][b]["ln_2"]["b"])

        except KeyError as e:
            print(f"KeyError: {e} in block {b}. Check the structure of params['blocks'][{b}].")
            raise
        except AttributeError as e:
            print(f"AttributeError: {e} in block {b}. Check the GPT model definition.")
            raise
        except Exception as e:
            print(f"Unexpected error: {e} in block {b}.")
            raise

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


__all__ = ["generate", "generate_text", "load_model", "GPTModel", "generate_text_simple", "text_to_token_ids","token_ids_to_text","download_and_load_gpt2","load_weights_into_gpt","evaluate_model"]  