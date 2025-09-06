import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the visible GPU device to 0

import torch  # Import PyTorch for tensor operations and neural networks
import torch.nn as nn  # Import neural network modules from PyTorch
from torch.utils.data import DataLoader, Subset  # Import DataLoader for batching data and Subset for splitting datasets
from torch.amp import autocast, GradScaler  # Import autocast for mixed precision training and GradScaler for gradient scaling
from torch.optim.lr_scheduler import CosineAnnealingLR  # Import CosineAnnealingLR for learning rate scheduling
from transformers import CLIPImageProcessor, CLIPVisionModel, LlamaTokenizer, LlamaForCausalLM  # Import specific models and processors from Hugging Face Transformers
from models.transformer import VisualLlamaForCausalLM  # Import custom VisualLlamaForCausalLM model
from models.transformer.configuration_llama_visualgpt import VisualLlamaConfig  # Import custom configuration for VisualLlama
import yaml  # Import YAML for loading configuration files
import json  # Import JSON for handling JSON data
from PIL import Image  # Import Image from PIL for image processing
import gc  # Import garbage collector for manual memory management
import numpy as np  # Import NumPy for numerical operations
from tqdm import tqdm  # Import tqdm for progress bars
import time  # Import time for timing operations
import sys  # Import sys for system-specific parameters and functions
import signal  # Import signal for handling signals like interrupts
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # Import BLEU score and smoothing from NLTK
from nltk.translate.meteor_score import meteor_score  # Import METEOR score from NLTK
from rouge_score import rouge_scorer  # Import ROUGE scorer
from pycocoevalcap.cider.cider import Cider  # Import CIDEr scorer
import pickle  # Import pickle for serializing data
import hashlib  # Import hashlib for hashing
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import shutil  # Import shutil for file operations like deleting directories

# Ensure UTF-8 encoding for standard output
sys.stdout.reconfigure(encoding='utf-8')

def get_model_config(model):
    # Function to get the configuration of the model
    # Input: model (nn.Module or nn.DataParallel)
    # Output: config (VisualLlamaConfig)
    return model.module.config if isinstance(model, nn.DataParallel) else model.config

def clear_cache(cache_dir="/home/yazan/VisualLlama/cache"):
    """Clear dataset cache directory."""
    # Function to clear the cache directory
    # Input: cache_dir (str) - path to cache directory
    # Output: None, but prints confirmation if cleared
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Cleared cache directory: {cache_dir}")

class CocoDataset(torch.utils.data.Dataset):
    # Custom Dataset class for COCO dataset
    def __init__(self, annotation_path, image_path, processor, tokenizer, max_length=512, cache_dir="/home/yazan/VisualLlama/cache", karpathy_val=False, val_size=5000, seed=0):
        # Constructor for CocoDataset
        # Inputs:
        # - annotation_path (str): Path to annotation JSON file
        # - image_path (str): Path to image directory
        # - processor (CLIPImageProcessor): Image processor
        # - tokenizer (LlamaTokenizer): Text tokenizer
        # - max_length (int): Maximum length for tokenization (default 512)
        # - cache_dir (str): Directory for caching processed data
        # - karpathy_val (bool): Flag for Karpathy validation split
        # - val_size (int): Size for validation split
        # - seed (int): Random seed for reproducibility
        # Outputs: None, initializes instance variables
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file {annotation_path} not found. Please ensure the COCO subset was created correctly.")
        
        try:
            with open(annotation_path, 'r') as f:
                self.annotations = json.load(f)  # Load annotations as dict
        except Exception as e:
            print(f"Error loading annotations from {annotation_path}: {e}")
            raise
        
        images = self.annotations.get('images', [])  # List of image dicts
        annotations = self.annotations.get('annotations', [])  # List of annotation dicts
        
        # Karpathy validation split: sample val_size image-caption pairs
        if karpathy_val:
            np.random.seed(seed)
            indices = np.random.choice(len(images), min(val_size, len(images)), replace=False)  # Random indices for split
            images = [images[i] for i in indices]  # Selected images
            selected_image_ids = set(img['id'] for img in images)  # Set of selected image IDs
            annotations = [ann for ann in annotations if ann['image_id'] in selected_image_ids]  # Filtered annotations
        
        self.image_info = {img['id']: img for img in images}  # Dict: image_id -> image info
        self.image_captions = {}  # Dict: image_id -> list of captions
        valid_images = 0
        invalid_captions = 0
        for ann in annotations:
            image_id = ann.get('image_id')
            caption = ann.get('caption', "")
            if image_id and isinstance(caption, str) and len(caption.strip()) > 0 and caption != "No caption available":
                if image_id not in self.image_captions:
                    self.image_captions[image_id] = []
                self.image_captions[image_id].append(caption[:150])  # Truncate caption to 150 chars
                valid_images += 1
            else:
                invalid_captions += 1
        self.image_captions = {k: v for k, v in self.image_captions.items() if v}  # Remove empty lists
        
        print(f"Loaded {len(self.image_captions)} valid images with captions, {valid_images} total valid annotations, {invalid_captions} invalid captions")
        for i, (img_id, captions) in enumerate(list(self.image_captions.items())[:5]):
            img_info = self.image_info.get(img_id, {})
            file_name = img_info.get('file_name', 'unknown')
            path = os.path.join(self.image_path, file_name)
            print(f"Sample {i}: Image ID={img_id}, Path={path}, Exists={os.path.exists(path)}, Captions={captions[:2]}")
        
        if len(self.image_captions) == 0:
            print(f"Warning: No valid images with captions found in {annotation_path}. Using dummy sample.")
            self.image_info = {0: {'id': 0, 'file_name': 'dummy.jpg'}}
            self.image_captions = {0: ["Dummy caption"]}
        
        self.valid_image_ids = list(self.image_captions.keys())  # List of valid image IDs

    def __len__(self):
        # Returns the length of the dataset
        # Input: None
        # Output: int - number of valid images
        return len(self.valid_image_ids)
    
    def __getitem__(self, idx):
        # Get item by index
        # Input: idx (int) - index
        # Output: dict with 'pixel_values' (torch.Tensor [3, 224, 224]), 'input_ids' (torch.Tensor [max_length]), 'attention_mask' (torch.Tensor [max_length]), 'labels' (torch.Tensor [max_length]), 'captions' (list of str), 'image_id' (int)
        image_id = self.valid_image_ids[idx]
        image_info = self.image_info.get(image_id, {'file_name': 'dummy.jpg'})
        
        cache_key = hashlib.md5(f"{image_id}_{self.max_length}".encode()).hexdigest()  # Create cache key
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")  # Cache file path
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)  # Load from cache if exists
            except:
                pass
        
        possible_paths = [
            os.path.join(self.image_path, image_info['file_name']),
            os.path.join(self.image_path, 'train2017', image_info['file_name']),
            os.path.join(self.image_path, 'val2017', image_info['file_name'])
        ]
        
        image_path = next((path for path in possible_paths if os.path.exists(path)), None)  # Find existing image path
        try:
            image = Image.new('RGB', (224, 224), color='gray') if image_path is None else Image.open(image_path).convert('RGB')  # Load image or create dummy
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values  # Process image to tensor [1, 3, 224, 224]
        captions = self.image_captions.get(image_id, ["Dummy caption"])  # Get captions or dummy
        caption = captions[0]  # Take first caption
        
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding_side='left'
        )  # Tokenize caption
        input_ids = tokenized['input_ids'].squeeze(0)  # [max_length]
        
        data = {
            'pixel_values': pixel_values.squeeze(0),  # [3, 224, 224]
            'input_ids': input_ids,  # [max_length]
            'attention_mask': tokenized['attention_mask'].squeeze(0),  # [max_length]
            'labels': input_ids.clone(),  # [max_length]
            'captions': captions,  # list of str
            'image_id': image_id  # int
        }
        
        print(f"Dataset item {idx}: Image ID={image_id}, Caption='{caption[:50]}...', Input IDs shape={input_ids.shape}, Pixel values mean={pixel_values.mean().item():.4f}, std={pixel_values.std().item():.4f}")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)  # Save to cache
        except:
            pass
        
        return data

def custom_data_collator(features):
    # Custom collator for batching
    # Input: features (list of dicts from __getitem__)
    # Output: dict with batched tensors and lists
    from transformers.data.data_collator import default_data_collator
    tensorizable_features = [
        {k: v for k, v in f.items() if k not in ['captions', 'image_id']}
        for f in features
    ]
    batch = default_data_collator(tensorizable_features)  # Batch tensor fields
    batch['captions'] = [f['captions'] for f in features]  # List of captions
    batch['image_id'] = [f['image_id'] for f in features]  # List of image IDs
    return batch

def load_config(config_path):
    # Load configuration from YAML file
    # Input: config_path (str)
    # Output: dict - configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['train']['lr'] = float(config['train'].get('lr', 5e-5))  # Ensure LR is float
    config['train']['accumulation_steps'] = int(config['train'].get('accumulation_steps', 4))  # Ensure accumulation_steps is int
    config['train']['patience'] = int(config['train'].get('patience', 3))  # Ensure patience is int
    config['model']['freeze_vit'] = config['model'].get('freeze_vit', True)  # Ensure freeze_vit is bool
    config['train']['validate_interval'] = int(config['train'].get('validate_interval', 2))  # Ensure validate_interval is int
    return config

def create_model(config, llama_model_path):
    # Create and initialize the model
    # Input: config (dict), llama_model_path (str)
    # Output: VisualLlamaForCausalLM instance
    try:
        tokenizer = LlamaTokenizer.from_pretrained(
            llama_model_path,
            local_files_only=os.path.isdir(llama_model_path),
            token=True if not os.path.isdir(llama_model_path) else False
        )  # Load tokenizer
    except Exception as e:
        print(f"Error loading tokenizer from {llama_model_path}: {e}")
        print("Ensure the path is a valid local directory with tokenizer files or a Hugging Face model ID.")
        print("For local paths, required files include: tokenizer.json, vocab.json, tokenizer_config.json")
        print("For Hugging Face models, ensure you have access and are logged in via `huggingface-cli login`.")
        raise
    
    vocab_size = tokenizer.vocab_size  # Get vocabulary size
    pad_token_id = config['model'].get('pad_token_id', tokenizer.pad_token_id or 0)  # Get pad token ID
    if pad_token_id >= vocab_size:
        pad_token_id = vocab_size - 1

    model_config = VisualLlamaConfig(
        hidden_size=config['model'].get('llama_embed_dim', 5120),
        num_hidden_layers=config['model'].get('num_hidden_layers', 4),
        num_attention_heads=config['model'].get('cross_attention_heads', 40),
        intermediate_size=config['model'].get('intermediate_size', 13824),
        vocab_size=vocab_size,
        num_visual_features=config['model'].get('num_visual_features', 50),
        visual_feature_size=config['model'].get('vit_embed_dim', 768),
        max_position_embeddings=config['model'].get('max_length', 512),
        llama_model_path=llama_model_path,
        pad_token_id=pad_token_id,
        tau=config['model'].get('tau', 0.5),
        rms_norm_eps=1e-5
    )  # Create model configuration
    
    model = VisualLlamaForCausalLM(model_config)  # Instantiate model
    
    # Load pretrained Llama weights
    try:
        pretrained_llama = LlamaForCausalLM.from_pretrained(llama_model_path, local_files_only=os.path.isdir(llama_model_path))  # Load pretrained Llama
        print("Loaded pretrained Llama model.")
        
        # Copy embeddings, norm, lm_head
        model.model.embed_tokens.weight.data.copy_(pretrained_llama.model.embed_tokens.weight.data)  # Copy embedding weights
        model.model.norm.weight.data.copy_(pretrained_llama.model.norm.weight.data)  # Copy norm weights
        model.lm_head.weight.data.copy_(pretrained_llama.lm_head.weight.data)  # Copy LM head weights
        
        # Copy first num_hidden_layers layers
        for i in range(model_config.num_hidden_layers):
            visual_layer = model.model.layers[i]
            llama_layer = pretrained_llama.model.layers[i]
            
            # Self-attention
            visual_layer.self_attn.q_proj.weight.data.copy_(llama_layer.self_attn.q_proj.weight.data)
            visual_layer.self_attn.k_proj.weight.data.copy_(llama_layer.self_attn.k_proj.weight.data)
            visual_layer.self_attn.v_proj.weight.data.copy_(llama_layer.self_attn.v_proj.weight.data)
            visual_layer.self_attn.o_proj.weight.data.copy_(llama_layer.self_attn.o_proj.weight.data)
            
            # MLP
            visual_layer.mlp.gate_proj.weight.data.copy_(llama_layer.mlp.gate_proj.weight.data)
            visual_layer.mlp.down_proj.weight.data.copy_(llama_layer.mlp.down_proj.weight.data)
            visual_layer.mlp.up_proj.weight.data.copy_(llama_layer.mlp.up_proj.weight.data)
            
            # Norms
            visual_layer.input_layernorm.weight.data.copy_(llama_layer.input_layernorm.weight.data)
            visual_layer.post_attention_layernorm.weight.data.copy_(llama_layer.post_attention_layernorm.weight.data)
        
        print("Pretrained weights copied successfully.")
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
    
    model.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))  # Move model to device
    model.resize_token_embeddings(vocab_size)  # Resize embeddings if necessary
    
    # Debug: Model config check
    print(f"Model config: hidden_size={model_config.hidden_size}, num_layers={model_config.num_hidden_layers}, vocab_size={model_config.vocab_size}, pad_token_id={model_config.pad_token_id}")
    
    # Debug: Pretrained weights check (sample param)
    print(f"Sample embed weight sum: {model.model.embed_tokens.weight.sum().item():.4f}")
    
    return model

def train_epoch(model, dataloader, optimizer, device, vision_model, epoch_num, accumulation_steps):
    # Train one epoch
    # Inputs:
    # - model (VisualLlamaForCausalLM): Model to train
    # - dataloader (DataLoader): Training data loader
    # - optimizer (torch.optim.Optimizer): Optimizer
    # - device (torch.device): Device to use
    # - vision_model (CLIPVisionModel): Vision model for feature extraction
    # - epoch_num (int): Current epoch number
    # - accumulation_steps (int): Gradient accumulation steps
    # Output: float - average loss
    model.train()  # Set model to training mode
    total_loss = 0  # Initialize total loss
    scaler = GradScaler('cuda')  # Initialize gradient scaler for mixed precision
    optimizer.zero_grad(set_to_none=True)  # Zero gradients
    
    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch_num}")):
        # Loop over batches
        # batch: dict with 'pixel_values' [batch, 3, 224, 224], 'input_ids' [batch, max_length], etc.
        torch.cuda.empty_cache()  # Clear CUDA cache
        batch_tensor = {k: v.to(device, dtype=torch.long if k in ['input_ids', 'labels'] else torch.float16) 
                       for k, v in batch.items() if k not in ['captions', 'image_id']}  # Move tensors to device with appropriate dtype
        captions = batch['captions']  # list of list of str
        
        model_config = get_model_config(model)  # Get model config
        input_ids = batch_tensor['input_ids']  # [batch, max_length]
        if input_ids.max() >= model_config.vocab_size or input_ids.min() < 0:
            input_ids = torch.clamp(input_ids, min=0, max=model_config.vocab_size-1)  # Clamp input IDs
            batch_tensor['input_ids'] = input_ids
            batch_tensor['labels'] = input_ids.clone()
        
        encoder_output = None
        visual_attention_mask = None
        if 'pixel_values' in batch_tensor:
            with torch.no_grad():  # No gradient for vision model
                vision_model_single = vision_model.module if isinstance(vision_model, nn.DataParallel) else vision_model
                vision_model_single.to(device, dtype=torch.float16)  # Move vision model to device
                if torch.isnan(batch_tensor['pixel_values']).any() or torch.isinf(batch_tensor['pixel_values']).any():
                    print(f"Skipping batch {step} due to invalid pixel values")
                    continue
                vision_outputs = vision_model_single(pixel_values=batch_tensor['pixel_values'])  # Extract features
                encoder_output = vision_outputs.last_hidden_state  # [batch, num_visual_features, visual_feature_size]
                # Debug: Visual Feature Verification
                print(f"Batch {step}: Visual features mean={encoder_output.mean().item():.4f}, std={encoder_output.std().item():.4f}, max={encoder_output.max().item():.4f}, min={encoder_output.min().item():.4f}")
                if encoder_output.std().item() < 1e-6:
                    print(f"Warning: Visual features have low variance in batch {step}")
                expected_shape = (batch_tensor['pixel_values'].size(0), model_config.num_visual_features, model_config.visual_feature_size)
                if encoder_output.shape != expected_shape:
                    if encoder_output.shape[1] > model_config.num_visual_features:
                        encoder_output = encoder_output[:, :model_config.num_visual_features, :].to(torch.float16)
                    elif encoder_output.shape[1] < model_config.num_visual_features:
                        pad = torch.zeros(
                            batch_tensor['pixel_values'].size(0),
                            model_config.num_visual_features - encoder_output.shape[1],
                            model_config.visual_feature_size,
                            dtype=torch.float16,
                            device=device
                        )
                        encoder_output = torch.cat([encoder_output, pad], dim=1)  # Pad if necessary
                visual_attention_mask = torch.ones(
                    encoder_output.size(0), encoder_output.size(1), dtype=torch.float16, device=device
                )  # [batch, num_visual_features]
        
        model_inputs = {
            'input_ids': batch_tensor['input_ids'],  # [batch, max_length]
            'attention_mask': batch_tensor['attention_mask'],  # [batch, max_length]
            'labels': batch_tensor['labels'],  # [batch, max_length]
            'encoder_output': encoder_output,  # [batch, num_visual_features, visual_feature_size] or None
            'visual_attention_mask': visual_attention_mask  # [batch, num_visual_features] or None
        }
        
        with autocast('cuda', dtype=torch.float16):  # Mixed precision context
            outputs = model(**model_inputs)  # Model forward pass, outputs: CausalLMOutputWithPast
            loss = outputs.loss  # Scalar tensor or None
            if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                loss = loss / accumulation_steps  # Scale loss for accumulation
                scaler.scale(loss).backward()  # Backward pass with scaled loss
            else:
                print(f"Skipping batch {step} due to invalid loss: {loss}")
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            if scaler._scale is not None:  # Check if scale is set
                scaler.unscale_(optimizer)  # Unscale gradients
            # Debug: Gradient check
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()  # Compute norm
                    total_grad_norm += grad_norm
                    if grad_norm > 100 or grad_norm < 1e-6:
                        print(f"Warning: Unusual grad norm for {name}: {grad_norm:.4f}")
            print(f"Batch {step}: Total grad norm before clip: {total_grad_norm:.4f}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            scaler.step(optimizer)  # Optimizer step
            scaler.update()  # Update scaler
            optimizer.zero_grad(set_to_none=True)  # Zero gradients
        
        total_loss += loss.item() * accumulation_steps if loss is not None else 0  # Accumulate loss
        torch.cuda.empty_cache()  # Clear cache
        gc.collect()  # Garbage collect
    return total_loss / max(1, len(dataloader))  # Average loss

def validate_model(model, dataloader, device, vision_model, tokenizer, epoch_num, skip_generation=False):
    # Validate the model
    # Inputs: similar to train_epoch, plus tokenizer (LlamaTokenizer), skip_generation (bool)
    # Output: avg_loss (float), metrics (dict with 'bleu', 'meteor', 'rougeL', 'cider')
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total loss
    bleu_scores = []  # List for BLEU scores
    meteor_scores = []  # List for METEOR scores
    rouge_scores = []  # List for ROUGE scores
    cider_scores = []  # List for CIDEr scores
    cider_scorer = Cider()  # CIDEr scorer instance
    smoothing = SmoothingFunction().method4  # Smoothing for BLEU
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)  # ROUGE scorer
    skipped_batches = 0  # Count skipped batches
    skip_reasons = {'invalid_labels': 0, 'invalid_pixel_values': 0, 'vision_error': 0, 'loss_invalid': 0, 'generation_error': 0, 'no_valid_captions': 0}  # Dict for skip reasons
    sample_captions = []  # List to collect sample captions for saving
    
    if len(dataloader) == 0:
        print(f"Validation Epoch {epoch_num}: Empty dataloader, skipping validation")
        return 0.0, {'bleu': 0.0, 'meteor': 0.0, 'rougeL': 0.0, 'cider': 0.0}
    
    # Debug: Invalid validation check
    if len(dataloader.dataset) < 10:
        print(f"Warning: Small validation set size: {len(dataloader.dataset)}")
    
    with torch.no_grad():  # No gradient context
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validating Epoch {epoch_num}")):
            # Loop over validation batches
            batch_start_time = time.time()  # Start timing
            batch_tensor = {k: v.to(device, dtype=torch.long if k in ['input_ids', 'labels'] else torch.float16) 
                           for k, v in batch.items() if k not in ['captions', 'image_id']}  # Move to device
            captions = batch['captions']  # list of list of str
            image_ids = batch['image_id']  # list of int
            
            batch_size = batch_tensor['pixel_values'].size(0)  # Batch size
            print(f"Batch {batch_idx}: Batch size={batch_size}")
            
            labels = batch_tensor['labels']  # [batch, max_length]
            non_pad_mask = (labels != tokenizer.pad_token_id) & (labels != -100)  # Mask for valid labels
            if not torch.any(non_pad_mask):
                print(f"Batch {batch_idx}: Skipping due to invalid labels")
                skip_reasons['invalid_labels'] += 1
                skipped_batches += 1
                continue
            
            encoder_output = None
            visual_attention_mask = None
            if 'pixel_values' in batch_tensor:
                try:
                    vision_model_single = vision_model.module if isinstance(vision_model, nn.DataParallel) else vision_model
                    vision_model_single.to(device, dtype=torch.float16)
                    if torch.isnan(batch_tensor['pixel_values']).any() or torch.isinf(batch_tensor['pixel_values']).any():
                        print(f"Batch {batch_idx}: Skipping due to invalid pixel values")
                        skip_reasons['invalid_pixel_values'] += 1
                        skipped_batches += 1
                        continue
                    vision_outputs = vision_model_single(pixel_values=batch_tensor['pixel_values'])  # Extract features
                    encoder_output = vision_outputs.last_hidden_state  # [batch, num_visual_features, visual_feature_size]
                    # Debug: Visual Feature Verification
                    print(f"Batch {batch_idx}: Visual features mean={encoder_output.mean().item():.4f}, std={encoder_output.std().item():.4f}")
                    model_config = get_model_config(model)
                    expected_shape = (batch_size, model_config.num_visual_features, model_config.visual_feature_size)
                    if encoder_output.shape != expected_shape:
                        if encoder_output.shape[1] > model_config.num_visual_features:
                            encoder_output = encoder_output[:, :model_config.num_visual_features, :].to(torch.float16)
                        elif encoder_output.shape[1] < model_config.num_visual_features:
                            pad = torch.zeros(
                                batch_size,
                                model_config.num_visual_features - encoder_output.shape[1],
                                model_config.visual_feature_size,
                                dtype=torch.float16,
                                device=device
                            )
                            encoder_output = torch.cat([encoder_output, pad], dim=1)
                    visual_attention_mask = torch.ones(
                        batch_size, encoder_output.size(1), dtype=torch.float16, device=device
                    )  # [batch, num_visual_features]
                    print(f"Batch {batch_idx}: encoder_output shape={encoder_output.shape}, visual_attention_mask shape={visual_attention_mask.shape}")
                except Exception as e:
                    print(f"Batch {batch_idx}: Vision processing error: {e}")
                    skip_reasons['vision_error'] += 1
                    skipped_batches += 1
                    continue
            
            model_inputs = {
                'input_ids': batch_tensor['input_ids'],  # [batch, max_length]
                'attention_mask': batch_tensor['attention_mask'],  # [batch, max_length]
                'labels': batch_tensor['labels'],  # [batch, max_length]
                'encoder_output': encoder_output,  # [batch, num_visual_features, visual_feature_size] or None
                'visual_attention_mask': visual_attention_mask  # [batch, num_visual_features] or None
            }
            print(f"Batch {batch_idx}: Forward pass inputs: input_ids shape={batch_tensor['input_ids'].shape}, "
                  f"attention_mask shape={batch_tensor['attention_mask'].shape}, "
                  f"labels shape={batch_tensor['labels'].shape}")
            
            with autocast('cuda', dtype=torch.float16):  # Mixed precision
                outputs = model(**model_inputs)  # Forward pass
                loss = outputs.loss  # Scalar or None
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    print(f"Batch {batch_idx}: Skipping due to invalid loss")
                    skip_reasons['loss_invalid'] += 1
                    skipped_batches += 1
                    continue
                total_loss += loss.item()  # Accumulate loss
            
            if skip_generation:
                print(f"Batch {batch_idx}: Skipping generation for debugging, time taken={time.time() - batch_start_time:.2f}s")
                continue
            
            bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1  # BOS token
            prompt_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)  # [batch, 1]
            prompt_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)  # [batch, 1]
            
            if encoder_output is None or visual_attention_mask is None:
                print(f"Batch {batch_idx}: Skipping due to None encoder_output or visual_attention_mask")
                skip_reasons['vision_error'] += 1
                skipped_batches += 1
                continue
            
            print(f"Batch {batch_idx}: Generation inputs: input_ids shape={prompt_input_ids.shape}, "
                  f"attention_mask shape={prompt_attention_mask.shape}, "
                  f"encoder_output shape={encoder_output.shape}, "
                  f"visual_attention_mask shape={visual_attention_mask.shape}")
            
            try:
                print(f"Batch {batch_idx}: Calling model.generate with num_beams=1")
                preds = model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    encoder_output=encoder_output,
                    visual_attention_mask=visual_attention_mask,
                    max_new_tokens=32,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    use_cache=True
                )  # Generated token IDs [batch, seq_len]
                
                cider_refs = {}  # Dict for CIDEr references
                cider_hyps = {}  # Dict for CIDEr hypotheses
                for pred, ref_captions, img_id in zip(preds, captions, image_ids):
                    pred_text = tokenizer.decode(pred, skip_special_tokens=True).strip()  # Decode prediction
                    ref_texts = [ref.strip() for ref in ref_captions if ref and ref != "No caption available" and len(ref.strip()) > 0]  # Clean references
                    if not ref_texts:
                        print(f"Batch {batch_idx}: Skipping image {img_id} due to no valid captions")
                        skip_reasons['no_valid_captions'] += 1
                        continue
                    
                    pred_text = pred_text if pred_text else "empty"  # Handle empty prediction
                    print(f"Batch {batch_idx}: Generated: '{pred_text}', References: {ref_texts}")
                    print(f"Batch {batch_idx}: Sample prediction for image {img_id}: {pred_text[:50]}, time taken={time.time() - batch_start_time:.2f}s")
                    cider_refs[str(img_id)] = ref_texts  # Add to CIDEr refs
                    cider_hyps[str(img_id)] = [pred_text]  # Add to CIDEr hyps
                    
                    # Save sample caption
                    sample_captions.append({
                        'image_id': img_id,
                        'image_path': os.path.join(dataloader.dataset.image_path, self.image_info[img_id]['file_name']),
                        'generated_caption': pred_text,
                        'reference_captions': ref_texts
                    })
                    
                    try:
                        # Metric Computation: BLEU-4 with smoothing
                        bleu = sentence_bleu([ref.split() for ref in ref_texts], pred_text.split(), smoothing_function=smoothing)  # Compute BLEU
                        bleu_scores.append(bleu)
                        
                        # METEOR
                        meteor = meteor_score([ref.split() for ref in ref_texts], pred_text.split())  # Compute METEOR
                        meteor_scores.append(meteor)
                        
                        # ROUGE-L average fmeasure
                        rouge_total = 0
                        for ref in ref_texts:
                            rouge = rouge_scorer_obj.score(ref, pred_text)['rougeL'].fmeasure  # Compute ROUGE for each ref
                            rouge_total += rouge
                        rouge_scores.append(rouge_total / len(ref_texts))  # Average ROUGE
                    except Exception as e:
                        print(f"Batch {batch_idx}: Metric computation error for image {img_id}: {e}")
                        continue
                
                if cider_refs and cider_hyps:
                    cider_score, _ = cider_scorer.compute_score(cider_refs, cider_hyps)  # Compute CIDEr
                    cider_scores.append(cider_score)
                
            except Exception as e:
                print(f"Batch {batch_idx}: Generation error: {e}, time taken={time.time() - batch_start_time:.2f}s")
                skip_reasons['generation_error'] += 1
                skipped_batches += 1
                continue
            finally:
                torch.cuda.empty_cache()  # Clear cache
                gc.collect()  # Garbage collect
    
    # Save sample captions to file
    sample_captions_path = f"/home/yazan/VisualLlama/sample_captions_{epoch_num}.json"
    with open(sample_captions_path, 'w') as f:
        json.dump(sample_captions, f, indent=2)
    print(f"Sample captions saved to {sample_captions_path}")
    
    avg_loss = total_loss / max(1, len(dataloader) - skipped_batches) if len(dataloader) > skipped_batches else 0.0  # Average loss
    avg_bleu = sum(bleu_scores) / max(1, len(bleu_scores)) if bleu_scores else 0.0  # Average BLEU
    avg_meteor = sum(meteor_scores) / max(1, len(meteor_scores)) if meteor_scores else 0.0  # Average METEOR
    avg_rouge = sum(rouge_scores) / max(1, len(rouge_scores)) if rouge_scores else 0.0  # Average ROUGE
    avg_cider = sum(cider_scores) / max(1, len(cider_scores)) if cider_scores else 0.0  # Average CIDEr
    print(f"Validation Epoch {epoch_num}: Skipped {skipped_batches}/{len(dataloader)} batches")
    print(f"Skip reasons: {skip_reasons}")
    print(f"Avg metrics: BLEU={avg_bleu:.4f}, METEOR={avg_meteor:.4f}, ROUGE-L={avg_rouge:.4f}, CIDEr={avg_cider:.4f}")
    return avg_loss, {'bleu': avg_bleu, 'meteor': avg_meteor, 'rougeL': avg_rouge, 'cider': avg_cider}

def save_plots(training_history, save_dir="/home/yazan/VisualLlama/plots"):
    # Save training plots
    # Input: training_history (list of dicts), save_dir (str)
    # Output: None, saves PNG files
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))  # Create figure
    train_losses = [h['train_loss'] for h in training_history if 'train_loss' in h]  # Extract train losses
    val_losses = [h['val_loss'] for h in training_history if 'val_loss' in h and h['val_loss'] != 0]  # Extract val losses
    epochs = [h['epoch'] for h in training_history if 'train_loss' in h]  # Extract epochs for train
    plt.plot(epochs, train_losses, label='Train Loss')  # Plot train loss
    if val_losses:
        val_epochs = [h['epoch'] for h in training_history if 'val_loss' in h and h['val_loss'] != 0]  # Epochs for val
        plt.plot(val_epochs, val_losses, label='Validation Loss')  # Plot val loss
    plt.xlabel('Epoch')  # Set x label
    plt.ylabel('Loss')  # Set y label
    plt.title('Training and Validation Loss')  # Set title
    plt.legend()  # Add legend
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))  # Save plot
    plt.close()  # Close figure
    
    plt.figure(figsize=(10, 5))  # Create new figure
    bleu_scores = [h['bleu'] for h in training_history if 'bleu' in h and h['bleu'] != 0]  # Extract BLEU
    meteor_scores = [h['meteor'] for h in training_history if 'meteor' in h and h['meteor'] != 0]  # Extract METEOR
    rouge_scores = [h['rougeL'] for h in training_history if 'rougeL' in h and h['rougeL'] != 0]  # Extract ROUGE
    cider_scores = [h['cider'] for h in training_history if 'cider' in h and h['cider'] != 0]  # Extract CIDEr
    if bleu_scores or meteor_scores or rouge_scores or cider_scores:
        val_epochs = [h['epoch'] for h in training_history if 'bleu' in h and h['bleu'] != 0]  # Val epochs
        if bleu_scores:
            plt.plot(val_epochs, bleu_scores, label='BLEU')  # Plot BLEU
        if meteor_scores:
            plt.plot(val_epochs, meteor_scores, label='METEOR')  # Plot METEOR
        if rouge_scores:
            plt.plot(val_epochs, rouge_scores, label='ROUGE-L')  # Plot ROUGE
        if cider_scores:
            plt.plot(val_epochs, cider_scores, label='CIDEr')  # Plot CIDEr
        plt.xlabel('Epoch')  # Set x label
        plt.ylabel('Score')  # Set y label
        plt.title('Validation Metrics')  # Set title
        plt.legend()  # Add legend
        plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))  # Save plot
        plt.close()  # Close figure

def save_training_history(training_history, save_path="/home/yazan/VisualLlama/training_history.json"):
    # Save training history to JSON
    # Input: training_history (list of dicts), save_path (str)
    # Output: None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'w') as f:
            json.dump(training_history, f, indent=2)  # Dump to file
    except OSError as e:
        print(f"Error saving training history to {save_path}: {e}")

def load_training_history(history_path="/home/yazan/VisualLlama/training_history.json"):
    # Load training history from JSON
    # Input: history_path (str)
    # Output: list of dicts or empty list if error
    try:
        with open(history_path, 'r') as f:
            return json.load(f)  # Load from file
    except:
        return []

def signal_handler(sig, frame):
    # Signal handler for interrupts
    # Input: sig (int), frame (frame)
    # Output: None, prints message and exits
    print("\nTraining interrupted")
    torch.cuda.empty_cache()
    gc.collect()
    sys.exit(0)

def main():
    # Main function
    # Input: None
    # Output: None, runs the training process
    signal.signal(signal.SIGINT, signal_handler)  # Set signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    
    dataset_size_percentage = 0.0  # 0.01% Dataset
    seeds = [0]  # List of seeds
    config_path = 'configs/visualgpt_llama.yaml'  # Config path
    config = load_config(config_path)  # Load config
    
    validate_interval = config['train']['validate_interval']  # Validation interval
    
    val_annotation_path = config['dataset']['coco'].get('val_annotation_path', '/home/yazan/VisualLlama/coco/annotations/captions_val2017.json')  # Val annotation path
    if not os.path.exists(val_annotation_path):
        raise FileNotFoundError(f"Validation annotation file {val_annotation_path} not found. Please download annotations_trainval2017.zip and extract to /home/yazan/VisualLlama/coco/annotations/")
    
    clear_cache("/home/yazan/VisualLlama/cache")  # Clear cache
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Set CUDA launch blocking
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device
    
    all_metrics = {seed: {'bleu': [], 'meteor': [], 'rougeL': [], 'cider': [], 'val_loss': []} for seed in seeds}  # Dict for metrics
    
    for seed in seeds:
        print(f"\nRunning experiment with {dataset_size_percentage}% dataset, seed {seed}")
        
        np.random.seed(seed)  # Set seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        config['dataset']['coco']['annotation_path'] = f"/home/yazan/VisualLlama/coco_subsets/coco_{dataset_size_percentage:.1f}percent_seed{seed}/annotations/captions_train2017_{dataset_size_percentage:.1f}percent_seed{seed}.json"
        config['dataset']['coco']['image_path'] = f"/home/yazan/VisualLlama/coco_subsets/coco_{dataset_size_percentage:.1f}percent_seed{seed}/train2017"
        
        torch.backends.cudnn.deterministic = False  # Set CuDNN settings
        torch.backends.cudnn.benchmark = True
        
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                config['model']['llama_model_path'],
                local_files_only=os.path.isdir(config['model']['llama_model_path']),
                token=True if not os.path.isdir(config['model']['llama_model_path']) else False
            )  # Load tokenizer
        except Exception as e:
            print(f"Error loading tokenizer from {config['model']['llama_model_path']}: {e}")
            print("Ensure the path is a valid local directory with tokenizer files or a Hugging Face model ID.")
            print("For local paths, required files include: tokenizer.json, vocab.json, tokenizer_config.json")
            print("For Hugging Face models, ensure you have access and are logged in via `huggingface-cli login`.")
            raise
        
        tokenizer.padding_side = 'left'  # Set padding side
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.pad_token = '[PAD]'
            tokenizer.pad_token_id = 0
        print(f"Tokenizer: bos_token_id={tokenizer.bos_token_id}, pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
        # Debug: Tokenizer config check
        print(f"Tokenizer vocab size={tokenizer.vocab_size}, padding_side={tokenizer.padding_side}")
        
        processor = CLIPImageProcessor.from_pretrained(config['model']['vit'])  # Load image processor
        vision_model = CLIPVisionModel.from_pretrained(config['model']['vit'], dtype=torch.float16).to(device)  # Load vision model
        if config['model'].get('freeze_vit', True):
            vision_model.eval()  # Set to eval if frozen
        
        train_dataset = CocoDataset(
            annotation_path=config['dataset']['coco']['annotation_path'],
            image_path=config['dataset']['coco']['image_path'],
            processor=processor,
            tokenizer=tokenizer,
            max_length=config['model'].get('max_length', 512),
            cache_dir="/home/yazan/VisualLlama/cache"
        )  # Create train dataset
        
        val_dataset = CocoDataset(
            annotation_path="/home/yazan/VisualLlama/coco/annotations/captions_train2017.json",
            image_path="/home/yazan/VisualLlama/coco/train2017",
            processor=processor,
            tokenizer=tokenizer,
            max_length=config['model'].get('max_length', 512),
            cache_dir="/home/yazan/VisualLlama/cache",
            karpathy_val=True,
            val_size=10,  # Reduced for simple validation to save time
            seed=seed
        )  # Create val dataset
        
        test_dataset = CocoDataset(
            annotation_path=val_annotation_path,
            image_path=config['dataset']['coco'].get('val_image_path', '/home/yazan/VisualLlama/coco/val2017'),
            processor=processor,
            tokenizer=tokenizer,
            max_length=config['model'].get('max_length', 512),
            cache_dir="/home/yazan/VisualLlama/cache"
        )  # Create test dataset
        
        total_size = len(train_dataset)  # Total train size
        if total_size == 0:
            print("Error: Training dataset is empty. Check annotation file and image paths.")
            continue
        
        train_size = int(0.9 * total_size)  # 90% for train
        train_indices = list(range(total_size))
        np.random.shuffle(train_indices)  # Shuffle indices
        train_indices = train_indices[:train_size]  # Select train indices
        train_dataset = Subset(train_dataset, train_indices)  # Subset for train
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=custom_data_collator,
            num_workers=4,
            pin_memory=True
        )  # Train loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_data_collator,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )  # Val loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_data_collator,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )  # Test loader
        print(f"Training dataset size: {len(train_dataset)} samples ({dataset_size_percentage}%)")
        print(f"Validation dataset size: {len(val_dataset)} samples (Karpathy validation)")
        print(f"Test dataset size: {len(test_dataset)} samples (Karpathy test)")
        
        model = create_model(config, config['model']['llama_model_path'])  # Create model
        model.to(device)  # Move to device
        model.resize_token_embeddings(tokenizer.vocab_size)  # Resize embeddings
        model.config.pad_token_id = tokenizer.pad_token_id  # Set pad token
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=0.01)  # Optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['epochs'], eta_min=1e-6)  # Scheduler
        
        training_history = load_training_history(f"/home/yazan/VisualLlama/training_history_seed_{seed}_percentage_{dataset_size_percentage:.1f}.json")  # Load history
        best_val_loss = float('inf')  # Best val loss
        patience = config['train']['patience']  # Patience for early stopping
        counter = 0  # Early stopping counter
        num_epochs = config['train'].get('epochs', 5)  # Number of epochs
        accumulation_steps = config['train']['accumulation_steps']  # Accumulation steps
        
        for epoch in range(1, num_epochs + 1):
            # Loop over epochs
            torch.cuda.empty_cache()  # Clear cache
            train_loss = train_epoch(
                model, train_loader, optimizer, device, vision_model,
                epoch_num=epoch, accumulation_steps=accumulation_steps
            )  # Train epoch
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'epoch_time': time.time(),
                'seed': seed,
                'percentage': dataset_size_percentage
            })  # Append to history
            
            val_loss, metrics = None, None
            if epoch % validate_interval == 0:
                val_loss, metrics = validate_model(model, val_loader, device, vision_model, tokenizer, epoch_num=epoch, skip_generation=False)  # Validate
                training_history.append({
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'bleu': metrics['bleu'],
                    'meteor': metrics['meteor'],
                    'rougeL': metrics['rougeL'],
                    'cider': metrics['cider'],
                    'seed': seed,
                    'percentage': dataset_size_percentage
                })  # Append val metrics
                all_metrics[seed]['bleu'].append(metrics['bleu'])  # Append to all metrics
                all_metrics[seed]['meteor'].append(metrics['meteor'])
                all_metrics[seed]['rougeL'].append(metrics['rougeL'])
                all_metrics[seed]['cider'].append(metrics['cider'])
                all_metrics[seed]['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}" + 
                  (f", Val Loss={val_loss:.4f}, BLEU={metrics['bleu']:.4f}, METEOR={metrics['meteor']:.4f}, ROUGE-L={metrics['rougeL']:.4f}, CIDEr={metrics['cider']:.4f}" 
                   if val_loss is not None else ""))
            
            if val_loss is not None and val_loss < best_val_loss and val_loss != 0:
                best_val_loss = val_loss  # Update best loss
                counter = 0  # Reset counter
                best_dir = f"/home/yazan/VisualLlama/checkpoints/best_seed_{seed}_percentage_{dataset_size_percentage:.1f}"  # Best checkpoint dir
                os.makedirs(best_dir, exist_ok=True)
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'seed': seed,
                        'percentage': dataset_size_percentage
                    }, os.path.join(best_dir, f"checkpoint_seed_{seed}_percentage_{dataset_size_percentage:.1f}.pt"))  # Save checkpoint
                    tokenizer.save_pretrained(best_dir)  # Save tokenizer
                except OSError as e:
                    print(f"Error saving best checkpoint to {best_dir}: {e}")
            else:
                counter += 1  # Increment counter
                if counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs")
                    break
            
            save_training_history(training_history, f"/home/yazan/VisualLlama/training_history_seed_{seed}_percentage_{dataset_size_percentage:.1f}.json")  # Save history
            save_plots(training_history, f"/home/yazan/VisualLlama/plots_seed_{seed}_percentage_{dataset_size_percentage:.1f}")  # Save plots
            scheduler.step()  # Scheduler step
            torch.cuda.empty_cache()  # Clear cache
            gc.collect()  # Garbage collect
        
        # Save plots after all training epochs, before test
        save_plots(training_history, f"/home/yazan/VisualLlama/plots_seed_{seed}_percentage_{dataset_size_percentage:.1f}_final")
        
        final_dir = f"/home/yazan/VisualLlama/checkpoints/final_seed_{seed}_percentage_{dataset_size_percentage:.1f}"  # Final checkpoint dir
        os.makedirs(final_dir, exist_ok=True)
        try:
            torch.save({
                'epoch': 'final',
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'seed': seed,
                'percentage': dataset_size_percentage
            }, os.path.join(final_dir, f"checkpoint_seed_{seed}_percentage_{dataset_size_percentage:.1f}.pt"))  # Save final checkpoint
            tokenizer.save_pretrained(final_dir)  # Save tokenizer
        except OSError as e:
            print(f"Error saving final checkpoint to {final_dir}: {e}")
        
        test_loss, test_metrics = validate_model(model, test_loader, device, vision_model, tokenizer, epoch_num=f"final_seed_{seed}", skip_generation=False)  # Test evaluation
        print(f"Test Results (Seed {seed}): Test Loss={test_loss:.4f}, BLEU={test_metrics['bleu']:.4f}, "
              f"METEOR={test_metrics['meteor']:.4f}, ROUGE-L={test_metrics['rougeL']:.4f}, CIDEr={test_metrics['cider']:.4f}")
        
        training_history.append({
            'epoch': 'final',
            'test_loss': test_loss,
            'test_bleu': test_metrics['bleu'],
            'test_meteor': test_metrics['meteor'],
            'test_rougeL': test_metrics['rougeL'],
            'test_cider': test_metrics['cider'],
            'seed': seed,
            'percentage': dataset_size_percentage
        })  # Append test metrics
        save_training_history(training_history, f"/home/yazan/VisualLlama/training_history_seed_{seed}_percentage_{dataset_size_percentage:.1f}.json")  # Save history
        
        torch.cuda.empty_cache()  # Clear cache
        gc.collect()  # Garbage collect
    
    avg_metrics = {'bleu': [], 'meteor': [], 'rougeL': [], 'cider': [], 'val_loss': []}  # Average metrics dict
    for seed in seeds:
        if all_metrics[seed]['bleu']:
            avg_metrics['bleu'].append(np.mean(all_metrics[seed]['bleu']))  # Mean BLEU
            avg_metrics['meteor'].append(np.mean(all_metrics[seed]['meteor']))  # Mean METEOR
            avg_metrics['rougeL'].append(np.mean(all_metrics[seed]['rougeL']))  # Mean ROUGE
            avg_metrics['cider'].append(np.mean(all_metrics[seed]['cider']))  # Mean CIDEr
            avg_metrics['val_loss'].append(np.mean(all_metrics[seed]['val_loss']))  # Mean val loss
    
    if avg_metrics['bleu']:
        print(f"\nAverage Metrics for {dataset_size_percentage}% Dataset (across {len(avg_metrics['bleu'])} seeds):")
        print(f"Average BLEU: {np.mean(avg_metrics['bleu']):.4f}")
        print(f"Average METEOR: {np.mean(avg_metrics['meteor']):.4f}")
        print(f"Average ROUGE-L: {np.mean(avg_metrics['rougeL']):.4f}")
        print(f"Average CIDEr: {np.mean(avg_metrics['cider']):.4f}")
        print(f"Average Validation Loss: {np.mean(avg_metrics['val_loss']):.4f}")
        
        avg_metrics_path = f"/home/yazan/VisualLlama/average_metrics_percentage_{dataset_size_percentage:.1f}.json"  # Path for avg metrics
        os.makedirs(os.path.dirname(avg_metrics_path), exist_ok=True)
        try:
            with open(avg_metrics_path, 'w') as f:
                json.dump({
                    'percentage': dataset_size_percentage,
                    'avg_bleu': np.mean(avg_metrics['bleu']),
                    'avg_meteor': np.mean(avg_metrics['meteor']),
                    'avg_rougeL': np.mean(avg_metrics['rougeL']),
                    'avg_cider': np.mean(avg_metrics['cider']),
                    'avg_val_loss': np.mean(avg_metrics['val_loss']),
                    'seeds_used': len(avg_metrics['bleu'])
                }, f, indent=2)  # Dump avg metrics
        except OSError as e:
            print(f"Error saving average metrics to {avg_metrics_path}: {e}")
    
    clear_cache("/home/yazan/VisualLlama/cache")  # Clear cache
    checkpoint_root = "/home/yazan/VisualLlama/checkpoints"  # Checkpoint root
    for dir_name in os.listdir(checkpoint_root):
        dir_path = os.path.join(checkpoint_root, dir_name)
        if os.path.isdir(dir_path) and "best_seed_" not in dir_name and "final_seed_" not in dir_name:
            shutil.rmtree(dir_path, ignore_errors=True)  # Delete intermediate dirs
            print(f"Deleted intermediate checkpoint directory: {dir_path}")

if __name__ == "__main__":
    os.makedirs("/home/yazan/VisualLlama/checkpoints", exist_ok=True)  # Create checkpoints dir
    os.makedirs("/home/yazan/VisualLlama/cache", exist_ok=True)  # Create cache dir
    
    try:
        main()  # Run main
    except OSError as e:
        print(f"Error during training: {e}")
    finally:
        torch.cuda.empty_cache()  # Clear cache
        gc.collect()  # Garbage collect