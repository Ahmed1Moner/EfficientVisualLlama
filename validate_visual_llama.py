import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, CLIPVisionModel, CLIPImageProcessor, AutoConfig
from PIL import Image
import os
import re
import json
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import yaml
import gc
from models.transformer import VisualLlamaConfig, VisualLlamaForCausalLM
import sys
sys.stdout.reconfigure(encoding='utf-8')

def load_config(config_path='configs/visualgpt_llama.yaml'):
    """Load configuration from YAML file and return VisualLlamaConfig object"""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return VisualLlamaConfig(**config_dict)
    except FileNotFoundError:
        print(f"⚠️  Config file not found at {config_path}, using default configuration")
        return VisualLlamaConfig()

def validate_environment():
    """Validate that all required components are available"""
    print("🔍 Validating environment...")
    
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"✅ PyTorch version: {torch.__version__}")
    
    return True

def validate_llama_model(model_path):
    """Validate LLaMA-2 model loading"""
    print(f"\n🔍 Validating LLaMA-2 model at {model_path}...")
    
    try:
        if not os.path.exists(model_path):
            print(f"❌ Model path does not exist: {model_path}")
            return False
        
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ LLaMA tokenizer loaded successfully")
        
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        print("✅ LLaMA model loaded successfully")
        
        test_text = "Hello, how are you?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test passed: {len(tokens['input_ids'][0])} tokens")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading LLaMA model: {e}")
        return False

def validate_clip_model():
    """Validate CLIP model loading"""
    print(f"\n🔍 Validating CLIP model...")
    
    try:
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP model loaded successfully")
        
        dummy_image = Image.new('RGB', (224, 224), color='red')
        inputs = processor(images=dummy_image, return_tensors="pt")
        print("✅ CLIP processor test passed")
        
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CLIP model: {e}")
        return False

def validate_dimension_flow():
    """Check dimension flow between components"""
    print("\n🔍 Checking Dimension Flow...")
    
    config = VisualLlamaConfig()
    config.hidden_size = 4096
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.intermediate_size = 11008
    config.vocab_size = 32000
    config.pad_token_id = 0
    config.num_visual_features = 3
    
    model = VisualLlamaForCausalLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✅ Model hidden size: {config.hidden_size}")
    print(f"✅ Vocabulary size: {config.vocab_size}")
    
    dummy_input = torch.tensor([[1, 2, 3]], device=device)
    embeddings = model.model.embed_tokens(dummy_input)
    print(f"✅ Embedding input: {dummy_input.shape} -> output: {embeddings.shape}")
    
    dummy_hidden = torch.randn(1, 5, config.hidden_size, device=device)
    logits = model.lm_head(dummy_hidden)
    print(f"✅ LM head input: {dummy_hidden.shape} -> output: {logits.shape}")
    
    dummy_x = torch.randn(1, 5, config.hidden_size, device=device)
    dummy_encoder = torch.randn(1, config.num_visual_features, 10, config.hidden_size, device=device)
    
    cross_attn = model.model.layers[0].cross_attentions[0]
    cross_output = cross_attn(dummy_x, dummy_encoder[:, 0, :, :])
    print(f"✅ Cross attention input: {dummy_x.shape}, encoder: {dummy_encoder[:, 0, :, :].shape} -> output: {cross_output.shape}")
    
    print("✅ Dimension flow check successfully!")
    return True

def validate_integration():
    print(f"\n🔍 Validating integration...")
    try:
        config = load_config()
        
        config.hidden_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.intermediate_size = 128
        config.vocab_size = 1000
        config.pad_token_id = 0
        config.num_visual_features = 3
        config.max_position_embeddings = 2048
        
        print(f"Config: hidden_size={config.hidden_size}, num_attention_heads={config.num_attention_heads}")
        
        original_llama_path = getattr(config, "llama_model_path", None)
        config.llama_model_path = None

        model = VisualLlamaForCausalLM(config)
        print("✅ VisualLLaMA model created with random weights.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"✅ Model moved to {device}")

        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
                encoder_output = torch.randn(1, config.num_visual_features, 10, config.hidden_size, device=device)
                position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(dummy_input, dtype=torch.float32, device=device)
                
                output = model(
                    input_ids=dummy_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    encoder_output=encoder_output
                )
            print("✅ Forward pass successful.")
        except Exception as e_forward:
            print(f"❌ Forward pass error: {e_forward}")
            return False

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        config.llama_model_path = original_llama_path
        return True

    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False

def validate_dataset():
    """Validate COCO Minitrain dataset"""
    print(f"\n🔍 Validating COCO Minitrain dataset...")

    try:
        base_dir = "/home/yazan/VisualLlama/data"

        # --- Paths ---
        annotation_path = os.path.join(base_dir, "coco/annotations/minitrain2017.json")
        train_image_dir = os.path.join(base_dir, "coco_minitrain_25k/images/train2017")
        val_image_dir = os.path.join(base_dir, "coco/val2017")
        val_ann_path = os.path.join(base_dir, "coco/annotations/annotations/instances_val2017.json")

        # --- Check annotation file ---
        if not os.path.exists(annotation_path):
            print(f"❌ Annotation file not found: {annotation_path}")
            return False

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        print(f"✅ Annotation file loaded: {len(annotations['images'])} images, {len(annotations['annotations'])} annotations")

        # --- Check training images ---
        if not os.path.exists(train_image_dir):
            print(f"❌ Train image directory not found: {train_image_dir}")
            return False

        train_images = len([f for f in os.listdir(train_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✅ Found {train_images} training images in minitrain")

        # --- Check validation images ---
        if not os.path.exists(val_image_dir):
            print(f"❌ Validation image directory not found: {val_image_dir}")
            return False
        if not os.path.exists(val_ann_path):
            print(f"❌ Validation annotations not found: {val_ann_path}")
            return False

        val_images = len([f for f in os.listdir(val_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✅ Found {val_images} validation images")

        # --- Sample check with CLIP ---
        import random
        from transformers import CLIPImageProcessor
        from PIL import Image

        sample_size = min(5, len(annotations['images']))
        sample_images = random.sample(annotations['images'], sample_size)
        clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        for img_info in sample_images:
            img_path = os.path.join(train_image_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                print(f"❌ Image file not found: {img_path}")
                return False
            try:
                image = Image.open(img_path)
                inputs = clip_processor(images=image, return_tensors="pt")
                print(f"✅ Image loaded and processed: {img_info['file_name']}")
            except Exception as e:
                print(f"❌ Error processing image {img_info['file_name']}: {e}")
                return False

        print("✅ Dataset validation successful!")
        return True

    except Exception as e:
        print(f"❌ Dataset validation error: {e}")
        return False
        
def main():
    print("=" * 60)
    print("🧪 VisualLLaMA + LLaMA-2 + CLIP Validation Script")
    print("=" * 60)
    
    try:
        config = load_config()
        llama_model_path = config.llama_model_path
    except:
        print("⚠️  Using default model path")
        llama_model_path = "/home/yazan/Llama-2-13b-hf"
    
    success = True
    success &= validate_environment()
    success &= validate_llama_model(llama_model_path)
    success &= validate_clip_model()
    success &= validate_dimension_flow()
    success &= validate_integration()
    success &= validate_dataset()

    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All validations passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Prepare your dataset")
        print("2. Run: python train.py")
        print("3. Test with: python inference.py")
    else:
        print("❌ Some validations failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()