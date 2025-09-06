# import torch
# import torch.nn as nn
# from transformers import LlamaForCausalLM, LlamaTokenizer, CLIPVisionModel, CLIPImageProcessor, AutoConfig
# from PIL import Image
# import os
# import yaml
# from models.transformer import VisualLlamaConfig, VisualLlamaForCausalLM

# def load_config(config_path='configs/visualgpt_llama2.yaml'):
#     """Load configuration from YAML file and return VisualLlamaConfig object"""
#     try:
#         with open(config_path, 'r') as f:
#             config_dict = yaml.safe_load(f)
#         return VisualLlamaConfig(**config_dict)
#     except FileNotFoundError:
#         print(f"⚠️  Config file not found at {config_path}, using default configuration")
#         # Return a default config if file not found
#         return VisualLlamaConfig()

# def validate_environment():
#     """Validate that all required components are available"""
#     print("🔍 Validating environment...")
    
#     # Check CUDA
#     cuda_available = torch.cuda.is_available()
#     print(f"✅ CUDA available: {cuda_available}")
#     if cuda_available:
#         print(f"   GPU: {torch.cuda.get_device_name(0)}")
#         print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
#     # Check PyTorch version
#     print(f"✅ PyTorch version: {torch.__version__}")
    
#     return True

# def validate_llama_model(model_path):
#     """Validate LLaMA-2 model loading"""
#     print(f"\n🔍 Validating LLaMA-2 model at {model_path}...")
    
#     try:
#         # Check if model path exists
#         if not os.path.exists(model_path):
#             print(f"❌ Model path does not exist: {model_path}")
#             return False
        
#         # Try loading tokenizer
#         tokenizer = LlamaTokenizer.from_pretrained(model_path)
#         tokenizer.pad_token = tokenizer.eos_token
#         print("✅ LLaMA tokenizer loaded successfully")
        
#         # Try loading model
#         model = LlamaForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.float16,
#             device_map="auto" if torch.cuda.is_available() else None
#         )
#         print("✅ LLaMA model loaded successfully")
        
#         # Test tokenization
#         test_text = "Hello, how are you?"
#         tokens = tokenizer(test_text, return_tensors="pt")
#         print(f"✅ Tokenization test passed: {len(tokens['input_ids'][0])} tokens")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Error loading LLaMA model: {e}")
#         return False

# def validate_clip_model():
#     """Validate CLIP model loading"""
#     print(f"\n🔍 Validating CLIP model...")
    
#     try:
#         # Try loading CLIP processor and model
#         processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
#         print("✅ CLIP model loaded successfully")
        
#         # Test with a dummy image
#         dummy_image = Image.new('RGB', (224, 224), color='red')
#         inputs = processor(images=dummy_image, return_tensors="pt")
#         print("✅ CLIP processor test passed")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Error loading CLIP model: {e}")
#         return False

# def validate_integration():
#     """Validate that the integration works"""
#     print(f"\n🔍 Validating integration...")
    
#     try:
#         config = load_config()
        
#         # Test basic model components - using the new LLaMA-based model
#         model = VisualLlamaForCausalLM(config)
#         print("✅ VisualLLaMA model initialized successfully")
        
#         # Test model on CPU first to save memory
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model = model.to(device)
#         print(f"✅ Model moved to {device}")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Integration error: {e}")
#         return False

# def main():
#     """Main validation function"""
#     print("=" * 60)
#     print("🧪 VisualLLaMA + LLaMA-2 + CLIP Validation Script")
#     print("=" * 60)
    
#     # Load config to get model path (if available)
#     try:
#         config = load_config()
#         llama_model_path = config.llama_model_path
#     except:
#         print("⚠️  Using default model path")
#         llama_model_path = "/home/yazan/Llama-2-13b-hf"
    
#     # Run all validations
#     success = True
#     success &= validate_environment()
#     success &= validate_llama_model(llama_model_path)
#     success &= validate_clip_model()
#     success &= validate_integration()
    
#     print("\n" + "=" * 60)
#     if success:
#         print("🎉 All validations passed! Your setup is ready.")
#         print("\nNext steps:")
#         print("1. Prepare your dataset")
#         print("2. Run: python train.py")
#         print("3. Test with: python inference.py")
#     else:
#         print("❌ Some validations failed. Please check the errors above.")
    
#     print("=" * 60)

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, CLIPVisionModel, CLIPImageProcessor, AutoConfig
from PIL import Image
import os
import yaml
import gc
from models.transformer import VisualLlamaConfig, VisualLlamaForCausalLM

def load_config(config_path='configs/visualgpt_llama.yaml'):
    """Load configuration from YAML file and return VisualLlamaConfig object"""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return VisualLlamaConfig(**config_dict)
    except FileNotFoundError:
        print(f"⚠️  Config file not found at {config_path}, using default configuration")
        # Return a default config if file not found
        return VisualLlamaConfig()

def validate_environment():
    """Validate that all required components are available"""
    print("🔍 Validating environment...")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"✅ CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check PyTorch version
    print(f"✅ PyTorch version: {torch.__version__}")
    
    return True

def validate_llama_model(model_path):
    """Validate LLaMA-2 model loading"""
    print(f"\n🔍 Validating LLaMA-2 model at {model_path}...")
    
    try:
        # Check if model path exists
        if not os.path.exists(model_path):
            print(f"❌ Model path does not exist: {model_path}")
            return False
        
        # Try loading tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ LLaMA tokenizer loaded successfully")
        
        # Try loading model with memory optimization
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        print("✅ LLaMA model loaded successfully")
        
        # Test tokenization
        test_text = "Hello, how are you?"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test passed: {len(tokens['input_ids'][0])} tokens")
        
        # Clean up to save memory
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
        # Try loading CLIP processor and model
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP model loaded successfully")
        
        # Test with a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        inputs = processor(images=dummy_image, return_tensors="pt")
        print("✅ CLIP processor test passed")
        
        # Clean up
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CLIP model: {e}")
        return False

def validate_integration():
    """Validate that the integration works"""
    print(f"\n🔍 Validating integration...")
    
    try:
        config = load_config()
        
        # Test basic model components - using the new LLaMA-based model
        # Load with memory optimization
        model = VisualLlamaForCausalLM.from_pretrained(
            config.llama_model_path,
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✅ VisualLLaMA model initialized successfully")
        
        # Test model on CPU first to save memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✅ Model moved to {device}")
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False

def main():
    """Main validation function"""
    print("=" * 60)
    print("🧪 VisualLLaMA + LLaMA-2 + CLIP Validation Script")
    print("=" * 60)
    
    # Load config to get model path (if available)
    try:
        config = load_config()
        llama_model_path = config.llama_model_path
    except:
        print("⚠️  Using default model path")
        llama_model_path = "/home/yazan/Llama-2-13b-hf"
    
    # Run all validations
    success = True
    success &= validate_environment()
    success &= validate_llama_model(llama_model_path)
    success &= validate_clip_model()
    success &= validate_integration()
    
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