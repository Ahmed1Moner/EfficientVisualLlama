import torch
import os
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from .safetensors_config import get_model_loading_kwargs

def load_local_llama_model(model_path="/home/yazan/Llama-2-13b-hf", use_quantization=True):
    """Load the locally downloaded LLaMA model and tokenizer"""
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    print("Loading tokenizer from local directory...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
    print("Loading model from local directory...")
    
    # Get the proper loading kwargs
    kwargs = get_model_loading_kwargs(use_quantization)
    
    try:
        model = LlamaModel.from_pretrained(model_path, **kwargs)
        print("✓ Model loaded successfully with safetensors")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        
        # Fallback: try without safetensors
        print("Trying fallback loading without safetensors...")
        kwargs.pop("use_safetensors", None)
        model = LlamaModel.from_pretrained(model_path, **kwargs)
        print("✓ Model loaded successfully with fallback method")
    
    return model, tokenizer

def create_visual_llama_from_local(base_model_path="/home/yazan/Llama-2-13b-hf"):
    """Create a VisualLLaMA model from a local LLaMA model"""
    from .configuration_llama_visualgpt import VisualLlamaConfig
    from .llama_decoder_visualgpt import VisualLlamaForCausalLM
    
    # Load the base config
    base_config = LlamaConfig.from_pretrained(base_model_path)
    
    # Create visual config
    visual_config = VisualLlamaConfig(
        **base_config.to_dict(),
        visual_feature_size=768,
        num_visual_features=3,
        tau=0.5
    )
    
    # Create the visual model
    visual_model = VisualLlamaForCausalLM(visual_config)
    
    # Load weights from the base model using safetensors
    kwargs = get_model_loading_kwargs(use_quantization=False)
    base_model = LlamaModel.from_pretrained(base_model_path, **kwargs)
    
    # Load state dict with strict=False to handle architecture differences
    visual_model.model.load_state_dict(base_model.state_dict(), strict=False)
    
    return visual_model