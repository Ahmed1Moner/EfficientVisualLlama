import torch
from transformers import BitsAndBytesConfig

def get_quantization_config():
    """
    Returns the proper quantization configuration for 4-bit loading
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def get_model_loading_kwargs(use_quantization=True):
    """
    Returns the proper keyword arguments for model loading
    """
    kwargs = {
        "device_map": "auto",
        "use_safetensors": True,
        "low_cpu_mem_usage": True
    }
    
    if use_quantization:
        kwargs["quantization_config"] = get_quantization_config()
    else:
        kwargs["torch_dtype"] = torch.float16
        
    return kwargs