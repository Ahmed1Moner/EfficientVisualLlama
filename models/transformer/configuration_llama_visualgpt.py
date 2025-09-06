# configuration_llama_visualgpt.py
import os
from transformers import LlamaConfig

# Set a sensible default here for your environment (change as needed)
DEFAULT_LLAMA_MODEL_PATH = "/home/yazan/Llama-2-13b-hf"

class VisualLlamaConfig(LlamaConfig):
    
    def __init__(
        self,
        visual_feature_size=768,
        num_visual_features=3,
        tau=0.5,
        resid_dropout=0.1,
        attn_dropout=0.1,
        embd_dropout=0.1,
        llama_model_path=None,
        **kwargs
    ):
        # Initialize parent config with kwargs first
        super().__init__(**kwargs)

        # Visual-specific parameters
        self.visual_feature_size = visual_feature_size
        self.num_visual_features = num_visual_features
        self.tau = tau  # Alpha gate threshold parameter
        self.resid_dropout = resid_dropout
        self.attn_dropout = attn_dropout
        self.embd_dropout = embd_dropout

        # Resolve llama_model_path from supplied arg, environment variable, or default.
        resolved_path = llama_model_path
        if not resolved_path:
            resolved_path = os.environ.get("LLAMA_MODEL_PATH") or os.environ.get("MODEL_PATH")
        if not resolved_path:
            resolved_path = DEFAULT_LLAMA_MODEL_PATH

        # Ensure we store a string (or None if explicitly passed as None)
        self.llama_model_path = str(resolved_path) if resolved_path is not None else None


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a VisualLlamaConfig from a pretrained model path or HF repo id.
        This wraps PretrainedConfig.get_config_dict to ensure visual params
        and llama_model_path are preserved and returns an instance of cls.
        """
        if not pretrained_model_name_or_path:
            raise ValueError("Model path must be provided and cannot be None.")
        
        print(f"Loading model config from: {pretrained_model_name_or_path!r}")
    
        # get_config_dict will return either (config_dict, kwargs) or raise
        # it is defined in PretrainedConfig (parent of LlamaConfig).
        if os.path.exists(pretrained_model_name_or_path):
            # Local path: load config dict and set llama_model_path to the local path
            config_dict = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
            # get_config_dict sometimes returns (config_dict, kwargs) depending on transformers version
            if isinstance(config_dict, tuple):
                config_dict, kwargs = config_dict
            config_dict['llama_model_path'] = str(pretrained_model_name_or_path)
        else:
            # Hub or repo id: get config dict (and updated kwargs)
            config_result = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
            if isinstance(config_result, tuple):
                config_dict, kwargs = config_result
            else:
                config_dict = config_result
            # Record the repo id / name as llama_model_path so downstream code can see it
            config_dict['llama_model_path'] = str(pretrained_model_name_or_path)
    
        # Merge visual parameters if present in config_dict (keeps existing ones)
        visual_params = {}
        if "visual_feature_size" in config_dict:
            visual_params["visual_feature_size"] = config_dict["visual_feature_size"]
        if "num_visual_features" in config_dict:
            visual_params["num_visual_features"] = config_dict["num_visual_features"]
        if "tau" in config_dict:
            visual_params["tau"] = config_dict["tau"]
        
        config_dict.update(visual_params)
    
        # Debug log of the loaded config dict
        print(f"Config dict loaded (keys): {list(config_dict.keys())}")
        print(f" -> llama_model_path in config: {config_dict.get('llama_model_path')}")

        # Return an instance of this class (not the parent class)
        return cls.from_dict(config_dict, **kwargs)
