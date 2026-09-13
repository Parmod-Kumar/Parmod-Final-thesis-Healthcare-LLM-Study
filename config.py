"""
config.py - Configuration settings for the study
Author: [Your Name]
Date: [Current Date]
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for base models and fine-tuning."""
    
    # Base Models
    base_models: List[str] = None
    
    # Prompt Engineering Configurations
    prompt_types: List[str] = None
    few_shot_examples: int = 3
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    
    # LoRA Configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # QLoRA Configuration
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Training Parameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    
    # Dataset
    dataset_name: str = "pubmedqa"
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Paths
    output_dir: str = "./results"
    model_save_dir: str = "./models"
    data_dir: str = "./data"
    
    def __post_init__(self):
        if self.base_models is None:
            self.base_models = [
                "mistralai/Mistral-7B-Instruct-v0.3",
                "google/gemma-7b-it",
                "microsoft/Phi-3-mini-4k-instruct"
            ]
        
        if self.prompt_types is None:
            self.prompt_types = [
                "zero_shot",
                "few_shot",
                "chain_of_thought",
                "role_based",
                "prompt_ensemble"
            ]
        
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]


# Instantiate default configuration
config = ModelConfig()