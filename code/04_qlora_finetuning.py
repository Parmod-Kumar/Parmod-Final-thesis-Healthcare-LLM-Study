"""
04_qlora_finetuning.py - QLoRA Fine‑Tuning
Self‑contained version – runs with synthetic data.

This script applies QLoRA-style quantised low-rank adaptation to a compact
language model. It is designed to approximate resource-efficient fine-tuning
patterns for medical question answering with lower memory requirements.
"""

import os
import time
import random
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Set stable random seeds for reproducible QLoRA training runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def save_results(results, filename, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"Results saved to {os.path.join(output_dir, filename)}")

def load_healthcare_dataset(dataset_name="pubmedqa"):
    """Generate a small synthetic dataset for demonstration."""
    # Replace with your real CSV loader if needed.
    synthetic = {
        "train": [
            {"question": "Is aspirin effective for headache?", "answer": "yes"},
            {"question": "Does paracetamol cause liver damage?", "answer": "maybe"},
            {"question": "Can antibiotics treat viral infections?", "answer": "no"},
            {"question": "Is exercise beneficial for heart health?", "answer": "yes"},
            {"question": "Does smoking increase lung cancer risk?", "answer": "yes"},
            {"question": "Can meditation reduce blood pressure?", "answer": "maybe"},
            {"question": "Is surgery necessary for all hernias?", "answer": "no"},
            {"question": "Does vitamin C prevent colds?", "answer": "maybe"},
            {"question": "Is coffee harmful during pregnancy?", "answer": "maybe"},
            {"question": "Can stress cause ulcers?", "answer": "yes"},
        ],
        "validation": [
            {"question": "Does sugar cause hyperactivity in children?", "answer": "no"},
            {"question": "Is MRI safe during pregnancy?", "answer": "maybe"},
        ],
        "test": [
            {"question": "Can probiotics treat IBS?", "answer": "maybe"},
            {"question": "Is sunscreen effective against UV rays?", "answer": "yes"},
            {"question": "Does high cholesterol always lead to heart disease?", "answer": "no"},
            {"question": "Can acupuncture relieve back pain?", "answer": "yes"},
            {"question": "Is gluten harmful for everyone?", "answer": "no"},
            {"question": "Can lack of sleep cause weight gain?", "answer": "yes"},
            {"question": "Does drinking water help with kidney stones?", "answer": "yes"},
            {"question": "Is yoga good for flexibility?", "answer": "yes"},
        ]
    }
    return synthetic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    # Model
    base_models = ["distilgpt2"]          # small model for quick testing
    # QLoRA / LoRA
    use_4bit = torch.cuda.is_available()  # only use 4-bit if GPU available
    bnb_4bit_quant_type = "nf4"
    lora_r = 4
    lora_alpha = 8
    lora_dropout = 0.1
    # Target modules for GPT‑2; adjust for other architectures
    lora_target_modules = ["c_attn", "c_proj"]
    # Training
    num_epochs = 2
    batch_size = 2
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    max_tokens = 50
    temperature = 0.7
    top_p = 0.9
    # Paths
    output_dir = "./results"
    model_save_dir = "./models"
    dataset_name = "pubmedqa"

config = Config()

# -------------------- QLoRATrainer Class --------------------
class QLoRATrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.peft_model = None

    def load_model_4bit(self):
        """Load the model with 4‑bit quantization (if GPU available)."""
        print(f"Loading model: {self.model_name}")
        print(f"Using 4-bit: {config.use_4bit}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if config.use_4bit and torch.cuda.is_available():
            # 4‑bit configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            # Prepare for k‑bit training
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # Fallback to standard FP32 model (no quantization)
            print("⚠️ 4-bit quantization disabled (no GPU or config). Using full precision.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            ).to(self.device)

        print("Model loaded successfully!")

    def setup_qlora(self):
        """Apply LoRA adapters (works with or without 4‑bit)."""
        print("Setting up QLoRA/LoRA adapters...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        return self.peft_model

    def prepare_dataset(self, data: list) -> Dataset:
        formatted_data = []
        for item in data:
            prompt = f"Question: {item['question']}\nAnswer:"
            response = item["answer"]
            formatted_data.append({"text": prompt + " " + response})

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128  # shorter for speed
            )

        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train(self, train_data: list, val_data: list):
        print("Preparing datasets...")
        train_dataset = self.prepare_dataset(train_data)
        val_dataset = self.prepare_dataset(val_data) if val_data else None

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_kwargs = dict(
            output_dir=f"{config.model_save_dir}/qlora_{self.model_name.split('/')[-1]}",
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=10,
            logging_steps=5,
            eval_steps=10,
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames:
            training_kwargs["eval_strategy"] = "steps" if val_dataset else "no"
        else:
            training_kwargs["evaluation_strategy"] = "steps" if val_dataset else "no"

        training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        print("Starting QLoRA training...")
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.2f} minutes")

        # Save adapters
        self.peft_model.save_pretrained(
            f"{config.model_save_dir}/qlora_{self.model_name.split('/')[-1]}"
        )
        self.tokenizer.save_pretrained(
            f"{config.model_save_dir}/qlora_{self.model_name.split('/')[-1]}"
        )
        return training_time

    def generate_response(self, question: str) -> tuple:
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        inference_time = time.time() - start_time

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        return response, inference_time

    def evaluate(self, test_data: list) -> list:
        print(f"Evaluating on {len(test_data)} test samples...")
        results = []
        for i, item in enumerate(test_data):
            prediction, inf_time = self.generate_response(item["question"])
            results.append({
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": prediction,
                "inference_time": inf_time
            })
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(test_data)}")
        return results

# -------------------- Main --------------------
def main():
    set_seed(42)
    print("=" * 60)
    print("QLORA FINE-TUNING EXPERIMENTS")
    print("=" * 60)

    # Load synthetic data
    data = load_healthcare_dataset()
    train_data = data.get("train", [])
    val_data = data.get("validation", [])
    test_data = data.get("test", [])
    if not test_data:
        test_data = val_data

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    all_results = []

    for model_name in config.base_models:
        print(f"\n{'='*50}\nModel: {model_name}\n{'='*50}")
        trainer = QLoRATrainer(model_name)

        # Check if bitsandbytes is available when using 4‑bit
        if config.use_4bit:
            try:
                import bitsandbytes
            except ImportError:
                print("⚠️ bitsandbytes not installed. Disabling 4‑bit quantization.")
                config.use_4bit = False

        trainer.load_model_4bit()
        trainer.setup_qlora()

        training_time = trainer.train(train_data, val_data)

        results = trainer.evaluate(test_data)

        for r in results:
            r["model"] = model_name
            r["method"] = "qlora"
            r["training_time"] = training_time

        all_results.extend(results)

        short_name = model_name.split("/")[-1]
        save_results(results, f"qlora_results_{short_name}.csv", config.output_dir)

    save_results(all_results, "qlora_results_combined.csv", config.output_dir)
    print("\nQLoRA fine-tuning experiments complete!")

if __name__ == "__main__":
    main()