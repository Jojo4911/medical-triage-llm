from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
ADAPTER_PATH = "models/dpo-lora-qwen3-1.7b"
OUTPUT_PATH = "models/merged-triage-model"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(OUTPUT_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Modèle fusionné sauvegardé dans {OUTPUT_PATH}")