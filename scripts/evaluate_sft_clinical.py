"""
Validation intermediaire du modele SFT+LoRA sur le jeu d'eval clinique.
Genère des réponses sur un échantillon pour inspection qualitative,
avant la comparaison complête base/SFT/DPO.
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_PATH = "models/sft-lora-qwen3-1.7b"
CLINICAL_EVAL_PATH = "data/splits/sft_clinical_eval.jsonl"
SAMPLE_SIZE = 15

# Meme template que sft_lora_train.py, mais coupe avant la reponse :
# c'est au modele de la génerer, pas à nous de la lui fournir.
PROMPT_PREFIX_TEMPLATE = """### Instruction:
Tu es un agent de triage medical. Evalue le niveau d'urgence et oriente le patient.

### Symptomes rapportes:
{symptoms}

### Reponse:
"""


def load_clinical_sample(path: str, size: int) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= size:
                break
    return records


def build_prompt(example: dict) -> str:
    """Meme fallback que format_example dans sft_lora_train.py : utilise
    question si symptoms est vide (sources type MedQuAD)."""
    symptoms = (example.get("symptoms", "") or "").strip()
    question = (example.get("question", "") or "").strip()
    medical_history = example.get("medical_history", "") or ""

    if not symptoms:
        symptoms = question

    if medical_history:
        symptoms = f"{symptoms}. Antecedents: {medical_history}"
    return PROMPT_PREFIX_TEMPLATE.format(symptoms=symptoms)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Chargement du tokenizer depuis {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    print("Chargement du modele de base")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B-Base",
        dtype=torch.bfloat16,
        device_map=device,
    )

    print(f"Chargement de l'adaptateur LoRA depuis {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    samples = load_clinical_sample(CLINICAL_EVAL_PATH, SAMPLE_SIZE)
    print(f"{len(samples)} exemples charges pour inspection\n")

    for i, example in enumerate(samples, start=1):
        prompt = build_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,  # deterministe, suffisant pour inspection POC
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_response = generated_text[len(prompt):].strip()

        print(f"--- Exemple {i} (id={example.get('id', '?')}) ---")
        print(f"Symptomes       : {example.get('symptoms', '')}")
        print(f"Reponse attendue: {example.get('correct_answer', '')[:200]}")
        print(f"Reponse generee : {generated_response[:400]}")
        print()


if __name__ == "__main__":
    main()