"""
Comparaison qualitative modele base vs SFT+LoRA vs DPO sur quelques vignettes cliniques.
Usage : uv run python scripts/compare_base_vs_sft.py
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
SFT_ADAPTER = "models/sft-lora-qwen3-1.7b"
DPO_ADAPTER = "models/dpo-lora-qwen3-1.7b"

# Vignettes tirées du jeu d'eval clinique, avec fallback question si symptoms vide
VIGNETTES = [
    {
        "symptoms": "Douleur thoracique brutale, essoufflement, sueurs froides depuis 20 minutes",
        "question": None,
    },
    {
        "symptoms": None,
        "question": "Fievre a 39,5 depuis 3 jours chez un enfant de 4 ans, refus de s'alimenter",
    },
    {
        "symptoms": "Cephalees legeres depuis ce matin, pas d'autre symptome",
        "question": None,
    },
    {
        "symptoms": "Faiblesse soudaine du bras droit et difficulté à parler depuis 15 minutes, léger vertige, pas de douleur",
        "question": None,
    },
    {
        "symptoms": "Douleur abdominale diffuse depuis 12 heures, désormais localisée en bas à droite, nausées, fièvre légère à 38",
        "question": None,
    },
]

PROMPT_TEMPLATE = """Instruction:

Tu es un agent de triage médical. Evalue le niveau d'urgence et oriente le patient.

Symptomes rapportés:

{symptoms}

Réponse:

"""


def build_prompt(vignette):
    text = vignette["symptoms"] or vignette["question"]
    return PROMPT_TEMPLATE.format(symptoms=text)


def generate(model, tokenizer, prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


def main():
    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Chargement du modele base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print("Chargement de l'adaptateur SFT...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER, adapter_name="sft")

    print("Chargement de l'adaptateur DPO...")
    model.load_adapter(DPO_ADAPTER, adapter_name="dpo")

    for i, vignette in enumerate(VIGNETTES, start=1):
        prompt = build_prompt(vignette)
        print(f"\n{'=' * 60}")
        print(f"VIGNETTE {i}")
        print(f"{'=' * 60}")
        print(f"Entree : {vignette['symptoms'] or vignette['question']}\n")

        print("--- BASE (sans adaptateur) ---")
        with model.disable_adapter():
            print(generate(model, tokenizer, prompt))

        print("\n--- SFT+LoRA ---")
        model.set_adapter("sft")
        print(generate(model, tokenizer, prompt))

        print("\n--- DPO ---")
        model.set_adapter("dpo")
        print(generate(model, tokenizer, prompt))


if __name__ == "__main__":
    main()
