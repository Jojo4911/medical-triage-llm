"""
Pipeline SFT + LoRA pour l'agent de triage medical (P14).
Formalise depuis le notebook Colab pilote pour le run principal.

Format de prompt : structure (instruction / symptomes / reponse balisee),
choisi pour permettre un parsing fiable du niveau d'urgence et du service
cote API de triage (livrable 3), plutot qu'un texte libre non exploitable
automatiquement.

Usage:
    uv run python scripts/sft_lora_train.py
"""

import os
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# --- Config generale ---------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
TRAIN_PATH = "data/splits/sft_train.jsonl"
VAL_PATH = "data/splits/sft_val.jsonl"
OUTPUT_DIR = "models/sft-lora-qwen3-1.7b"
WANDB_PROJECT = "p14-triage-medical"
WANDB_RUN_NAME = "sft-lora-run-full-j6"          # à adapter si relance


# --- Formatage du prompt (structure) ------------------------------------

PROMPT_TEMPLATE = """### Instruction:
Tu es un agent de triage medical. Evalue le niveau d'urgence et oriente le patient.

### Symptomes rapportes:
{symptoms}

### Reponse:
Niveau d'urgence: {emergency_level}
Service: {hospital_department}
Explication: {correct_answer}"""


def format_example(example: dict) -> dict:
    """Assemble les champs du schema (symptoms, correct_answer,
    emergency_level, hospital_department) en un champ 'text' unique,
    format attendu par SFTTrainer. Le préfixe medical_history est ajouté
    aux symptomes s'il est présent et non vide, pour enrichir le contexte
    clinique sans multiplier les champs du template."""
    symptoms = example.get("symptoms", "") or ""
    medical_history = example.get("medical_history", "") or ""
    if medical_history:
        symptoms = f"{symptoms}. Antecedents: {medical_history}"

    text = PROMPT_TEMPLATE.format(
        symptoms=symptoms,
        emergency_level=example.get("emergency_level", "NON PRECISE"),
        hospital_department=example.get("hospital_department", "NON PRECISE"),
        correct_answer=example.get("correct_answer", ""),
    )
    return {"text": text}


# --- Config LoRA (validee au pilote T20) --------------------------------

def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


# --- Main ----------------------------------------------------------------

def main():
    # Rappel du fix torchao rencontré au pilote : si l'environnement
    # est frais (VM GCP neuve), vérifier la version avant de lancer :
    #   pip install -U torchao
    # puis redemarrer le kernel/process si l'import plante.

    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    print(f"Chargement du dataset depuis {TRAIN_PATH} / {VAL_PATH}")
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_PATH, "validation": VAL_PATH},
    )
    dataset = dataset.map(format_example)

    dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
    dataset["validation"] = dataset["validation"].select(range(min(20, len(dataset["validation"]))))

    print(f"Chargement du modele de base : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = build_lora_config()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,              # point de depart POC, pas un objectif à tuner
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,              # garder les 3 derniers checkpoints, évite de saturer le disque 100GB
        bf16=True,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
        dataset_text_field="text",
        max_length=1024,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Debut de l'entrainement SFT + LoRA")
    trainer.train()

    print(f"Sauvegarde du modele final dans {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    wandb.finish()
    print("Termine.")


if __name__ == "__main__":
    main()