import json
from pathlib import Path
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

SOURCE_PATH = Path("data/processed/dpo_pairs_anonymized.jsonl")
MODEL_PATH = "models/sft-lora-qwen3-1.7b"
OUTPUT_PATH = "models/dpo-lora-qwen3-1.7b"


def get_content(turns: list[dict]) -> str:
    for turn in turns:
        return turn.get("content", "")
    return ""


def preprocess_row(row: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": get_content(row["prompt"])}],
        "chosen": [{"role": "assistant", "content": get_content(row["chosen"])}],
        "rejected": [{"role": "assistant", "content": get_content(row["rejected"])}],
    }


def load_dataset(path: Path) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            rows.append(preprocess_row(raw))
    return Dataset.from_list(rows)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    dataset = load_dataset(SOURCE_PATH)

    # Adaptateur SFT deja entrainé, charge directement, pas de peft_config
    # ê repasser au DPOTrainer, sinon double wrapping de l'adaptateur.
    model = AutoPeftModelForCausalLM.from_pretrained(MODEL_PATH, is_trainable=True)

    config = DPOConfig(
        output_dir=OUTPUT_PATH,
        beta=0.1,                       # valeur par defaut TRL, pas de tuning marathon en POC
        learning_rate=5e-6,             # nettement plus bas qu'en SFT
        num_train_epochs=1,             # 1 a 3 pour DPO, on part sur 1 pour le run pilote
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",              # cohérent avec le tracking déjà utilisé en SFT
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_PATH)


if __name__ == "__main__":
    main()