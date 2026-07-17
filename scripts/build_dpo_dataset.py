"""
Construit le jeu de preferences DPO a partir de UltraMedical-Preference.
Filtre sur label_type == "hard", limite a un volume cible, format conversationnel
compatible TRL DPOTrainer (role/content partout).
"""

import json
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("data/processed/dpo_pairs.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET_COUNT = 5000
SYSTEM_PROMPT = "You are a helpful, honest medical triage assistant."


def get_assistant_content(turns: list[dict]) -> str:
    """Recupere le contenu du tour assistant par role, pas par position fixe."""
    for turn in turns:
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""


def build_record(idx: int, row: dict) -> dict:
    return {
        "id": f"dpo_{idx:05d}",
        "source": "UltraMedical-Preference",
        "language": "EN",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ],
        "chosen": [{"role": "assistant", "content": get_assistant_content(row["chosen"])}],
        "rejected": [{"role": "assistant", "content": get_assistant_content(row["rejected"])}],
    }


def main():
    dataset = load_dataset("TsinghuaC3I/UltraMedical-Preference", split="train")

    hard_dataset = dataset.filter(lambda row: row["label_type"] == "hard")
    available = len(hard_dataset)
    print(f"Lignes label_type=hard disponibles : {available}")

    if available < TARGET_COUNT:
        print(f"Attention : seulement {available} lignes hard, en dessous de la cible {TARGET_COUNT}.")

    take = min(TARGET_COUNT, available)
    subset = hard_dataset.select(range(take))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for idx, row in enumerate(subset):
            record = build_record(idx, row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{take} paires DPO ecrites dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()