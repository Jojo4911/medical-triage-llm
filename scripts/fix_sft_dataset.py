import json
import unicodedata
from pathlib import Path

DATASET_PATH = Path("data/processed/sft_pairs.jsonl")


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def fix_record(record: dict) -> dict:
    # Normalise emergency_level sans accents
    record["emergency_level"] = strip_accents(record.get("emergency_level", ""))

    # Normalise symptoms en string, quel que soit le type source
    symptoms = record.get("symptoms", "")
    if isinstance(symptoms, list):
        record["symptoms"] = ", ".join(str(s) for s in symptoms)

    return record


def main():
    records = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    fixed = [fix_record(r) for r in records]

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for record in fixed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(fixed)} lignes corrigees et reecrites.")


if __name__ == "__main__":
    main()