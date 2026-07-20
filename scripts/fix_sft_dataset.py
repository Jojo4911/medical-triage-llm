import json
import unicodedata
from pathlib import Path

DATASET_PATH = Path("data/processed/sft_pairs.jsonl")


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def flatten_to_text(value) -> str:
    """Convertit une valeur de type quelconque (dict, liste, str, int...)
    en texte lisible, en préservant le sens clinique. Récursif pour gérer
    les structures imbriquées (ex: diagnostic + explications)."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(flatten_to_text(item) for item in value)
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            readable_key = key.replace("_", " ").capitalize()
            parts.append(f"{readable_key} : {flatten_to_text(val)}")
        return ". ".join(parts)
    return str(value)


def fix_record(record: dict) -> dict:
    # Normalise emergency_level sans accents
    record["emergency_level"] = strip_accents(record.get("emergency_level", ""))

    # Normalise symptoms en string, quel que soit le type source
    symptoms = record.get("symptoms", "")
    if isinstance(symptoms, list):
        record["symptoms"] = ", ".join(str(s) for s in symptoms)

    # Normalise correct_answer en string, quelle que soit la structure source
    # (dict avec description/effets_indesirables/diagnostic+explications/
    # options, ou liste).
    correct_answer = record.get("correct_answer", "")
    if not isinstance(correct_answer, str):
        record["correct_answer"] = flatten_to_text(correct_answer)

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

    print(f"{len(fixed)} lignes corrigées et réecrites.")


if __name__ == "__main__":
    main()