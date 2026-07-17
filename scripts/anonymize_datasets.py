"""
Applique l'anonymisation Presidio (scripts/anonymize_dataset.py) aux datasets
SFT et DPO générés. Anonymise les champs texte libre susceptibles
de contenir des noms de patients, log le nombre d'anonymisations par fichier.
"""

import json
from pathlib import Path

from anonymize_dataset import build_analyzer, anonymize_text
from presidio_anonymizer import AnonymizerEngine

SFT_INPUT = Path("data/processed/sft_pairs.jsonl")
SFT_OUTPUT = Path("data/processed/sft_pairs_anonymized.jsonl")
DPO_INPUT = Path("data/processed/dpo_pairs.jsonl")
DPO_OUTPUT = Path("data/processed/dpo_pairs_anonymized.jsonl")

# Champs texte libre a passer dans Presidio, par dataset
SFT_TEXT_FIELDS = ["question", "symptoms", "medical_history", "correct_answer"]
DPO_TEXT_FIELDS = ["prompt", "chosen", "rejected"]  # structure imbriquee role/content


def count_person_entities(text: str, analyzer, language: str) -> int:
    """Recompte les entites PERSON reellement anonymisees (apres filtrage
    des faux positifs), pour le log. Reutilise la meme logique que anonymize_text."""
    from anonymize_dataset import filter_medical_false_positives, ALLOW_LIST

    results = analyzer.analyze(text=text, language=language, allow_list=ALLOW_LIST)
    results = filter_medical_false_positives(text, results, language)
    return sum(1 for r in results if r.entity_type == "PERSON")


def ensure_string(value) -> str:
    """Force une valeur en string pour Presidio, quel que soit son type source.
    Gere le cas dict {description, justification} observé sur certaines réponses
    correct_answer mal structurées par le LLM de reformatage, en plus du
    filet de securité générique pour tout autre type inattendu."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [str(v) for v in value.values() if v]
        return " ".join(parts)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def anonymize_sft_record(record: dict, analyzer, anonymizer) -> tuple[dict, int]:
    language = "fr" if record.get("language") == "FR" else "en"
    total_anonymized = 0

    for field in SFT_TEXT_FIELDS:
        value = ensure_string(record.get(field, ""))
        if not value:
            continue
        total_anonymized += count_person_entities(value, analyzer, language)
        record[field] = anonymize_text(value, analyzer, anonymizer, language=language)

    return record, total_anonymized


def anonymize_dpo_record(record: dict, analyzer, anonymizer) -> tuple[dict, int]:
    language = "fr" if record.get("language") == "FR" else "en"
    total_anonymized = 0

    for field in DPO_TEXT_FIELDS:
        turns = record.get(field, [])
        for turn in turns:
            content = ensure_string(turn.get("content", ""))
            if not content:
                continue
            total_anonymized += count_person_entities(content, analyzer, language)
            turn["content"] = anonymize_text(content, analyzer, anonymizer, language=language)

    return record, total_anonymized


def process_file(input_path: Path, output_path: Path, record_fn, analyzer, anonymizer) -> None:
    if not input_path.exists():
        print(f"Fichier introuvable, ignore : {input_path}")
        return

    total_records = 0
    total_anonymized_entities = 0
    records_with_anonymization = 0

    with open(input_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            record, count = record_fn(record, analyzer, anonymizer)

            total_records += 1
            total_anonymized_entities += count
            if count > 0:
                records_with_anonymization += 1

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n--- {input_path.name} ---")
    print(f"Lignes traitees : {total_records}")
    print(f"Entites PERSON anonymisees (total) : {total_anonymized_entities}")
    print(f"Lignes contenant au moins une anonymisation : {records_with_anonymization} ({100 * records_with_anonymization / total_records:.1f}%)")
    print(f"Fichier ecrit : {output_path}")


def main():
    analyzer = build_analyzer()
    anonymizer = AnonymizerEngine()

    process_file(SFT_INPUT, SFT_OUTPUT, anonymize_sft_record, analyzer, anonymizer)
    process_file(DPO_INPUT, DPO_OUTPUT, anonymize_dpo_record, analyzer, anonymizer)


if __name__ == "__main__":
    main()