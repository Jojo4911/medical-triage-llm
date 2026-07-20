"""
Vérification rapide du dataset SFT généré.
Controle : nombre de lignes, répartition par source, cohérence language/medical_subject,
champs vides, échantillon pour inspection manuelle.
"""

import json
from collections import Counter
from pathlib import Path

DATASET_PATH = Path("data/processed/sft_pairs.jsonl")

EXPECTED_FIELDS = [
    "id", "source", "language", "question", "symptoms", "medical_history",
    "vital_signs", "correct_answer", "medical_subject", "emergency_level",
    "hospital_department",
]


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Ligne {line_num} invalide : {e}")
    return records


def check_fields(records: list[dict]) -> None:
    missing_field_counts = Counter()
    for record in records:
        for field in EXPECTED_FIELDS:
            if field not in record or record[field] in ("", None):
                missing_field_counts[field] += 1

    print("\n--- Champs manquants ou vides ---")
    if not missing_field_counts:
        print("Aucun champ manquant.")
    else:
        for field, count in missing_field_counts.most_common():
            pct = 100 * count / len(records)
            print(f"{field} : {count} lignes ({pct:.1f}%)")


def check_language_subject_consistency(records: list[dict]) -> None:
    """Vérifie que medical_subject reste dans la langue attendue."""
    fr_markers = ["cardiologie", "neurologie", "pneumologie", "gastro", "traumatologie",
                  "maladies", "endocrinologie", "autre"]
    en_markers = ["cardiology", "neurology", "pulmonology", "gastroenterology", "trauma",
                  "infectious", "endocrinology", "other"]

    suspects = []
    for record in records:
        subject = record.get("medical_subject", "").lower()
        language = record.get("language", "")
        if not subject:
            continue
        looks_fr = any(m in subject for m in fr_markers)
        looks_en = any(m in subject for m in en_markers)
        if language == "FR" and looks_en and not looks_fr:
            suspects.append(record)
        elif language == "EN" and looks_fr and not looks_en:
            suspects.append(record)

    print(f"\n--- Coherence language / medical_subject ---")
    print(f"Suspects de mismatch : {len(suspects)} / {len(records)}")
    for record in suspects[:5]:
        print(f"  id={record['id']} language={record['language']} medical_subject={record['medical_subject']!r}")


def check_distribution(records: list[dict]) -> None:
    sources = Counter(r.get("source", "unknown") for r in records)
    languages = Counter(r.get("language", "unknown") for r in records)
    emergency_levels = Counter(r.get("emergency_level", "unknown") for r in records)

    print("\n--- Répartition par source ---")
    for source, count in sources.most_common():
        print(f"{source} : {count}")

    print("\n--- Répartition par langue ---")
    for lang, count in languages.most_common():
        print(f"{lang} : {count}")

    print("\n--- Répartition par niveau d'urgence ---")
    for level, count in emergency_levels.most_common():
        print(f"{level} : {count}")


def check_duplicates(records: list[dict]) -> None:
    ids = [r.get("id") for r in records]
    id_counts = Counter(ids)
    duplicates = {id_: count for id_, count in id_counts.items() if count > 1}

    print(f"\n--- Doublons d'id ---")
    print(f"{len(duplicates)} id(s) dupliqués" if duplicates else "Aucun doublon d'id.")


def print_sample(records: list[dict], n: int = 3) -> None:
    print(f"\n--- Echantillon ({n} lignes) ---")
    for record in records[:n]:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print()


def main():
    if not DATASET_PATH.exists():
        print(f"Fichier introuvable : {DATASET_PATH}")
        return

    records = load_records(DATASET_PATH)
    print(f"Total lignes valides : {len(records)}")

    check_distribution(records)
    check_fields(records)
    check_language_subject_consistency(records)
    check_duplicates(records)
    print_sample(records)


if __name__ == "__main__":
    main()