"""
Splits train/val/test pour les datasets SFT et DPO anonymisés,
plus un jeu d'évaluation clinique sépare et jamais vu a l'entrainement.
Ratios : 80% train, 10% val, 10% test. Le jeu d'eval clinique est prelevé
depuis le SFT (échantillon représentatif par source et niveau d'urgence).
"""

import json
import random
from pathlib import Path

random.seed(42)  # reproductibilité

SFT_INPUT = Path("data/processed/sft_pairs_anonymized.jsonl")
DPO_INPUT = Path("data/processed/dpo_pairs_anonymized.jsonl")

SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1 (le reste)

CLINICAL_EVAL_SIZE = 150  # échantillon fixe, prelevé avant le split train/val/test


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stratified_clinical_sample(records: list[dict], size: int) -> tuple[list[dict], list[dict]]:
    """Prelève un échantillon pour le jeu d'eval clinique, stratifie par source
    et niveau d'urgence pour rester représentatif. Retourne (eval_set, reste)."""
    from collections import defaultdict

    groups = defaultdict(list)
    for record in records:
        key = (record.get("source", "unknown"), record.get("emergency_level", "unknown"))
        groups[key].append(record)

    eval_set = []
    per_group_target = max(1, size // len(groups))

    for key, group_records in groups.items():
        random.shuffle(group_records)
        take = min(per_group_target, len(group_records))
        eval_set.extend(group_records[:take])

    # Complete si l'échantillonnage stratifié n'a pas atteint la taille cible
    eval_ids = {r["id"] for r in eval_set}
    if len(eval_set) < size:
        remaining = [r for r in records if r["id"] not in eval_ids]
        random.shuffle(remaining)
        eval_set.extend(remaining[: size - len(eval_set)])
        eval_ids = {r["id"] for r in eval_set}

    rest = [r for r in records if r["id"] not in eval_ids]
    return eval_set, rest


def split_train_val_test(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def process_sft():
    records = load_jsonl(SFT_INPUT)
    print(f"SFT : {len(records)} lignes chargees")

    clinical_eval, rest = stratified_clinical_sample(records, CLINICAL_EVAL_SIZE)
    print(f"SFT : jeu d'eval clinique preleve, {len(clinical_eval)} lignes, {len(rest)} restantes")

    train, val, test = split_train_val_test(rest)
    print(f"SFT : train={len(train)}, val={len(val)}, test={len(test)}")

    write_jsonl(clinical_eval, SPLITS_DIR / "sft_clinical_eval.jsonl")
    write_jsonl(train, SPLITS_DIR / "sft_train.jsonl")
    write_jsonl(val, SPLITS_DIR / "sft_val.jsonl")
    write_jsonl(test, SPLITS_DIR / "sft_test.jsonl")


def process_dpo():
    records = load_jsonl(DPO_INPUT)
    print(f"DPO : {len(records)} lignes chargees")

    train, val, test = split_train_val_test(records)
    print(f"DPO : train={len(train)}, val={len(val)}, test={len(test)}")

    write_jsonl(train, SPLITS_DIR / "dpo_train.jsonl")
    write_jsonl(val, SPLITS_DIR / "dpo_val.jsonl")
    write_jsonl(test, SPLITS_DIR / "dpo_test.jsonl")


def main():
    process_sft()
    process_dpo()
    print(f"\nTous les splits ecrits dans {SPLITS_DIR}/")


if __name__ == "__main__":
    main()