"""
Diagnostic de fraicheur des corrections d'anonymisation sur le jeu DPO.
Verifie l'absence de regressions liees aux corrections 4 a 7 du pipeline.
Usage : uv run python scripts/checks/check_dpo_anonymization.py
"""

import json
import re
from collections import Counter
from pathlib import Path

SOURCE_PATH = Path("data/processed/dpo_pairs_anonymized.jsonl")

# Correction 7 : pattern groupes sanguins, ne doit jamais etre tagué
BLOOD_GROUP_PATTERN = re.compile(
    r"\b(blood\s+(type|group)\s*:?\s*)(A|B|AB|O)[+-]?\b"
    r"|\b(A|B|AB|O)[+-]\b"  # forme compacte avec signe, ex. "O-", "AB+"
    r"|\bRh\s*[+-]?\b",
    re.IGNORECASE,
)

LOCATION_TAG_PATTERN = re.compile(r"<LOCATION>")

# Correction 6 : eponymes frequents, ne doivent jamais etre anonymises
EPONYM_KEYWORDS = ["maladie de", "syndrome de", "syndrome d'", "disease", "syndrome", "'s disease", "'s syndrome", "' disease",
    "' syndrome", "sign", "test", "disorder", "palsy", "'s palsy",
    "thyroiditis", "'s thyroiditis", "aphasia", "'s aphasia",
    "chorea", "'s chorea"]

TAG_PATTERN = re.compile(r"<(PERSON|LOCATION|ORGANIZATION|[A-Z_]+)>")


def extract_text(field) -> str:
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        return " ".join(turn.get("content", "") for turn in field)
    return ""


def main():
    tag_counts = Counter()
    organization_hits = []
    eponym_near_tag = []
    blood_group_near_tag = []
    total = 0

    with open(SOURCE_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total += 1
            row = json.loads(line)
            full_text = " ".join(
                extract_text(row.get(k)) for k in ("prompt", "chosen", "rejected")
            )

            tags_found = TAG_PATTERN.findall(full_text)
            for tag in tags_found:
                tag_counts[tag] += 1
                if tag == "ORGANIZATION":
                    organization_hits.append(line_no)

            for kw in EPONYM_KEYWORDS:
                idx = full_text.lower().find(kw)
                if idx != -1:
                    window = full_text[idx:idx + 60]
                    if LOCATION_TAG_PATTERN.search(window):
                        eponym_near_tag.append((line_no, window))

            for match in BLOOD_GROUP_PATTERN.finditer(full_text):
                start = max(0, match.start() - 15)
                window = full_text[start:match.end() + 15]
                if "<" in window:
                    blood_group_near_tag.append((line_no, window))

    print(f"Total paires DPO : {total}")
    print(f"\nRepartition des tags rencontres : {dict(tag_counts)}")

    print(f"\n--- Correction 4 (ORGANIZATION doit etre absent) ---")
    if organization_hits:
        print(f"ALERTE : {len(organization_hits)} occurrence(s), lignes : {organization_hits[:10]}")
    else:
        print("OK, aucun tag ORGANIZATION residuel.")

    print(f"\n--- Correction 6 (eponymes en LOCATION mal filtres) ---")
    if eponym_near_tag:
        print(f"A VERIFIER MANUELLEMENT, {len(eponym_near_tag)} occurrence(s) :")
        for line_no, window in eponym_near_tag[:5]:
            print(f"  ligne {line_no} : ...{window}...")
    else:
        print("OK, aucun pattern eponyme proche d'un tag.")

    print(f"\n--- Correction 7 (groupes sanguins faussement tagues) ---")
    if blood_group_near_tag:
        print(f"A VERIFIER MANUELLEMENT, {len(blood_group_near_tag)} occurrence(s) :")
        for line_no, window in blood_group_near_tag[:5]:
            print(f"  ligne {line_no} : ...{window}...")
    else:
        print("OK, aucun groupe sanguin proche d'un tag.")


if __name__ == "__main__":
    main()