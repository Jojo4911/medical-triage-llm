"""
Génère le dataset SFT (~5000 paires) à partir de MediQA (mcqu), FrenchMedMCQA et MedQuAD.
Reformate chaque exemple brut en paire instruction/réponse via Mistral Small, en
respectant le schéma de métadonnées (notes/metadata-schema.md) et en forçant la cohérence language/medical_subject.
"""

import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from mistralai.client import Mistral
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import unicodedata

load_dotenv()

CLIENT = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
MODEL = "mistral-small-latest"

_rate_lock = threading.Lock()
_last_call_time = [0.0]
MIN_INTERVAL_S = 1.2  # a ajuster : monter a 1.5-2 si des 429 persistent, baisser a 0.8 si aucun 429 n'apparait

OUTPUT_PATH = Path("data/processed/sft_pairs.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Cible de volume par source, ajustable. Total visé : ~5000.
SOURCE_TARGETS = {
    "mediqa": 1800,         # FR, config mcqu
    "frenchmedmcqa": 1500,  # FR
    "medquad": 1700,        # EN
}

EMERGENCY_LEVELS = ["urgence maximale", "urgence moderee", "urgence differee"]

# Liste fermée pour medical_subject, a completer selon les valeurs observees
MEDICAL_SUBJECTS_FR = [
    "Maladies infectieuses", "Cardiologie", "Endocrinologie et metabolisme",
    "Neurologie", "Pneumologie", "Gastro-enterologie", "Traumatologie", "Autre",
]
MEDICAL_SUBJECTS_EN = [
    "Infectious Diseases", "Cardiology", "Endocrinology and Metabolism",
    "Neurology", "Pulmonology", "Gastroenterology", "Trauma", "Other",
]

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

def load_sources():
    """Charge les 3 sources brutes. Ajuste les noms de colonnes selon
    notes/data-sources-exploration.md si besoin."""
    mediqa = load_dataset("ANR-MALADES/MediQAl", "mcqu", split="train")
    frenchmedmcqa = load_dataset(
        "nthngdy/frenchmedmcqa", split="train", revision="refs/convert/parquet"
    )
    medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    return {
        "mediqa": mediqa,
        "frenchmedmcqa": frenchmedmcqa,
        "medquad": medquad,
    }


def build_prompt(source: str, language: str, raw_question: str, raw_answer: str) -> str:
    subjects = MEDICAL_SUBJECTS_FR if language == "FR" else MEDICAL_SUBJECTS_EN
    lang_label = "francais" if language == "FR" else "anglais"
    return f"""Tu reformates une paire question/reponse medicale brute en paire
d'entrainement pour un agent de triage aux urgences.

Contrainte stricte : toute la sortie doit etre redigee en {lang_label}, y compris
le champ medical_subject qui doit venir EXACTEMENT de cette liste fermee :
{subjects}

Question brute : {raw_question}
Reponse brute : {raw_answer}

Reponds uniquement avec un objet JSON valide, sans texte autour, avec ces champs :
{{
  "question": "reformulation de la question comme si un patient decrivait ses symptomes",
  "symptoms": "liste des symptomes extraits ou deduits",
  "medical_history": "antecedents pertinents si mentionnes, sinon chaine vide",
  "vital_signs": "constantes vitales si mentionnees, sinon chaine vide",
  "correct_answer": "reponse medicale claire et structuree",
  "medical_subject": "une valeur EXACTE de la liste fournie",
  "emergency_level": "une valeur parmi {EMERGENCY_LEVELS}"
}}"""


def derive_hospital_department(medical_subject: str, emergency_level: str) -> str:
    """Derivation deterministe, PAS d'appel LLM. A verifier contre
    notes/metadata-schema.md (hypotheses documentees)."""
    mapping = {
        "Cardiologie": "Cardiologie", "Cardiology": "Cardiology",
        "Neurologie": "Neurologie", "Neurology": "Neurology",
        "Pneumologie": "Pneumologie", "Pulmonology": "Pulmonology",
        "Traumatologie": "Traumatologie", "Trauma": "Trauma",
    }
    if emergency_level == "urgence maximale":
        return "Urgences"
    return mapping.get(medical_subject, "Medecine generale")


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize_symptoms(symptoms) -> str:
    """Force symptoms en string, quel que soit le type renvoye par le LLM
    (parfois liste, parfois string, malgre la consigne du prompt)."""
    if isinstance(symptoms, list):
        return ", ".join(str(s) for s in symptoms)
    return symptoms or ""


def call_mistral(prompt: str, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        with _rate_lock:
            elapsed = time.time() - _last_call_time[0]
            if elapsed < MIN_INTERVAL_S:
                time.sleep(MIN_INTERVAL_S - elapsed)
            _last_call_time[0] = time.time()

        try:
            response = CLIENT.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            wait = 2 ** attempt
            print(f"Erreur ({e}), retry dans {wait}s")
            time.sleep(wait)
    return None

def normalize_correct_answers(correct_answers) -> list[str]:
    """Normalise correct_answers vers une liste de lettres a..e, quel que soit
    le type source (liste, int isole, string isolee)."""
    if correct_answers is None:
        return []
    if isinstance(correct_answers, (list, tuple)):
        raw_list = correct_answers
    else:
        raw_list = [correct_answers]

    index_to_letter = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
    letters = []
    for item in raw_list:
        if isinstance(item, str):
            letters.append(item.lower())
        elif isinstance(item, int):
            letters.append(index_to_letter.get(item, ""))
    return [l for l in letters if l]


def extract_qa(source_name: str, row: dict) -> tuple[str, str]:
    if source_name in ("mediqa", "frenchmedmcqa"):
        question = row.get("question", "")
        options = {
            letter: row.get(f"answer_{letter}", "")
            for letter in ["a", "b", "c", "d", "e"]
        }
        correct_letters = normalize_correct_answers(row.get("correct_answers"))
        correct_texts = [
            options[letter] for letter in correct_letters if letter in options
        ]
        answer = " / ".join(correct_texts) if correct_texts else ""
        if source_name == "mediqa" and row.get("clinical_case"):
            question = f"{row['clinical_case']}\n{question}"
        return question, answer

    if source_name == "medquad":
        return row.get("Question", ""), row.get("Answer", "")

    raise ValueError(f"Source inconnue : {source_name}")


def process_one(source_name: str, language: str, idx: int, row: dict) -> dict | None:
    """Traite une ligne, appelee en parallele."""
    raw_question, raw_answer = extract_qa(source_name, row)
    prompt = build_prompt(source_name, language, raw_question, raw_answer)
    parsed = call_mistral(prompt)
    if parsed is None:
        return None

    record = {
        "id": f"{source_name}_{idx:05d}",
        "source": source_name,
        "language": language,
        "question": parsed.get("question", ""),
        "symptoms": normalize_symptoms(parsed.get("symptoms", "")),
        "medical_history": parsed.get("medical_history", ""),
        "vital_signs": parsed.get("vital_signs", ""),
        "correct_answer": parsed.get("correct_answer", ""),
        "medical_subject": parsed.get("medical_subject", ""),
        "emergency_level": strip_accents(parsed.get("emergency_level", "")),
    }
    record["hospital_department"] = derive_hospital_department(
        record["medical_subject"], record["emergency_level"]
    )
    return record


def process_source(source_name: str, dataset, language: str, target: int, start_idx: int, max_workers: int = 3):
    results = []
    subset = dataset.select(range(min(target, len(dataset))))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, source_name, language, start_idx + i, row): i
            for i, row in enumerate(subset)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=source_name):
            record = future.result()
            if record is not None:
                results.append(record)

    return results


def main():
    sources = load_sources()
    language_map = {"mediqa": "FR", "frenchmedmcqa": "FR", "medquad": "EN"}

    # MediQA deja terminee et ecrite (1800 paires). On ne traite que le reste.
    remaining_sources = ["frenchmedmcqa", "medquad"]

    all_records = []
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:  # "a" : on ajoute, pas "w"
        for source_name in remaining_sources:
            dataset = sources[source_name]
            language = language_map[source_name]
            target = SOURCE_TARGETS[source_name]
            records = process_source(
                source_name, dataset, language, target, start_idx=len(all_records)
            )
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.extend(records)
            print(f"{source_name} : {len(records)} paires generees")

    print(f"Total ajoute cette session : {len(all_records)} paires")


if __name__ == "__main__":
    main()