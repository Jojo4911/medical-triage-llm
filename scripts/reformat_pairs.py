import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
MODEL = "mistral-small-latest"

SYSTEM_PROMPT = """Tu es un assistant qui structure des données médicales brutes
selon un schéma fixe, à partir d'un cas clinique brut.

Retourne UNIQUEMENT un JSON valide avec ces champs :
- symptoms (string) : symptômes identifiés
- medical_history (string) : antécédents médicaux, vide si absent
- vital_signs (string) : constantes vitales, vide si absentes
- question (string) : la question posée
- correct_answer (string) : la réponse correcte
- medical_subject (string) : spécialité médicale concernée
- emergency_level (string) : un parmi "urgence maximale", "urgence modérée", "urgence différée"

Pas de texte hors du JSON. Pas de markdown, pas de balises code."""

def build_user_prompt(raw_entry: dict) -> str:
    return f"Cas brut :\n{raw_entry}"

import json

def reformat_entry(raw_entry: dict) -> dict | None:
    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(raw_entry)},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"JSON invalide pour l'entrée : {raw_entry}")
        return None
    
from datasets import load_dataset

# Exemple sur MediQA, à adapter aux 3 autres sources
mediqa = load_dataset("ANR-MALADES/MediQAl", "mcqu", split="train")
sample = mediqa.select(range(20))  # échantillon de prototype, PAS les 5000

results = [reformat_entry(dict(row)) for row in sample]
success_rate = sum(r is not None for r in results) / len(results)
print(f"Taux de succès parsing JSON : {success_rate:.0%}")

for r in results[:3]:
    print(r)
    print("---")