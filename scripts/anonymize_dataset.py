"""
Anonymisation RGPD des paires SFT/DPO via Presidio.
Detecte les entites PERSON, exclut les eponymes medicaux non ambigus
(allow_list) et les eponymes ambigus/non listes uniquement s'ils sont
precedes ou suivis d'un marqueur medical, anonymise le reste.
"""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import json
from pathlib import Path
import re

# Eponymes surs : jamais utilises comme patronymes de patients en pratique.
# Exemptes sans condition, geres nativement par Presidio via allow_list.
ALLOW_LIST_UNAMBIGUOUS = [
    "Parkinson", "Parkinson's", "Alzheimer", "Alzheimer's", "Basedow",
    "Crohn", "Crohn's", "Hodgkin", "Hodgkin's", "Sjogren", "Sjögren",
    "Guillain-Barre", "Guillain-Barré", "Klinefelter", "Marfan",
    "Ehlers-Danlos", "Kaposi", "Raynaud", "Tourette", "Asperger",
    "Creutzfeldt-Jakob", "Meniere", "Ménière", "Charcot", "Babinski",
    "Korsakoff", "Dupuytren", "Paget", "Trousseau", "Wernicke",
    "Wernicke's", "De Clerambault", "de Clerambault", "Clerambault",
]

# Pays exemptes de l'anonymisation LOCATION : information cliniquement
# pertinente (medecine des voyages, conseil vaccinal), et un pays seul
# n'identifie pas un individu, contrairement a une ville ou une adresse
# precise. Liste non exhaustive (POC), a completer si besoin en production.
COUNTRY_ALLOW_LIST = [
    "France", "Gabon", "Sénégal", "Senegal", "Côte d'Ivoire", "Cote d'Ivoire",
    "Mali", "Cameroun", "Cameroon", "Congo", "RDC", "Madagascar", "Maroc",
    "Morocco", "Tunisie", "Tunisia", "Algérie", "Algeria", "Bénin", "Benin",
    "Togo", "Burkina Faso", "Niger", "Tchad", "Chad", "Guinée", "Guinea",
    "Inde", "India", "Thaïlande", "Thailand", "Vietnam", "Cambodge", "Cambodia",
    "Brésil", "Brazil", "Pérou", "Peru", "Bolivie", "Bolivia", "Kenya",
    "Tanzanie", "Tanzania", "Éthiopie", "Ethiopia", "Nigeria", "Ghana",
    "Égypte", "Egypt", "Chine", "China", "Indonésie", "Indonesia",
    "Philippines", "Mexique", "Mexico", "Colombie", "Colombia",
    "Haïti", "Haiti", "États-Unis", "USA", "United States", "Canada",
    "Royaume-Uni", "United Kingdom", "Allemagne", "Germany", "Espagne", "Spain",
    "Italie", "Italy", "Portugal", "Belgique", "Belgium", "Suisse", "Switzerland",
]
_country_allow_list_lower = [c.lower() for c in COUNTRY_ALLOW_LIST]

# Pattern groupe sanguin : A, B, AB, ou O, seul ou suivi de "Rh".
# Capture aussi bien "A" isole que "A Rh" tel que Presidio le decoupe.
BLOOD_TYPE_PATTERN = re.compile(r"^(AB|A|B|O)(\s*Rh)?$", re.IGNORECASE)

def is_blood_type_pattern(text: str, start: int, end: int) -> bool:
    entity_text = text[start:end].strip()
    return bool(BLOOD_TYPE_PATTERN.match(entity_text))


def is_allowed_country(text: str, start: int, end: int) -> bool:
    entity_text = text[start:end].strip().lower()
    return entity_text in _country_allow_list_lower

# Eponymes ambigus : ce sont aussi des patronymes courants. Risque RGPD reel
# si on les exempte sans condition, un vrai patient portant ce nom ne serait
# jamais anonymise. Anonymises par defaut, exemptes seulement si le contexte
# medical (prefixe ou suffixe) est detecte juste a cote de l'entite.
AMBIGUOUS_EPONYMS = [
    "Wilson", "Down", "Turner", "Graves", "Addison", "Cushing",
    "Bowen", "Bell", "Sheehan", "Broca", "Huntington", "Barrett",
    "Hashimoto",
]

MEDICAL_PREFIXES = [
    "maladie de", "maladie d'", "syndrome de", "syndrome d'",
    "signe de", "test de", "trouble de",
]

# Elargi : formes avec apostrophe seule (Graves' disease) et pathologies
# specifiques manquantes (palsy, thyroiditis, aphasia, chorea) qui
# concernent plusieurs eponymes ambigus de la liste ci-dessus.
MEDICAL_SUFFIXES = [
    "disease", "syndrome", "'s disease", "'s syndrome", "' disease",
    "' syndrome", "sign", "test", "disorder", "palsy", "'s palsy",
    "thyroiditis", "'s thyroiditis", "aphasia", "'s aphasia",
    "chorea", "'s chorea",
]

CONTEXT_WINDOW_CHARS = 20


def is_medical_eponym_by_context(text: str, entity_start: int, entity_end: int, language: str) -> bool:
    """Verifie que l'entite est IMMEDIATEMENT precedee ou suivie d'un marqueur
    medical, pas seulement proche dans une fenetre de caracteres. Evite les faux
    negatifs quand un mot medical apparait plus loin dans la meme phrase sans
    rapport avec l'entite detectee."""
    window_before = text[max(0, entity_start - CONTEXT_WINDOW_CHARS):entity_start].lower().rstrip()
    window_after = text[entity_end:entity_end + CONTEXT_WINDOW_CHARS].lower().lstrip()

    has_prefix_match = any(window_before.endswith(prefix) for prefix in MEDICAL_PREFIXES)
    has_suffix_match = any(window_after.startswith(suffix) for suffix in MEDICAL_SUFFIXES)

    return has_prefix_match or has_suffix_match


def build_analyzer() -> AnalyzerEngine:
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "fr", "model_name": "fr_core_news_md"},
            {"lang_code": "en", "model_name": "en_core_web_lg"},
        ],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry(supported_languages=["en", "fr"])
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en", "fr"])

    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en", "fr"],
        registry=registry,
    )


# Types d'entites Presidio susceptibles de capturer un eponyme medical mal
# classé. PERSON est le cas attendu, ORGANIZATION est le cas qui casse
# "Bell's palsy" observe dans le diagnostic initial.
FILTERABLE_ENTITY_TYPES = {"PERSON", "LOCATION"}


def filter_medical_false_positives(text: str, results: list, language: str) -> list:
    """Retire les entites PERSON ou ORGANIZATION qui sont en realite des
    eponymes medicaux, detectes soit par allow_list non ambigue, soit par
    contexte precedent ou suivant."""
    filtered = []
    unambiguous_lower = [term.lower() for term in ALLOW_LIST_UNAMBIGUOUS]

    for result in results:
        if result.entity_type not in FILTERABLE_ENTITY_TYPES:
            filtered.append(result)
            continue

        entity_text = text[result.start:result.end].lower()
        if entity_text in unambiguous_lower:
            continue

        if is_medical_eponym_by_context(text, result.start, result.end, language):
            continue

        filtered.append(result)

    return filtered


MIN_ENTITY_LENGTH = 3  # en dessous, quasi certainement un faux positif (initiales, groupes sanguins, sigles courts)


def is_too_short_to_be_meaningful(text: str, start: int, end: int) -> bool:
    return (end - start) < MIN_ENTITY_LENGTH


def anonymize_text(text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine, language: str = "fr") -> str:
    results = analyzer.analyze(
        text=text,
        language=language,
        allow_list=ALLOW_LIST_UNAMBIGUOUS,
        score_threshold=0.5,
    )
    results = [r for r in results if r.entity_type != "ORGANIZATION"]
    results = [
        r for r in results
        if not (r.entity_type == "LOCATION" and is_allowed_country(text, r.start, r.end))
    ]
    # Ecarte les entites trop courtes (groupes sanguins, initiales, sigles),
    # toutes entites confondues, avant le filtre de contexte medical
    results = [r for r in results if not is_too_short_to_be_meaningful(text, r.start, r.end)]
    results = [r for r in results if not is_blood_type_pattern(text, r.start, r.end)]
    results = filter_medical_false_positives(text, results, language)

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"PERSON": OperatorConfig("replace", {"new_value": "[PATIENT]"})},
    )
    return anonymized.text


TEXT_FIELDS_TO_ANONYMIZE = ["question", "symptoms", "medical_history", "correct_answer"]


def anonymize_jsonl_file(
    input_path: str,
    output_path: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
) -> dict:
    """Parcourt un JSONL ligne par ligne, anonymise les champs texte selon
    la langue de chaque ligne, ecrit le resultat dans un nouveau fichier.
    Retourne un petit rapport de comptage pour verification."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"lines_processed": 0, "lines_with_patient_tag": 0, "lines_with_org_tag": 0}

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            language = record.get("language", "fr").lower()
            language = language if language in ("fr", "en") else "en"

            for field in TEXT_FIELDS_TO_ANONYMIZE:
                if field in record and record[field]:
                    record[field] = anonymize_text(record[field], analyzer, anonymizer, language=language)

            record_text = json.dumps(record, ensure_ascii=False)
            if "[PATIENT]" in record_text:
                stats["lines_with_patient_tag"] += 1
            if "<ORGANIZATION>" in record_text:
                stats["lines_with_org_tag"] += 1

            outfile.write(record_text + "\n")
            stats["lines_processed"] += 1

    return stats


if __name__ == "__main__":
    analyzer = build_analyzer()
    anonymizer = AnonymizerEngine()

    test_cases = [
        ("Monsieur Martin presente la maladie de Parkinson", "fr"),
        ("La patiente Sophie Dubois souffre du syndrome de Crohn", "fr"),
        ("Jean-Pierre a consulte pour des douleurs abdominales", "fr"),
        ("The patient John Smith was diagnosed with Parkinson's disease", "en"),
        ("Mary Johnson reported symptoms of Crohn's disease", "en"),
        # Cas ambigus, contexte medical present : doivent rester non anonymises
        ("The patient has Wilson's disease", "en"),
        ("Le bebe presente un syndrome de Down", "fr"),
        ("Diagnosed with Bell's palsy last week", "en"),
        # Cas ambigus, AUCUN contexte medical : doivent etre anonymises,
        # ce sont les cas que l'ancienne version laissait fuiter
        ("Le patient se nomme Wilson et vit a Lyon", "fr"),
        ("Mr. Down called to reschedule his appointment", "en"),
        ("Ms. Turner will arrive at 3pm for her consultation", "en"),
    ]

    for text, lang in test_cases:
        result = anonymize_text(text, analyzer, anonymizer, language=lang)
        print(f"Original  : {text}")
        print(f"Anonymise : {result}\n")

    # Traitement du fichier reel, une fois les 11 cas de test valides
    report = anonymize_jsonl_file(
        input_path="data/processed/sft_pairs.jsonl",          # a confirmer
        output_path="data/processed/sft_pairs_anonymized_v2.jsonl",
        analyzer=analyzer,
        anonymizer=anonymizer,
    )
    print(report)