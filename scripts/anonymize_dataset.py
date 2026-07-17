"""
Anonymisation RGPD des paires SFT/DPO via Presidio.
Detecte les entites PERSON, exclut les eponymes medicaux connus (allow_list)
et les eponymes non listes mais precedes d'un marqueur medical (maladie de,
syndrome de...), anonymise le reste par remplacement generique.
"""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Eponymes medicaux connus a l'avance, jamais anonymises quel que soit le contexte
ALLOW_LIST = [
    "Parkinson", "Alzheimer", "Basedow", "Crohn", "Hodgkin",
    "Parkinson's", "Alzheimer's", "Crohn's", "Hodgkin's",
    "Addison", "Cushing", "Graves", "Hashimoto", "Sjogren", "Sjögren",
    "Guillain-Barre", "Guillain-Barré", "Wilson", "Huntington",
    "Down", "Turner", "Klinefelter", "Marfan", "Ehlers-Danlos",
    "Kaposi", "Raynaud", "Sheehan", "Wernicke", "Korsakoff",
    "Tourette", "Asperger", "Creutzfeldt-Jakob", "Bell",
    "Barrett", "Bowen", "Paget", "Dupuytren", "Meniere", "Ménière",
    "De Clerambault", "de Clerambault", "Clerambault",
    "Charcot", "Trousseau", "Babinski", "Broca", "Wernicke's",
]

# Marqueurs qui, juste avant un nom detecte, signalent un terme medical plutot
# qu'un vrai patient. Couvre les eponymes absents de la allow_list.
MEDICAL_PREFIXES = [
    "maladie de", "maladie d'", "syndrome de", "syndrome d'",
    "signe de", "test de", "trouble de",
]

MEDICAL_SUFFIXES = [
    "disease", "syndrome", "'s disease", "'s syndrome",
    "sign", "test", "disorder",
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


def filter_medical_false_positives(text: str, results: list, language: str) -> list:
    """Retire les entites PERSON qui sont en realite des eponymes medicaux,
    detectes soit par allow_list, soit par contexte precedent."""
    filtered = []
    allow_list_lower = [term.lower() for term in ALLOW_LIST]

    for result in results:
        if result.entity_type != "PERSON":
            filtered.append(result)
            continue

        entity_text = text[result.start:result.end].lower()
        if entity_text in allow_list_lower:
            continue  # deja gere par allow_list native, mais double securite
        if is_medical_eponym_by_context(text, result.start, result.end, language):
            continue  # eponyme non liste mais signale par le contexte

        filtered.append(result)

    return filtered


def anonymize_text(text: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine, language: str = "fr") -> str:
    results = analyzer.analyze(text=text, language=language, allow_list=ALLOW_LIST)
    results = filter_medical_false_positives(text, results, language)

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"PERSON": OperatorConfig("replace", {"new_value": "[PATIENT]"})},
    )
    return anonymized.text


if __name__ == "__main__":
    analyzer = build_analyzer()
    anonymizer = AnonymizerEngine()

    test_cases = [
        ("Monsieur Martin presente la maladie de Parkinson", "fr"),
        ("La patiente Sophie Dubois souffre du syndrome de Crohn", "fr"),
        ("Jean-Pierre a consulte pour des douleurs abdominales", "fr"),
        ("The patient John Smith was diagnosed with Parkinson's disease", "en"),
        ("Mary Johnson reported symptoms of Crohn's disease", "en"),
    ]

    for text, lang in test_cases:
        result = anonymize_text(text, analyzer, anonymizer, language=lang)
        print(f"Original  : {text}")
        print(f"Anonymise : {result}\n")