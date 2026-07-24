# Medical Triage LLM

POC d'un agent IA de triage médical pour le service des urgences, développé pour le Centre Hospitalier Saint-Aurélien (CHSA).

## Objectif

Fine-tuner un modèle de langage compact (Qwen3-1.7B) pour assister le personnel soignant dans l'évaluation initiale du niveau de priorité des patients (urgence maximale / modérée / différée), à partir de leurs symptômes déclarés.

## Approche

- **SFT (Supervised Fine-Tuning)** avec LoRA sur un dataset médical bilingue FR/EN
- **DPO (Direct Preference Optimization)** pour aligner le modèle sur les pratiques cliniques validées
- **Déploiement** via vLLM, exposé en API FastAPI, conteneurisé avec Docker
- **CI/CD** automatisé avec GitHub Actions (à venir)

Statut : Proof of Concept, pas de mise en production.

## Structure du projet

```
├── api/             # API FastAPI exposant l'endpoint de triage
├── data/
│   ├── raw/         # données brutes, non versionnées dans git (DVC)
│   ├── processed/   # données nettoyées et anonymisées, non versionnées dans git (DVC)
│   ├── eval/        # jeu d'évaluation clinique
├── src/             # code source
├── scripts/         # scripts d'exploration, de préparation des données et d'entraînement
│   ├── checks/      # scripts de vérification ponctuelle (audits, diagnostics)
├── models/          # checkpoints et poids, non versionnés dans git (DVC)
├── reports/         # rapport technique et livrables
├── Dockerfile       # conteneurisation de l'API de triage
```

## Setup

```bash
uv sync
uv run dvc pull  # récupère les données versionnées
```

## Sources de données

| Source | Licence | Date d'accès | Notes |
|---|---|---|---|
| [MediQA](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | CC-BY-4.0 | 13/07/2026 | Config `mcqu` utilisée : 10 113 train / 2 561 val / 4 343 test. Le volume total du dataset (32 603) inclut d'autres configs non retenues ici. |
| [FrenchMedMCQA](https://huggingface.co/datasets/qanastek/frenchmedmcqa) | Apache 2.0 | 13/07/2026 | 2 171 train / 312 val / 622 test, 3 105 questions au total (examens de pharmacie). |
| [MedQuAD (keivalya)](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) | Non spécifiée sur le repo HF source | 13/07/2026 | 16 407 exemples, split unique. Un mirror Kaggle équivalent indique CC0, non formellement confirmé sur cette source. Contenu à faible risque RGPD (Q&A génériques issues de sites publics d'information santé, pas de données patients). |
| [UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | MIT | 13/07/2026 | 109 353 train / 2 232 val / 777 test. Paires chosen/rejected pour la construction du jeu DPO. |

## Anonymisation RGPD

Pipeline basé sur Presidio (`AnalyzerEngine`/`AnonymizerEngine`), avec `allow_list` dédiée aux éponymes médicaux non ambigus et filtre de contexte (`maladie de`, `syndrome de`) pour limiter les faux positifs sur le vocabulaire clinique.

Deux points d'entrée :
- `scripts/anonymize_dataset.py` : traite le dataset SFT seul.
- `scripts/anonymize_datasets.py` : orchestre l'anonymisation SFT **et** DPO.

Limite connue et documentée, non bloquante : résidu de tags `<LOCATION>` sur des termes médicaux figés à consonance géographique (ex. noms de syndromes), quantifié à environ 0,9% sur le jeu DPO (`scripts/checks/check_dpo_anonymization.py`). Décision assumée de ne pas itérer davantage sur Presidio au-delà de ce point pour ce POC.

**Point ouvert (identifié lors des tests d'inférence, J10)** : des tags d'anonymisation non résolus (`<LOCATION>`, `[PATIENT]`) apparaissent parfois dans les réponses générées par le modèle, suggérant qu'un sous-ensemble des paires SFT contient des placeholders Presidio non nettoyés avant l'entraînement. À investiguer et documenter dans le rapport comme limite du pipeline de données.

## Modèle

- **SFT + LoRA** : entraîné sur ~5 000 paires instruction-réponse, format structuré (Niveau d'urgence / Service / Explication). Config LoRA : `r=16, lora_alpha=32, target_modules=[q_proj,k_proj,v_proj,o_proj]`. Checkpoint versionné DVC (`models/sft-lora-qwen3-1.7b`).
- **DPO** : entraînement terminé à partir du modèle SFT, sur les 5 000 paires préférentielles issues d'UltraMedical-Preference. Config : `beta=0.1, learning_rate=5e-6, num_train_epochs=1`, adaptateur LoRA désactivable utilisé comme politique de référence (pas de copie séparée du modèle en mémoire). Checkpoint versionné DVC (`models/dpo-lora-qwen3-1.7b`).

Métriques finales du run DPO (dernier step logué, tracking Weights & Biases) :
- `rewards/accuracies` : 0,75
- `rewards/margins` : 0,84
- `rewards/chosen` : +0,55, `rewards/rejected` : -0,29
- `train_loss` : 0,55

## Évaluation clinique (base vs SFT vs SFT+DPO)

Comparaison qualitative menée sur 7 vignettes cliniques (`scripts/compare_base_vs_sft.py`), incluant des cas ciblés sur des confusions diagnostiques à risque (AVC, appendicite).

Constat principal : le DPO améliore la pertinence générale de forme (structure, niveau d'urgence, service), mais **ne corrige pas de façon fiable les erreurs de diagnostic différentiel sur des cas ambigus**, et introduit un risque spécifique de **fabrication d'éléments cliniques absents de l'énoncé** (antécédents, résultats d'examens inventés). Ce risque n'est pas capturé par les métriques de reward globales.

Une dérive occasionnelle hors du vocabulaire fermé attendu pour `emergency_level` a également été observée en test d'inférence.

Détail complet des cas observés : `notes/limites-securite-clinique-dpo.md` (non versionné, usage interne).

Ce constat n'appelle pas de nouvel entraînement dans le cadre du POC. Il alimente la section limites et la roadmap du rapport technique (garde-fous à prévoir en production : vérification factuelle, citation des sources de l'énoncé, refus de générer un antécédent ou un résultat d'examen non fourni).

## Déploiement

- **Serveur d'inférence** : vLLM, servant le modèle SFT+DPO avec l'adaptateur LoRA fusionné (`merge_and_unload`) au modèle de base, sur le port 8000.
- **API** : FastAPI (`api/main.py`), endpoint `POST /triage`, reconstruit le format de prompt d'entraînement à partir des symptômes fournis, journalise chaque interaction (`logs/triage_interactions.jsonl`) pour traçabilité et audit médical.
- **Conteneurisation** : image Docker dédiée à l'API (dépendances minimales, pas d'entraînement embarqué), communique avec le serveur vLLM via `--network host`.
- **CI/CD** : pipeline GitHub Actions à mettre en place (prochaine étape).

## Infrastructure

- Entraînement et inférence sur VM GCP (`europe-west4-c`, `g2-standard-8`, GPU NVIDIA L4). Le run DPO complet a été exécuté en zone `europe-west4-c` suite à une indisponibilité temporaire (stockout) de la zone `europe-west4-a` initialement utilisée pour le SFT.
- Versionnement des données et modèles via DVC, remote GCS.
- Tracking des runs via Weights & Biases (projet `p14-triage-medical`).
- Dépendances de sécurité : `cryptography` et `gitpython` mis à jour suite à alertes Dependabot (high). Une vulnérabilité modérée résiduelle sur `diskcache` (dépendance transitive DVC) reste sans correctif amont disponible, risque accepté et documenté pour ce cadre POC.

## Statut du projet

Dataset bilingue anonymisé, modèle SFT+LoRA, alignement DPO, évaluation clinique comparative, et endpoint de démonstration (vLLM + FastAPI + Docker) tous finalisés. Restent à faire : pipeline CI/CD GitHub Actions, rédaction du rapport technique, préparation de la soutenance.