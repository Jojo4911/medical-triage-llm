# Medical Triage LLM

POC d'un agent IA de triage médical pour le service des urgences, développé pour le Centre Hospitalier Saint-Aurélien (CHSA).

## Objectif

Fine-tuner un modèle de langage compact (Qwen3-1.7B) pour assister le personnel soignant dans l'évaluation initiale du niveau de priorité des patients (urgence maximale / modérée / différée), à partir de leurs symptômes déclarés.

## Approche

- **SFT (Supervised Fine-Tuning)** avec LoRA sur un dataset médical bilingue FR/EN
- **DPO (Direct Preference Optimization)** pour aligner le modèle sur les pratiques cliniques validées
- **Déploiement** via vLLM, exposé en API FastAPI
- **CI/CD** automatisé avec GitHub Actions

Statut : Proof of Concept, pas de mise en production.

## Structure du projet

```
├── data/
│   ├── raw/         # données brutes, non versionnées dans git (DVC)
│   ├── processed/   # données nettoyées et anonymisées, non versionnées dans git (DVC)
│   ├── eval/        # jeu d'évaluation clinique
├── src/             # code source
├── scripts/         # scripts d'exploration, de préparation des données et d'entraînement
│   ├── checks/      # scripts de vérification ponctuelle (audits, diagnostics)
├── models/          # checkpoints et poids, non versionnés dans git (DVC)
├── reports/         # rapport technique et livrables
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

## Modèle

- **SFT + LoRA** : entraîné sur ~5 000 paires instruction-réponse, format structuré (Niveau d'urgence / Service / Explication). Config LoRA : `r=16, lora_alpha=32, target_modules=[q_proj,k_proj,v_proj,o_proj]`. Checkpoint versionné DVC (`models/sft-lora-qwen3-1.7b`).
- **DPO** : entraînement terminé à partir du modèle SFT, sur les 5 000 paires préférentielles issues d'UltraMedical-Preference. Config : `beta=0.1, learning_rate=5e-6, num_train_epochs=1`, adaptateur LoRA désactivable utilisé comme politique de référence (pas de copie séparée du modèle en mémoire). Checkpoint versionné DVC (`models/dpo-lora-qwen3-1.7b`).

Métriques finales du run DPO (dernier step logué, tracking Weights & Biases) :
- `rewards/accuracies` : 0,75
- `rewards/margins` : 0,84
- `rewards/chosen` : +0,55, `rewards/rejected` : -0,29
- `train_loss` : 0,55

Limite connue côté SFT seul, avant alignement DPO : le modèle pouvait produire des hallucinations de contenu clinique, dont certaines à risque (diagnostic différentiel erroné sur des tableaux évocateurs d'urgence vitale). Point de comparaison de référence pour l'évaluation post-DPO à venir (comparaison base / SFT / SFT+DPO sur le jeu d'évaluation clinique).

## Infrastructure

- Entraînement sur VM GCP (`europe-west4`, GPU NVIDIA L4). Le run DPO complet a été exécuté en zone `europe-west4-c` suite à une indisponibilité temporaire (stockout) de la zone `europe-west4-a` initialement utilisée pour le SFT.
- Versionnement des données et modèles via DVC, remote GCS.
- Tracking des runs via Weights & Biases (projet `huggingface`).

## Statut du projet

En cours de développement. Dataset bilingue anonymisé, modèle SFT+LoRA, et alignement DPO tous finalisés. Évaluation comparative (base / SFT / SFT+DPO) et contrôles de sécurité clinique à venir.