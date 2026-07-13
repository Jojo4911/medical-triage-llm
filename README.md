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
│   ├── processed/   # données nettoyées, non versionnées dans git (DVC)
│   ├── eval/        # jeu d'évaluation clinique
├── src/             # code source
├── scripts/         # scripts d'exploration et de préparation des données
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

## Statut du projet

En cours de développement.