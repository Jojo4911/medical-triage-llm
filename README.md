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
├── notebooks/       # exploration
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

## Statut du projet

En cours de développement.