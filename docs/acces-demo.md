# Accès à l'endpoint de démonstration

L'API de triage n'est pas exposée publiquement (aucune règle firewall ouverte).
Accès via tunnel SSH uniquement, à la demande.

## Commande

gcloud compute ssh p14-sft-lora-c --zone=europe-west4-c -- -L 8080:localhost:8080

## Test une fois le tunnel actif

curl http://localhost:8080/health
curl -X POST http://localhost:8080/triage -H "Content-Type: application/json" -d '{"symptoms": "..."}'

## Prérequis avant la démo

- VM démarrée : gcloud compute instances start p14-sft-lora-c --zone=europe-west4-c
- vLLM lancé (fenêtre SSH dédiée) : voir tmux session "vllm" ou relancer si besoin
- Conteneur API actif : docker ps doit montrer "triage-api" Up