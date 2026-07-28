## Commande de lancement du conteneur (avec persistance des logs)

docker run -d --name triage-api --network host -v ~/medical-triage-llm/logs:/app/logs triage-api:poc
