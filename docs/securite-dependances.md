# Vulnérabilités de dépendances, suivi et décisions

## cryptography, OpenSSL vulnérable dans les wheels (high)

- Dépendance transitive via `dvc` et `presidio-anonymizer`.
- Version vulnérable : 46.0.7. Version corrigée : 48.0.1.
- Action : `cryptography` épinglé en 48.0.1 dans `pyproject.toml`.
- Compatibilité vérifiée avec DVC (`dvc status`) et Presidio (`AnonymizerEngine`), aucune régression observée.
- Statut : résolu.

## diskcache, désérialisation pickle non sûre (moderate)

- Dépendance transitive via `dvc-objects` (composant interne de DVC), version 5.6.3.
- Aucune version corrigée disponible en amont a ce jour.
- Condition d'exploitation : necessite un acces en ecriture au répertoire de cache par un tiers non autorisé.
- Contexte du projet : usage local (poste de travail) et sur VM GCP dediée, sans exposition reseau du cache DVC ni accés multi-utilisateur.
- Décision : risque accepté pour le cadre POC. Retirer DVC pour éliminer cette dependance transitive serait disproportionné au regard du risque réel et remettrait en cause un livrable (versionnement des données et modèles).
- Statut : risque accepté et documenté, pas d'action technique supplémentaire prévue.

## setuptools, MANIFEST.in bypass (Moderate, GHSA-h35f-9h28-mq5c)

Corrigé en 83.0.0. Verrouille a 80.10.2 : vllm 0.25.1 impose setuptools>=77.0.3,<81.0.0 comme dépendance transitive, incompatible avec le correctif. Risque non exploitable dans notre contexte (vulnerabilité liée a la construction de sdist sur macOS APFS/HFS+, sans rapport avec notre usage de setuptools ici). Risque accepté, aucune action possible sans abandonner
vllm.

## Bilan final, 24/07

3 vulnerabilités résiduelles (1 moderate setuptools, 1 moderate diskcache, 1 low torch), toutes documentées, toutes issues de contraintes de compatibilité transitives non contournables sans dégrader une dépendance coeur du projet (vllm). 10 CVE vllm corrigees en relevant vllm de 0.22.1 a 0.25.1 via restriction de la resolution uv a la plateforme Linux (sys_platform == 'linux'), la resolution universelle Windows+Linux retombant sur une version anterieure au correctif.