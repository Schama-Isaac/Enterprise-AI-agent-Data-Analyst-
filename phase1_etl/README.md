# Phase 1 — ETL vectoriel et validation avec dataset fictif

## Objectif
Valider la logique de base de la phase 1 avant de remplacer le dataset fictif par 10 documents réels.

## Étapes
1. Démarrer Qdrant localement.
2. Exécuter `python phase1_etl/01_fake_ingest.py` pour charger les documents fictifs dans Qdrant.
3. Exécuter `python phase1_etl/02_phase1_validation.py` pour vérifier que la recherche sémantique fonctionne avec métadonnées.

## Ce que vous devez vérifier
- la collection contient bien les documents attendus,
- les résultats retournés correspondent à la requête et au filtre metadata,
- la logique est stable avant de passer à la phase 2.
