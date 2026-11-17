# Modèle de probabilité de défaut (PD) : scorecard par régression logistique

Pipeline complet de modélisation de la PD sur un portefeuille de crédit à la consommation, dans l'esprit des modèles réglementaires de type Retail (Bâle / IRB) : discrétisation WoE, sélection de variables, régression logistique, scorecard, évaluation de la discrimination, de la calibration et de la stabilité.

## Données

Jeu de données utilisé : **Give Me Some Credit** (compétition Kaggle 2011), fichier `cs-training.csv`, soit 150 000 emprunteurs et un taux de défaut d'environ 6,7 % (cible `SeriousDlqin2yrs` : défaut de paiement de 90 jours ou plus dans les 2 ans). Le fichier n'est pas versionné : à télécharger sur Kaggle et à placer dans `data/`.

Format attendu pour un autre jeu de données : un CSV avec une colonne cible binaire (1 = défaut), éventuellement une colonne identifiant (`--id-col`) et une colonne temporelle (`--time-col`). Toute variable connue seulement après l'octroi du crédit doit être exclue (fuite d'information) ; le script signale les variables dont l'IV dépasse 0,5.

## Utilisation

```bash
pip install -r requirements.txt

python src/run_pd_model.py --data data/cs-training.csv --target SeriousDlqin2yrs \
    --id-col "Unnamed: 0" --out outputs
```

Un jeu de données synthétique (`python src/make_synthetic.py`) est fourni uniquement pour tester que le code s'exécute ; il ne sert pas à produire de résultats.

## Méthode

1. **Découpage** : out-of-time si `--time-col` est fourni, sinon aléatoire stratifié (70 % apprentissage / 30 % test).
2. **Binning WoE** (`src/binning.py`) : classes par quantiles, fusion des classes adjacentes jusqu'à un taux de défaut monotone et une taille minimale de 5 %, classe dédiée aux valeurs manquantes, regroupement des modalités rares. Convention : WoE = ln(% défauts / % non-défauts).
3. **Sélection** : Information Value minimale (0,02), puis élimination des variables dont la corrélation de WoE dépasse 0,7.
4. **Régression logistique** sur les WoE ; toute variable au coefficient de signe contraire à l'attendu est retirée (contrôle de cohérence économique).
5. **Scorecard** : points par classe, avec PDO = 20 et 600 points pour des odds de 50:1 (paramétrable).
6. **Évaluation** (`src/metrics.py`) : AUC / Gini / KS en apprentissage et en test ; calibration par déciles de PD ; PSI du score et de chaque variable entre apprentissage et test.

## Sorties (`outputs/`)

`metrics.json`, `iv_table.csv`, `coefficients.csv`, `binning_tables.csv`, `scorecard_points.csv`, `calibration_deciles.csv`, `roc.png`, `calibration.png`.

## Résultats (Give Me Some Credit, découpage aléatoire 70 / 30)

Variables retenues : utilisation du crédit renouvelable, nombre de retards de 30-59 jours, de 60-89 jours et de 90 jours ou plus, âge, revenu mensuel, nombre de personnes à charge.

| Indicateur | Apprentissage | Test |
|---|---|---|
| Gini | 0,704 | 0,707 |
| KS | 0,546 | 0,548 |

- PD moyenne prédite sur le test : 6,61 % pour un taux de défaut observé de 6,68 %.
- PSI du score entre apprentissage et test : 0,0004.
- Les performances sont proches en apprentissage et en test, ce qui ne suggère pas de surapprentissage.
- La calibration par déciles est proche de la diagonale ; le risque est légèrement sous-estimé dans le décile le plus risqué (voir `calibration.png`).
- L'utilisation du crédit renouvelable et les variables de retard de paiement portent l'essentiel du pouvoir discriminant. Leur IV dépasse 0,5 : ce sont des prédicteurs très forts et non une fuite d'information, ces variables décrivant le comportement passé de l'emprunteur.

## Limites

- Le découpage est aléatoire (pas de dimension temporelle disponible) : le PSI quasi nul ne prouve donc pas la stabilité du modèle dans le temps.
- Les variables de retard sont regroupées en « 0 / plus de 0 » (ou en 3 classes pour les retards de 30-59 jours) après fusion monotone, ce qui ne distingue pas un retard de plusieurs. Les valeurs 96 et 98, codes spéciaux de ces variables dans ce jeu de données, ne sont pas traitées à part.
- La PD obtenue est *point-in-time* sur l'échantillon. Une PD réglementaire demanderait une calibration sur un taux de défaut moyen de long terme, la définition du défaut de Bâle, une marge de prudence et une validation indépendante.
- Pas de traitement du rejet d'inférence.

## Pistes

- Traitement spécifique des codes 96 / 98 et plafonnement des valeurs extrêmes.
- Calibration sur un taux central de long terme.
- Comparaison avec un gradient boosting en référence.
- Extension à la LGD et au CCF.