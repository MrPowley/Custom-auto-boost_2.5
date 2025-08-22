# Custom Auto-Boost_2.5
[![lang - EN](https://img.shields.io/badge/lang-EN-d5372d?style=for-the-badge)](README.md)
[![lang - FR](https://img.shields.io/badge/lang-FR-2d3181?style=for-the-badge)](README.fr.md)

Ce projet est une version modifiée du projet [auto-boost-algorithm](https://github.com/nekotrix/auto-boost-algorithm) version 2.5 de [nekotrix](https://github.com/nekotrix).

## Version 2.5 : Basé sur SSIMULACRA2 & XPSNR

Dépendances :
- [Vapoursynth](https://github.com/vapoursynth/vapoursynth)
- LSMASHSource
- [Av1an](https://github.com/rust-av/Av1an/)
- [vszip](https://github.com/dnjulek/vapoursynth-zip)
- tqdm
- psutil
- [mkvmerge](https://www.matroska.org/index.html)
  
Optionellement :
- [turbo-metrics](https://github.com/Gui-Yom/turbo-metrics)
- [vship](https://github.com/Line-fr/Vship)

# Changements notables de l'original
- Ajouté une barre de progression pour turbo-metrics
- Ajouté une étape 4 pour un encodage final avec Av1an
- Enlevé la dépendance à ffmpeg pour le calcul de XPSNR
- Agressivité ajustable
- Meilleures options d'implémentations des métriques
- Temporairement enlevé le skip personnalisé
- (Techniquement plus précis avec une faible marge)

# Todo, fonctionnalités
- (En cours) Ajouter un framework d'encodage fait maison (Alternative à Av1an), parce que pourquoi pas
- Pouvoir ajuster plus précisément l'agressivité
- Améliorer le CLI
- (Peut-être) Ajouter un TUI
- (Un jour) Ajouter un GUI

# Todo, code
- Rendre/Maintenir le code complètement conforme au [pep8](https://peps.python.org/pep-0008/)
- Refactoriser le code pour avoir une architecture Model-View-Controller

# Utilisation
|Paramètre|Valeur par défaut|Valeurs possibles|Explication|
|---|---|---|---|
|`-i, --input`|||Chemin de la vidéo|
|`-t, --temp`|Nom de la vidéo||Dossier temporaire|
|`-s, --stage`|`0`|`0`, `1`, `2`, `3`, `4`|Étape de boost (Tout, passe rapide, calcul des métriques, calcul des zones, encodage final)|
|`-crf`|`30`|`1`-`63`|CRF à utiliser|
|`-d, -deviation`|`10`|`0`-`63`|Écart maximum du CRF de base|
|`--max-positive-dev`||`0`-`63`|Écart maximum positif du CRF de base|
|`--max-negative-dev`||`0`-`63`|Écart maximum négatif du CRF de base|
|`-p, --preset`|`6`|`-1`-`13`|Preset SVT-AV1 à utiliser|
|`-w, --workers`|Nombre de coeurs CPU| `1`-`Any`|Nombre de workers Av1an|
|`-m, --method`|`1`|`1`, `2`, `3`, `4`|Méthode de calcul des zones(1: SSIMU2, 2: XPSNR, 3: Multiplié, 4: Minimum)|
|`-a, --aggressiveness`|`20` ou `40`|`0` - `Any`|Choisir l'agréssivité du boost|
|`-M, --metrics-implementations`|`vship,vszip`|`<vszip, vship, turbo-metrics>,<vszip>`|Choisir l'implémentation de chaque métrique. Premier: SSIMULACRA2, deuxième: XPSNR|
|`-v --video-params`|||Paramètres d'encodeur pour Av1an|
|`-ef, --encoder-framework`|`av1an`|`av1an`|Framework d'encodage à utiliser|
|`-o, --output`|Dossier de la vidéo original||Fichier de sortie pour l'encodage final|
<!-- |`-S, --skip`|`1` (GPU), `3` (CPU)|`1`-`Any`|Calculer le score toutes les X images| -->

---

_Ce projet est basé sur le travail original de **nekotrix**._  
_Les contributeurs originaux incluent **R1chterScale**, **Yiss**, **Kosaka**, et d'autres._  
_License MIT, voir LICENSE._