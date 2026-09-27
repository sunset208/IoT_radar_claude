# IA — état des lieux et marche à suivre

## Ce qu'il y avait (v1, `AICalibration/`)

Un autoencodeur Conv2D avec une tête de classification, entraîné sur des
spectrogrammes dB de 8192 × 32 (±1000 Hz sur ~13 s).

* **L'entrée est presque entièrement hors sujet.** Le signal utile occupe
  quelques bins autour de +500 Hz, et la phase (l'information principale) a
  été jetée.
* **La validation fuit.** Le split est aléatoire par fenêtre : des fenêtres
  voisines d'un même enregistrement se retrouvent en train et en val.
* **Le résultat n'est pas probant.** 92.6 % de précision sur 27 fenêtres de
  validation, alors que répondre toujours « respiration » donne 80 %. Un seul
  environnement.
* La reconstruction dominait l'apprentissage : on apprenait surtout à
  reproduire du bruit.

Le `best.pt` v1 n'est pas réutilisable avec la nouvelle entrée.

## Ce qu'il y a maintenant (`ai/`)

| Fichier | Rôle |
|---|---|
| `preprocess.py` | Fenêtre slow-time complexe → tenseur (3, T). Rotation dans la base (radiale, tangentielle) du clutter, normalisation par le bruit (les amplitudes sont en « unités de SNR »), suppression de la dérive lente. **Identique** à l'entraînement et en temps réel. |
| `data.py` | `SyntheticWindows` : simulateur physique avec randomisation de domaine (données infinies). `RecordingWindows` : enregistrements réels, avec leurs groupes de session. `SemiSyntheticWindows` : **vrai fond « vide » + personne simulée**. |
| `models.py` | `ResNet1D` (0.5 M paramètres, convolutions dilatées, champ réceptif > 20 s) et `ConvTransformer` (3 M). Deux têtes : présence, et rythme (tâche auxiliaire). |
| `train.py` | `pretrain` (simulateur seul) puis `finetune` (mélange réel, semi-synthétique et synthétique). Validation **par session**, bf16, arrêt anticipé. Rapport avec AUC, Pd à Pfa = 1 % et **comparaison au détecteur classique sur les mêmes fenêtres**. |
| `infer.py` | `radar run --ai ai/runs/.../best.pt` : le score IA s'affiche en parallèle du détecteur classique (courbe orange). |

Vérifié sans entraînement : `python ai/train.py pretrain --check` (un seul
passage avant, aucun poids modifié).

### Corrections du 27/09 (nuit)

* **Semi-synthétique** : la cible simulée ajoutée aux fonds vides réels
  utilisait la longueur d'onde de `ai/config.yaml` (3.5 GHz) alors que les fonds
  sont enregistrés à 1.8 GHz : profondeur de modulation fausse d'un facteur 2.
  La porteuse de chaque enregistrement (métadonnées) est maintenant utilisée.
* **Synthétique** : λ tirée parmi 1.8 / 2.45 / 5.8 GHz (`data.wavelength` est
  une liste) → le même modèle vaut après le passage à 5.8 GHz.
* **Sélection du modèle** : en pré-entraînement, elle se fait sur la
  validation synthétique.  Avant, dès qu'une validation réelle existait, elle
  servait de critère ; or avec 6 sessions le tirage peut n'y mettre que des
  positifs → AUC NaN → **aucun `best.pt` n'était sauvegardé**.
* **Vérité terrain du rythme** : si un module 60 GHz était branché pendant
  l'enregistrement (`<fichier>.mr60.json`), son rythme sert d'étiquette à la
  tête « rythme ».
* Les enregistrements SFCW (autre format) sont ignorés proprement.

## Marche à suivre sur la machine d'entraînement (5090)

```bat
:: 0. données : enregistrements v2 dans data\recordings (+ v1 convertis)
::    le plus utile : des séances guidées (≥ 3 « vide » et ≥ 3 « respiration »)
radar.bat --config configs\lab.yaml protocole --plan vide_long
radar.bat --config configs\lab.yaml protocole --plan labo --mr60 COM5
radar.bat convert-v1 <ancien AICalibration\data>

:: 1. pré-entraînement sur simulateur (quelques dizaines de minutes au plus)
..\.venv\Scripts\python.exe ai\train.py pretrain

:: 2. affinage sur le réel (il faut des sessions « vide » ET « respiration »)
..\.venv\Scripts\python.exe ai\train.py finetune --init ai\runs\pretrain_<date>\best.pt

:: 3. lire ai\runs\finetune_<date>\report.json :
::    final.real.auc / pd_at_pfa  vs  classical_baseline_real_val
```

**Règle de décision** : l'IA ne remplace le détecteur classique que si elle le
bat sur des **sessions (idéalement des lieux) jamais vues**, en Pd à Pfa fixé.
Sinon on garde le classique, qui est explicable et calibrable.

## Ce qui fera vraiment la différence (par ordre d'impact)

1. **Des données réelles variées, splittées par session ou par lieu** (protocole
   au §6.2 du rapport). Sans elles, aucun modèle ne sera crédible, quelle que
   soit la taille du GPU.
2. **Le semi-synthétique** : chaque minute de salle vide réelle devient un
   nombre illimité de positifs réalistes (vrai clutter, vraie dérive du Pluto,
   vrais parasites). C'est la meilleure façon d'exploiter peu de données avec
   personne.
3. **La vérité terrain du rythme** (ceinture ou téléphone sur le thorax) :
   elle entraîne la tête « rythme » et permet de mesurer l'erreur en resp/min.
4. Ensuite seulement : modèle plus gros (`convtransformer`), ensembles,
   calibration des probabilités, export ONNX pour un Raspberry Pi.

Pistes plus ambitieuses, une fois une base solide en place : distinguer
respiration et ventilateur/rideau (la forme d'onde et l'amplitude en mm
diffèrent), gérer plusieurs personnes (il faut une information spatiale, donc
du matériel 2T2R/FMCW), reconnaître la position du sujet.
