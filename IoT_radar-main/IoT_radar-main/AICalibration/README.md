# AICalibration — Autoencodeur supervisé micro-Doppler

Apprentissage **supervisé** sur spectrogrammes STFT enregistrés par le pipeline **MicroDopplerDetection** : autoencodeur Conv2D + tête de classification binaire (présence / respiration).

Entraînement orchestré dans **`train.py`**.

---

## Contexte & rôle dans le dépôt

Les `.npz` de calibration sont produits côté radar par **`MicroDopplerDetection/utils/record_acquisition.py`** (ou **`auto_record.py`**) et stockés par défaut sous **`AICalibration/data/<train|test|val>/`**.

Ce dossier fournit :

1. **`CalibrationDataset`** — fenêtres glissantes STFT + labels pour PyTorch ;
2. **`SpectrogramAutoencoder`** — encodeur / décodeur Conv2D + MLP de classification ;
3. **`train.py`** — boucle d'entraînement, checkpoint, historique CSV ;
4. **`inference.ipynb`** — courbes d'entraînement + visualisation de reconstructions.

---

## Architecture (flux données → modèle)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MicroDopplerDetection/utils/record_acquisition.py                       │
│  → AICalibration/data/<train|val|test>/<n>.npz                           │
│     clés : spectrogram_db (N_frames × n_fft), label, env, métadonnées    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │    dataset.py       │
                     │ CalibrationDataset  │
                     │ fenêtres (1,H,W)    │
                     │ H=n_fft, W=n_cols   │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │     model.py        │
                     │SpectrogramAutoencoder│
                     │  encode → latent    │
                     │  decode → x_hat     │
                     │  head   → logits    │
                     └──────────┬──────────┘
                                │
              loss = α·recon + (1−α)·BCE
                                │
                     ┌──────────▼──────────┐
                     │     train.py        │
                     │  best.pt, history   │
                     │  .csv, train.log    │
                     └─────────────────────┘
```

**Forme d'entrée par défaut** : `(B, 1, 8192, 32)` — fréquence × temps.  
**Latent par défaut** : `(128, 32, 2)` (compression 32×).

---

## Arborescence utile

```
AICalibration/
├── config.yaml          # source de vérité (données, modèle, entraînement, sorties)
├── dataset.py           # CalibrationDataset
├── model.py             # SpectrogramAutoencoder (+ smoke test : python model.py)
├── train.py             # entraînement CLI
├── inference.ipynb      # courbes + inférence visuelle
├── data/                # enregistrements .npz (ignoré par Git)
│   ├── train/
│   ├── val/
│   └── test/
└── results/             # sorties d'entraînement (ignoré ou partiellement versionné)
    ├── best.pt          # meilleur checkpoint (config embarquée)
    ├── history.csv      # métriques train/val par epoch (optionnel)
    └── train.log        # journal détaillé
```

---

## Format des `.npz`

Chaque fichier contient au minimum :

| Clé | Description |
|-----|-------------|
| `spectrogram_db` | `float`, forme `(N_frames, n_fft)` — colonnes STFT successives (dB) |
| `label` | `0` = salle vide, `1` = présence / respiration |
| `env` | étiquette d'environnement (texte) |
| `n_fft`, `f_s_dec_hz` | paramètres spectraux |

Métadonnées additionnelles : `subset`, `sample_index`, `config_path`, `t_wall_s`, etc. Un **`.json`** auxiliaire est écrit à côté du `.npz` par `record_acquisition`.

**`CalibrationDataset`** transpose en `(n_fft, N_frames)`, découpe en fenêtres de **`n_cols`** colonnes (défaut 32 ≈ 14 s), normalise optionnellement chaque fenêtre (zéro-mean / unit-var), et renvoie `(tensor[1, n_fft, n_cols], label)`.

---

## Installation

### Prérequis

- Python 3.10+
- Environnement avec **PyTorch** (CPU ou CUDA)
- Racine du dépôt **`IoT_radar`** sur le `PYTHONPATH`

### Dépendances Python

| Paquet | Rôle |
|--------|------|
| `torch` | modèle, entraînement, DataLoader |
| `numpy` | chargement `.npz`, fenêtres STFT |
| `matplotlib` | `inference.ipynb` |
| `pyyaml` | `config.yaml` |
| `tqdm` | barre de progression à l'entraînement |

Les dépendances radar (`pyadi-iio`, …) ne sont **pas** requises pour entraîner sur des `.npz` déjà enregistrés.

---

## Utilisation

### 1. Enregistrer des données (MicroDopplerDetection)

```bash
cd MicroDopplerDetection
python utils/record_acquisition.py --subset train --env salle --label 1 --duration 120
python utils/auto_record.py --subset train -n 10 --interval 30   # plusieurs prises
```

Voir **`MicroDopplerDetection/README.md`** pour le détail des scripts d'enregistrement.

### 2. Entraîner

Depuis la racine **`IoT_radar`** :

```bash
export PYTHONPATH="$(pwd)"
python AICalibration/train.py
```

Depuis ce dossier :

```bash
cd AICalibration
python train.py
python train.py --config config.yaml --epochs 100 --batch-size 16 --lr 5e-4 --device cuda
```

Au démarrage : chargement des données, **tableau récapitulatif** (device, split, modèle, hyperparamètres), puis entraînement avec barre **tqdm** tous les **`progress_interval`** epochs (défaut 10). La validation est exécutée **à chaque epoch** ; seul l'affichage console est espacé.

### 3. Inférence & visualisation

Ouvrir **`inference.ipynb`** (kernel pointant sur le `.venv` du dépôt) :

- courbes **loss / accuracy** depuis `results/history.csv` (si `save_metrics: true`) ;
- chargement de `results/best.pt` et comparaison entrée / reconstruction sur quelques fenêtres du jeu train.

### Options CLI (`train.py`)

| Option | Défaut | Description |
|--------|--------|-------------|
| `--config` | `AICalibration/config.yaml` | Fichier YAML |
| `--epochs` | *(YAML)* | Surcharge `training.epochs` |
| `--batch-size` | *(YAML)* | Surcharge `dataloader.batch_size` |
| `--lr` | *(YAML)* | Surcharge `training.optimizer.lr` |
| `--device` | *(YAML)* | `auto`, `cpu` ou `cuda` |

---

## Paramètres (`config.yaml`)

| Section | Rôle |
|---------|------|
| `seed`, `device` | reproductibilité, CPU / CUDA |
| `data` | `data_root`, splits `train`/`val`, `n_cols`, `stride`, `normalise`, `val_split` |
| `dataloader` | `batch_size`, `num_workers`, `pin_memory` |
| `model` | `expected_input_shape`, canaux encodeur, `downsample_factors`, noyaux, classifieur |
| `training` | `epochs`, loss composite (`alpha`, `reconstruction`, `bce_pos_weight`), AdamW, StepLR, `best_metric`, `progress_interval` |
| `output` | `results_dir`, `checkpoint_name`, `log_file`, `save_metrics`, `metrics_file` |

Détails et valeurs par défaut dans les commentaires du YAML.

**Split validation** : si `data/val/` contient au moins un `.npz`, il sert de val ; sinon split aléatoire de `data/train/` selon `val_split`.

**Loss** : `α · recon + (1 − α) · BCE` — reconstruction MSE ou L1, classification binaire (label respiration). Meilleur checkpoint selon `best_metric` (`val_total`, `val_bce` ou `val_recon`).

---

## Sorties d'entraînement

| Fichier | Contenu |
|---------|---------|
| `results/best.pt` | poids, config, epoch, métriques val au moment du record |
| `results/history.csv` | par epoch : `train_*` / `val_*` (total, recon, bce, acc), `lr` |
| `results/train.log` | journal INFO (split, checkpoints, …) |

Désactiver l'export CSV : `output.save_metrics: false`.

---

## Fichiers Python

| Fichier | Rôle |
|---------|------|
| `dataset.py` | `CalibrationDataset`, constante `N_COLS` |
| `model.py` | `SpectrogramAutoencoder`, `Encoder`, `Decoder`, `ClassifierHead` |
| `train.py` | chargement config, boucles train/val, checkpoint, CSV |

---

## Licence

Projet interne — usage académique et recherche.
