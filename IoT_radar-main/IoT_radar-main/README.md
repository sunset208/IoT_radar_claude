# IoT Radar — micro-Doppler & calibration

Dépôt pour un dispositif radar **onde continue (CW)** destiné à la **détection de signes de vie** (respiration) en contexte catastrophe, à partir d’un **micro-Doppler** spectral, avec volet **acquisition de données** et **calibration par apprentissage supervisé**.

Deux modules principaux :

| Module | Rôle |
|--------|------|
| [**MicroDopplerDetection/**](MicroDopplerDetection/README.md) | Chaîne temps réel : PlutoSDR ou simulation, décimation, suppression de clutter, fenêtrage, STFT, détection (Fisher × ACF), dashboard. Scripts d’**enregistrement** et de **relecture**. |
| [**AICalibration/**](AICalibration/README.md) | Jeu **PyTorch** `CalibrationDataset`, autoencodeur Conv2D supervisé, entraînement (`train.py`), inférence visuelle (`inference.ipynb`). |

---

## Arborescence racine

```
IoT_radar/
├── requirements.txt       # dépendances Python (venv recommandé)
├── accueil_pg.py          # lanceur graphique pygame → pipeline streaming
├── MicroDopplerDetection/ # détection temps réel + scripts d'enregistrement
└── AICalibration/         # dataset, modèle, entraînement, notebook d'inférence
    ├── data/              # .npz labellisés (ignoré par Git)
    └── results/           # checkpoints & métriques (ignoré par Git, .gitkeep conservé)
```

---

## Matériel visé

- **ADALM-PLUTO** (AD9363), libiio / `pyadi-iio`
- Pilotage typique depuis une machine Linux (dont **WSL2**) avec accès USB au Pluto

---

## Installation

### Prérequis système (Pluto)

```bash
sudo apt update
sudo apt install -y libiio-dev libiio-utils git python3 python3-venv python3-pip build-essential
iio_info -s   # doit lister la carte
```

### Environnement Python

Depuis la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

| Paquet | Module |
|--------|--------|
| `numpy`, `scipy`, `matplotlib`, `pyyaml` | MicroDopplerDetection + AICalibration |
| `pyadi-iio`, `pylibiio` | PlutoSDR |
| `torch`, `tqdm` | Entraînement AICalibration |
| `ipykernel` | `inference.ipynb` |
| `pygame` | `accueil_pg.py` |

---

## Lancement rapide

Toutes les commandes ci-dessous supposent la racine **`IoT_radar`** et :

```bash
export PYTHONPATH="$(pwd)"
source .venv/bin/activate
```

### 1. Pipeline radar (sans matériel)

```bash
python -m MicroDopplerDetection.main --simulation
```

Équivalent depuis le sous-dossier : `cd MicroDopplerDetection && python main.py --simulation`.

### 2. Interface d’accueil (pygame)

```bash
python accueil_pg.py
```

Lance le dashboard streaming via le bouton **Lancer Acquisition** (appelle `MicroDopplerDetection.main`).

### 3. Enregistrer des données pour la calibration

```bash
cd MicroDopplerDetection
python utils/record_acquisition.py --subset train --env salle --label 1 --duration 120
python utils/auto_record.py --subset train -n 10 --interval 30
```

### 4. Entraîner l’autoencodeur

```bash
python AICalibration/train.py
```

Sorties dans `AICalibration/results/` : `best.pt`, `history.csv` (optionnel), `train.log`.

### 5. Inférence & visualisation

Ouvrir **`AICalibration/inference.ipynb`** (kernel du `.venv`) : courbes d’entraînement, matrice de confusion, reconstructions sur quelques fenêtres.

---

## Documentation détaillée

- Pipeline streaming, fenêtrage, détection, enregistrement : [**MicroDopplerDetection/README.md**](MicroDopplerDetection/README.md)
- Format `.npz`, modèle, `config.yaml`, entraînement : [**AICalibration/README.md**](AICalibration/README.md)

---

## Licence

Projet interne — usage académique et recherche.
