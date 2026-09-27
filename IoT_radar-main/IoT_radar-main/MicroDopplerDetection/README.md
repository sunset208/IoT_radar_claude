# Radar Micro-Doppler — Détection de survivants ensevelis

Système radar portable à onde continue (CW) pour détecter la respiration de personnes ensevelies sous des décombres (séisme, avalanche, effondrement).

Matériel visé : **PlutoSDR** (ADALM-PLUTO, AD9363).
Orchestration **streaming** dans **`main.py`**.

---

## Contexte physique

Un signal CW à $f_c \approx 2.4\ \text{GHz}\ (\lambda \approx 12.5\ \text{cm})$ illumine la scène. Le mouvement thoracique (**$f_v \approx 0.2–0.5 \text{Hz}$**, amplitude **$D \approx 5–15 \text{mm}$**) module la phase du signal reçu. En bande de base :

$$x(t) = e^{j\varphi(t)}, \qquad \varphi(t) = \varphi_0 - \frac{4\pi D}{\lambda}\sin(2\pi f_v t)$$

Les raies micro-Doppler autour de **$\pm f_v$** sont analysées après STFT (Short-Time Fourier Transform).

---

## Architecture du pipeline (streaming)

La fenêtre FFT utilisée pour la colonne spectrale est choisie dans **`configs/config.yaml`** (`windowing.mode`) et instanciée par **`pipeline/windowing.py`** (`get_window`), puis passée à **`spectrogramme.compute_single_column`**.

```
┌─────────────┐      ┌────────────────┐      ┌───────────────┐
│ emission.py │─────▶│ acquisition.py │─────▶│decimation.py │
│  buffer TX  │      │ IQ Pluto / sim │      │ Decimator     │
│  ×2¹⁴ scale │      │                │      │ stateful      │
└─────────────┘      └────────────────┘      └──────┬────────┘
                                                    │
                                            ┌───────▼────────┐
                                            │   clutter.py   │
                                            │ ClutterFilter  │
                                            └───────┬────────┘
                                                    │ buffer n_fft + hop
                      ┌─────────────────────────────┼─────────────────────────┐
                      │                             │                         │
                      │                      ┌──────▼───────┐                 │
                      │                      │ windowing.py │ ← config YAML   │
                      │                      │  get_window  │                 │
                      │                      └──────┬───────┘                 │
                      │                             │ fenêtre × segment IQ    │
                      │                      ┌──────▼───────┐                 │
                      │                      │spectrogramme │                 │
                      │                      │compute_single│                 │
                      │                      │   _column    │ → col_db        │
                      │                      └──────┬───────┘                 │
                      └─────────────────────────────┼─────────────────────────┘
                                                    │
                                       ┌────────────▼────────────┐
                                       │      detection.py       │
                                       │ Fisher F-test + ACF     │
                                       │ → score (temps réel     │
                                       │    uniquement, pas dans │
                                       │    les .npz prod.)      │
                                       └────────────┬────────────┘
                                                    │
                                       ┌────────────▼───────────┐
                                       │    utils/display.py    │
                                       │ Dashboard matplotlib   │
                                       └────────────────────────┘
```

**Ordre logique dans `main.py`** : après décimation et clutter, un **anneau** de longueur `n_fft` avance par pas `hop`. Sur chaque fenêtre valide (hors warm-up) : **segment** → multiplication par la **fenêtre** → **STFT** (une colonne) → **détection**.

---

## Arborescence utile

```
MicroDopplerDetection/
├── main.py                # CLI streaming — point d’entrée principal
├── configs/
│   └── config.yaml        # source de vérité pipeline + fenêtrage + STFT + détection
├── logs/                  # radar_<timestamp>.log si activé dans la config
├── pipeline/
│   ├── emission.py
│   ├── acquisition.py
│   ├── decimation.py
│   ├── clutter.py
│   ├── windowing.py       # Hann / Hamming / rectangular …
│   ├── spectrogramme.py   # colonne STFT (consomme la fenêtre)
│   └── detection.py
├── utils/
│   ├── display.py         # Dashboard temps réel ou relecture sans panneau score
│   ├── record_acquisition.py
│   ├── record_visualization.py
│   ├── auto_record.py
│   └── repo_paths.py      # défaut AICalibration/data
└── legacy/                # batch hors-ligne — non utilisé par main.py
    ├── acquisition_batch.py
    ├── decimation_batch.py    # decimate_iq (API offline)
    ├── detection_batch.py     # detect_presence (offline)
    └── spectrogramme_batch.py
```

Les enregistrements **supervisés** (`record_acquisition`) sont par défaut sous le dépôt voisin **`../AICalibration/data/<train|test|val>/<index>.*`**. Voir **`utils/repo_paths.default_recording_data_root()`**.

---

## Installation

### Prérequis système

```bash
sudo apt install libiio-dev libiio-utils
iio_info -s
```

### Dépendances Python

```bash
pip install numpy scipy matplotlib pyyaml pyadi-iio
```

| Paquet       | Rôle                                              |
|--------------|---------------------------------------------------|
| `numpy`      | IQ, tableaux                                       |
| `scipy`      | décimation IIR, fenêtres (`signal.windows`), Butterworth clutter |
| `matplotlib` | dashboard                                          |
| `pyyaml`     | configuration                                      |
| `pyadi-iio`  | Pluto                                              |

---

## Utilisation

Depuis la racine du dépôt `IoT_radar` :

```bash
export PYTHONPATH="$(pwd)"
python -m MicroDopplerDetection.main
```

Depuis ce dossier :

```bash
cd MicroDopplerDetection
python main.py
```

### Mode simulation

```bash
python main.py --simulation
```

### Configuration

```bash
python main.py --config configs/mon_setup.yaml
```

`--simulation` force le mode simulé quel que soit le YAML.  
Journal par défaut : `logs/radar_<timestamp>.log` — voir `logging` dans `config.yaml` et `--log-file`.

### Options CLI (`main.py`)

| Option         | Défaut                         | Description               |
|----------------|-------------------------------|---------------------------|
| `--config`     | `configs/config.yaml`         | Fichier YAML              |
| `--simulation` | *(absent)*                    | Forcer la simulation      |
| `--log-file`   | horodaté dans `logs/`         | Fichier de log explicite |

---

## Enregistrement & relecture (hors GUI)

| Script | Rôle |
|--------|------|
| **`utils/record_acquisition.py`** | Une prise : spectrogramme + métadonnées + IQ optionnel sous `AICalibration/data/...` |
| **`utils/auto_record.py`** | Plusieurs prises (`-n`, `--interval`) — mêmes options que ci-dessus |
| **`utils/record_visualization.py`** | Replay d’un `.npz` avec dashboard (sans courbe de score) ; affiche le **label** dans le titre |

Exemples :

```bash
cd MicroDopplerDetection
python utils/record_acquisition.py --subset train --env salle --label 1 --duration 120
python utils/record_visualization.py --subset train --index 5
```

---

## Paramètres (`configs/config.yaml`)

| Section         | Rôle |
|-----------------|------|
| `sdr`           | `f_c`, `f_s`, gains, `uri`, taille de buffer |
| `emission`      | `cw` / `cw_offset`, `f_offset` |
| `decimation`    | `enable`, `D`, `f_max_utile`, anti-repliement |
| `clutter`       | `mode` (`mean`/`iir`/`butterworth`), `alpha`, `butterworth_order`, `butterworth_cutoff` |
| `windowing`     | `mode` : fenêtre appliquée au segment `n_fft` avant FFT |
| `spectrogramme` | `n_fft`, `overlap`, `skip_warmup` |
| `detection`     | `bande_respiration` / `bande_reference` (relatives à la porteuse), `alpha`, fusion `w`, `p_value_decades`, `acf_floor`, `acf_good`, `acf_buffer_seconds` |
| `affichage`     | `N_historique`, `seuil_proba`, `plein_ecran` |
| `bilan_liaison` | scénarios `optimiste`/`pessimiste`, `B_eff_hz` → portée affichée |
| `simulation`    | respiration synthétique, SNR, clutter |

Détails dans les commentaires du YAML.

---

## Licence

Projet interne — usage académique et recherche.
