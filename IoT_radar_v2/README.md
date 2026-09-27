# IoT Radar v2 — détection de respiration par radar CW (PlutoSDR)

Réécriture complète du projet `IoT_radar-main` (qui reste intact à côté pour comparaison).
Pourquoi et quoi : voir **[RAPPORT_ETAT_DES_LIEUX.md](RAPPORT_ETAT_DES_LIEUX.md)**.
Partie IA : **[ai/README.md](ai/README.md)**.

## Installation (déjà faite sur ce PC, tout est dans `F:\Claude\ProjetS7`)

| Élément | Emplacement | Rôle |
|---|---|---|
| venv Python 3.13 | `ProjetS7\.venv` | numpy, scipy, pyadi-iio, fastapi, torch CUDA 13… |
| libiio 0.25 (Windows) | `ProjetS7\tools\libiio` | bibliothèque du Pluto, **non installée dans le système** |
| cache pip | `ProjetS7\.cache\pip` | |

Rien n'a été installé ailleurs sur le PC.  Seul prérequis externe pour le vrai
matériel : le **driver USB du Pluto** (voir « Brancher le Pluto »).

Tout se lance avec `radar.bat` (ou `radar.ps1`, ou `python radar_cli.py`) :

```bat
radar.bat run --source sim                 :: tableau de bord, simulation rapide
radar.bat run --source sim-rf              :: simulation de l'IQ brut Pluto + vrai front-end
radar.bat run                              :: PlutoSDR (ip:192.168.2.1 par défaut)
radar.bat run --source replay --file data\recordings\xxx.npz
radar.bat check                            :: diagnostic matériel
radar.bat check --stability 60             :: + mesure du bruit de phase (60 s)
radar.bat record --label 0 --duration 120 --tag salleB_vide
radar.bat evaluate data\recordings         :: métriques du détecteur sur des fichiers
radar.bat evaluate --sim 5                 :: … sur des scénarios simulés
radar.bat calibrate data\recordings        :: seuils depuis des enregistrements étiquetés
radar.bat convert-v1 ..\IoT_radar-main\IoT_radar-main\AICalibration\data
```

Le tableau de bord s'ouvre sur <http://127.0.0.1:8050> (ajouter `--host 0.0.0.0`
pour le consulter depuis un téléphone/tablette sur le même réseau).

Tests : `..\.venv\Scripts\python.exe -m pytest tests -q` (15 tests, ~6 s).

## Chaîne de traitement

```
Pluto RX 1 MS/s ─► NCO −10 kHz ─► CIC² ÷100 ─► IIR ellip. ÷10÷5÷4 ─► slow-time 50 Hz (complexe)
 (TX : tonalité à +10 kHz,          zéros exactement sur          │     = donnée enregistrée
  nb entier de périodes)            DC récepteur / image TX       ▼
                                                  fenêtre 20 s glissante (pas 0.5 s)
                                                                  │
            ┌──────── voie radiale / tangentielle au clutter statique (dérive LO → tangentielle)
            │         périodogramme normalisé par un plancher coloré local, somme des 2 voies
            ▼
   features : SNR du pic 0.1–0.8 Hz, concentration spectrale, ACF, bouffées > 3 Hz, énergie
            │         démodulation arc-tangente + ajustement de cercle → déplacement en mm
            ▼
   détecteur (seuils ou régression logistique calibrés) + hystérésis
            → VIDE / MOUVEMENT / RESPIRATION + rythme (resp/min) + amplitude (mm) + cœur (exp.)
```

## Arborescence

```
radar/
  config.py            YAML → dataclasses validées (contraintes f_s / f_offset / buffers)
  scene.py             modèle physique : fuite, échos statiques, dérive LO, respiration
                       asymétrique à rythme variable, cœur, mouvements, marcheur, ventilateur, apnées
  dsp/frontend.py      NCO + CIC + IIR à état (streaming exact, testé bloc = flux)
  dsp/vitals.py        démodulation, spectre 2 voies, features, amplitude, cœur
  dsp/detector.py      états + hystérésis + calibration / régression logistique
  sources/pluto.py     thread RX dédié, détection de pertes, TX sans discontinuité
  sources/simulation.py  simulation RF brute (1 MS/s) ou slow-time directe
  sources/replay.py    relecture d'enregistrements
  pipeline.py          moteur temps réel + enregistrement
  recorder.py          format v2 (.npz : slow-time complexe + métadonnées + annotations)
  offline.py           évaluation, calibration, métriques (AUC…)
  hwcheck.py           diagnostic Pluto (niveaux, fuite, stabilité de phase, débit)
  legacy.py            conversion des .iq v1
  web/                 tableau de bord (FastAPI + WebSocket, canvas sans dépendance)
ai/                    IA sur slow-time (voir ai/README.md)
configs/default.yaml   tous les paramètres, commentés
tests/                 pytest
```

## Brancher le Pluto (Windows)

À la date du 27/09/2026, **aucun PlutoSDR n'était visible** sur ce PC : ni
périphérique USB `VID_0456` (Analog Devices), ni interface réseau 192.168.2.1.
Seul un adaptateur série **FTDI FT230X (COM3)** venait d'être branché (silencieux
en écoute passive).  Si c'est ton radar, ce n'est pas un Pluto ; sinon :

1. Brancher le micro-USB marqué **USB** (données), pas celui marqué *Power* ;
   câble data (pas « charge seule »).
2. Installer le driver ADI « PlutoSDR-M2k-USB-Drivers » (seule installation
   système nécessaire) → carte réseau RNDIS `192.168.2.1` + accès `usb:`.
3. `radar.bat check` doit afficher le modèle, le firmware et les niveaux.
4. Antennes débranchées, `radar.bat check --stability 60` mesure déjà une
   donnée essentielle : la stabilité de phase de la fuite TX→RX (voir rapport §4).
