# IoT Radar v2 — détection de respiration par radar (PlutoSDR)

Réécriture complète du projet `IoT_radar-main` (qui reste intact à côté pour comparaison).
Pourquoi et quoi : voir **[RAPPORT_ETAT_DES_LIEUX.md](RAPPORT_ETAT_DES_LIEUX.md)** (§8 : premiers
essais sur le vrai Pluto ; §9 : réactivité, SFCW, 5.8 GHz, 60 GHz, antennes).
Partie IA : **[ai/README.md](ai/README.md)**.

## Installation (déjà faite sur le PC du labo, tout est dans `F:\Claude\ProjetS7`)

| Élément | Emplacement | Rôle |
|---|---|---|
| venv Python 3.13 | `ProjetS7\.venv` | numpy, scipy, pyadi-iio, fastapi, pyserial, torch CUDA… |
| libiio 0.25 (Windows) | `ProjetS7\tools\libiio` | bibliothèque du Pluto (+ `iio_attr.exe`, `iio_info.exe`), **non installée dans le système** |

Environnement neuf : `pip install -r requirements.txt` (et `pip install pyserial` si le venv
existant ne l'a pas encore : module 60 GHz).

Tout se lance avec `radar.bat` (ou `radar.ps1`, ou `python radar_cli.py`) depuis `IoT_radar_v2` :

```bat
:: --- CW (le mode habituel) -------------------------------------------------
radar.bat --config configs\lab.yaml run --calibration calibration\lab_1800.json
radar.bat run --source sim                 :: tableau de bord, simulation rapide
radar.bat run --source sim-rf              :: simulation de l'IQ brut Pluto + vrai front-end
radar.bat run --source replay --file data\recordings\xxx.npz
radar.bat run ... --mr60 COM5              :: + module radar 60 GHz affiché à côté
radar.bat check                            :: diagnostic matériel (niveaux, plage LO, débit)
radar.bat check --stability 60             :: + bruit de phase de la fuite (60 s)
radar.bat scan --start 0.4e9 --stop 3.7e9  :: bruit ambiant par fréquence (--tx : + fuite TX→RX)
radar.bat record --label 0 --duration 120 --tag salleB_vide
radar.bat evaluate data\recordings         :: métriques (1re alerte, latence, fausses alarmes…)
radar.bat evaluate --sim 5                 :: … sur des scénarios simulés
radar.bat calibrate data\recordings --out calibration\x.json
radar.bat convert-v1 ..\IoT_radar-main\IoT_radar-main\AICalibration\data

:: --- séances guidées (données étiquetées propres) -------------------------
radar.bat --config configs\lab.yaml protocole --plan vide_long     :: 10 min de salle vide
radar.bat --config configs\lab.yaml protocole --plan labo          :: ~20 min, positions variées
radar.bat --config configs\lab.yaml protocole --plan planche [--mode sfcw] [--mr60 COM5]

:: --- SFCW : fréquence balayée, amplitude seule ----------------------------
radar.bat --config configs\lab.yaml sfcw bench        :: ce que le Pluto permet (à faire d'abord)
radar.bat --config configs\lab.yaml sfcw run          :: tableau de bord distance × temps
radar.bat sfcw run --source sim --scenario breathing_walker --range 1.0   :: démo sans Pluto
radar.bat sfcw run --source fake                      :: vrai code Pluto sur faux Pluto
radar.bat --config configs\lab.yaml sfcw record --label 1 --duration 60 --tag essai
radar.bat sfcw analyze data\sfcw
```

Le tableau de bord s'ouvre sur <http://127.0.0.1:8050> (`--host 0.0.0.0` pour un
téléphone ou une tablette sur le même réseau).

Tests : `..\.venv\Scripts\python.exe -m pytest tests -q` (37 tests, ~1 min, sans matériel).

## Chaîne de traitement (CW)

```
Pluto RX 1 MS/s ─► NCO −10 kHz ─► CIC² ÷100 ─► IIR ellip. ÷10÷5÷4 ─► slow-time 50 Hz (complexe)
 (TX : tonalité à +10 kHz,          zéros exactement sur          │     = donnée enregistrée
  nb entier de périodes)            DC récepteur / image TX       ▼
             ┌────────────────────────────────────────────────────┴───────────────────┐
             ▼ toutes les 0.1 s (filtres causaux)                                      ▼ toutes les 0.5 s
   indices rapides : présence 0.12–1 Hz / 4 s,                 fenêtres 8, 12 et 20 s : voies radiale /
   activité 1.5–15 Hz / 1 s, normalisés par le                 tangentielle au clutter, périodogramme / plancher
   bruit attendu en salle vide (thermique +                    CFAR en anneau, SNR du pic, périodicité,
   bruit multiplicatif de la fuite)                            bouffées > 3 Hz, arc-tangente → mm
             └──────────────────────────► détecteur (hystérésis à 2 cadences) ◄─────────┘
         → VIDE / PRÉSENCE (signe de vie, ~5 s) / MOUVEMENT / RESPIRATION (+ rythme, mm, cœur exp.)
```

## Arborescence

```
radar/
  config.py            YAML → dataclasses validées (f_s/f_offset, plages de LO ad9363/ad9364, SFCW)
  scene.py             modèle physique CW : fuite, échos, dérive LO, respiration, cœur, mouvements…
  dsp/frontend.py      NCO + CIC + IIR à état (streaming exact, testé bloc = flux)
  dsp/fast.py          indices rapides 10 Hz (présence, activité) + forme d'onde causale
  dsp/vitals.py        analyse d'une fenêtre : spectre 2 voies, features, amplitude, cœur
  dsp/detector.py      états + hystérésis à deux cadences, multi-échelle, calibration
  pipeline.py          Processor (logique pure, partagée temps réel / hors ligne) + Engine (thread)
  sources/             Pluto (thread RX, pertes, plage LO), simulation RF / slow-time, relecture
  sfcw/                mode SFCW : dsp (distance × temps), sources (Pluto, sim, relecture),
                       scene (simulateur + faux Pluto), engine, bench
  mmwave.py            lecteur série des modules 60 GHz (MR60BHA2 / LD6002, protocole TinyFrame)
  protocol.py          séances d'enregistrement guidées
  recorder.py          format v2 (.npz : slow-time complexe + métadonnées + annotations)
  offline.py           évaluation (rejeu exact), calibration par échelle, AUC…
  hwcheck.py           diagnostic Pluto, balayage de fréquences (radar scan)
  legacy.py            conversion des .iq v1
  web/                 tableau de bord (FastAPI + WebSocket, canvas sans dépendance)
protocols/             plans de séance (labo, planche, vide_long)
firmware/              croquis Arduino de relais pour le kit XIAO MR60BHA2
configs/               default.yaml (tout, commenté), lab.yaml (1.8 GHz), lab_5800.yaml (5.8 GHz)
calibration/           lab_1800.json (27/09 : seuils par échelle + bruit de la fuite)
ai/                    IA sur slow-time (voir ai/README.md)
tests/                 pytest (CW, SFCW, module 60 GHz)
```

## Le Pluto du labo

* **Pluto Rev.C (Z7010-AD9363A), firmware v0.39**, URI `ip:192.168.2.1`.
* Windows 11 sans driver ADI : l'interface USB du Pluto a été passée de RNDIS à
  **NCM** (`usb_ethernet_mode = ncm` dans `config.txt` du lecteur « PlutoSDR »),
  que Windows gère nativement.  Ancien fichier : `..\tools\pluto_config_backup_rndis.txt`.
* 1 MS/s tenu sans perte.  2.45 GHz inutilisable dans la chambre (Wi-Fi/BT) ;
  config retenue : **1.8 GHz, TX 0 dB (≈ 5 mW), RX 20 dB** (`configs/lab.yaml`).

## Passer à 5.8 GHz (déverrouillage AD9364)

L'AD9363 d'origine est limitée à 325 MHz–3.8 GHz (20 MHz de bande).  Le même
silicium se configure en AD9364 (70 MHz–6 GHz, 56 MHz).  **À faire par vous**
(mot de passe root par défaut du Pluto : voir la documentation ADI) :

```bat
ssh root@192.168.2.1
fw_setenv attr_name compatible
fw_setenv attr_val ad9364
reboot
```

Vérification après redémarrage (~30 s) :

```bat
..\tools\libiio\Windows-VS-2022-x64\iio_attr.exe -u ip:192.168.2.1 -c ad9361-phy altvoltage0 frequency_available
:: attendu : [70000000 1 6000000000]
radar.bat --config configs\lab_5800.yaml check        :: affiche « plage LO 70–6000 MHz (AD9364 déverrouillé) »
radar.bat --config configs\lab_5800.yaml scan --start 5.65e9 --stop 5.925e9 --step 5e6
radar.bat --config configs\lab_5800.yaml scan --start 5.65e9 --stop 5.925e9 --step 5e6 --tx
radar.bat --config configs\lab_5800.yaml run
```

* Le mode 2r2t (`fw_setenv attr_val ad9361` + `fw_setenv mode 2r2t`) est réputé
  capricieux sur les Rev.C et n'est pas nécessaire.
* **Avec les antennes d'origine, 5.8 GHz sera mauvais** (résonances vers 0.9 et
  1.7–2.0 GHz) : il faut des antennes adaptées (rapport §9.6 : log-périodiques
  LP0965 0.85–6.5 GHz, ou patchs 5.8 GHz).
* Refaire un enregistrement « vide » à 5.8 GHz puis `radar calibrate` (le bruit
  de phase de la fuite croît avec la fréquence).
* Retour en arrière : `fw_setenv attr_val ad9363` puis `reboot`.

## Mode SFCW (fréquence balayée, amplitude seule)

La phase TX/RX de l'AD936x est aléatoire après chaque saut de LO, mais
l'amplitude |S_k| ne l'est pas : la fuite TX→RX sert de référence
interférométrique, et une FFT sur les pas de fréquence donne un **profil en
distance** (15 cm de résolution sur 1 GHz, portée c/(4Δf) = 3.75 m pour Δf =
20 MHz).  Une victime immobile est ainsi séparée d'un sauveteur qui marche.
Détails : rapport §9.3.

1. `sfcw bench` : latence d'écriture des LO sur le lien réseau, lectures par
   pas (buffers périmés), cadence atteignable, répétabilité d'amplitude →
   ajuster `sfcw.rx_buffer_size`, `kernel_buffers`, `n_steps` dans la config.
2. `sfcw run` : carte distance × temps, profil, SNR respiratoire par case.
   Bouton « Figer le fond » en salle vide pour voir ce qui change ensuite.

## Module radar 60 GHz (Seeed XIAO MR60BHA2 ou HLK-LD6002)

* **MR60BHA2** (recommandé, 24.90 $) : téléverser `firmware/mr60bha2_passthrough`
  avec l'Arduino IDE (carte XIAO_ESP32C6, « USB CDC On Boot : Enabled »), puis
  `radar.bat run --mr60 COMx`.  Sans ce croquis, le firmware ESPHome d'origine
  est lu en mode texte (au mieux).
* **LD6002** : adaptateur USB-UART rapide (le FTDI FT230X convient) +
  alimentation 3.3 V / 600 mA séparée : `--mr60 COMx --mr60-baud 1382400`.
* Pendant un enregistrement Pluto, ses mesures sont sauvées dans
  `<fichier>.mr60.json` : rythme de référence pour `radar evaluate` et l'IA.
* Démo sans module : `--mr60 sim`.

## Antennes et placement (résumé du rapport §9.6)

Les fouets d'origine vissés à 2 cm l'un de l'autre sont le pire cas (fuite à
−9 dBFS, qui borne le SNR).  Viser : deux antennes **directives** (2 × LP0965,
~25 $ pièce), **20–50 cm d'écart**, orientées vers le thorax, **écran**
(métal à la masse ou absorbant) entre les deux, câbles et antennes fixés.
Contrôler la fuite avec `radar check` (objectif ≲ −30 dBFS), puis recalibrer.
