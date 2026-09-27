Tu reprends un projet en cours. Lis tout ce message avant d'agir, puis lis les fichiers cités. Réponds en français, travaille en autonomie et ne pose des questions que si c'est vraiment nécessaire.

Ce prompt a été écrit par Claude à la fin de la session précédente (session distante, **sans accès au Pluto**). Si tu penses pouvoir prendre de meilleures décisions, fais-le : ne suis pas bêtement ce prompt, réfléchis. En revanche, écoute ce que dit l'utilisateur.

# Contexte

Projet d'école (S7) : un **radar pour détecter la respiration de victimes ensevelies** (décombres), construit autour d'un **ADALM-PlutoSDR**. Le dépôt contient le dossier `ProjetS7` (en local : `F:\Claude\ProjetS7`, Windows 11) :

- `IoT_radar-main/` : version d'origine (v1) de l'équipe. **Ne pas modifier**, référence.
- `IoT_radar_v2/` : le projet actif (réécriture complète).
  - `README.md` : utilisation, toutes les commandes, déverrouillage 5.8 GHz, SFCW, module 60 GHz.
  - `RAPPORT_ETAT_DES_LIEUX.md` : critique de la v1, conception de la v2, **§8** premiers essais matériels, **§9** deuxième séance (tout ce qui est résumé ci-dessous, avec les chiffres).
  - `ai/README.md` : partie IA.
- `tools/libiio/` : libiio 0.25 Windows (contient aussi `iio_attr.exe`, `iio_info.exe`).
- `.venv/` n'est pas dans le dépôt : Python 3.13 avec numpy, scipy, pyyaml, pyadi-iio, pylibiio 0.25, fastapi, uvicorn, pytest, torch cu130. **À ajouter : `..\.venv\Scripts\pip install pyserial`** (module 60 GHz).
- Lancement : `IoT_radar_v2\radar.bat <commande>` (ou `python radar_cli.py <commande>` avec le venv `..\.venv`).
- Tests : `..\.venv\Scripts\python.exe -m pytest tests -q` → **39 tests, doivent passer** (~1 min, sans matériel).

**Cette fois, le Pluto est branché** (USB, `ip:192.168.2.1`) : l'objectif principal est de **valider sur le vrai matériel** ce qui a été codé et testé seulement en simulation.

# Matériel et faits mesurés (27/09/2026, chambre anéchoïque encombrée)

- **Pluto Rev.C (Z7010-AD9363A), firmware v0.39.** Interface USB passée de RNDIS à **NCM** (`usb_ethernet_mode = ncm` dans `config.txt`, Windows 11 natif ; ancien fichier `tools/pluto_config_backup_rndis.txt`). 1 MS/s tenu sans perte.
- **2.45 GHz inutilisable** ici (Wi-Fi/BT jusqu'à −8 dBFS). Balayage 0.4–3.7 GHz calme sauf 800 MHz et 2.45 GHz.
- **Antennes d'origine** (fouets) vissées à **2 cm** l'une de l'autre, résonantes vers 0.87–0.92 et 1.7–2.0 GHz. C'est le pire cas pour la fuite TX→RX.
- **Config retenue : `configs/lab.yaml` = 1.8 GHz, TX 0 dB (≈5 mW, sans danger), RX 20 dB.**
- **Facteur limitant : le bruit multiplicatif de la fuite TX→RX** (fuite à −9 dBFS ; −19 dBFS avec une barrette de RAM glissée entre les antennes).
  - Bruit de **phase** de la fuite : 0.55 mrad rms dans 0.1–0.8 Hz ≈ **−65.3 dBc** dans 0.12–1 Hz.
  - Bruit d'**amplitude** : **−84.7 dBc**, soit 20 dB de mieux. C'est pour ça que la **voie radiale** (≈ amplitude) porte la cible dans tous les enregistrements.
- **Enregistrements** (`IoT_radar_v2/data/recordings/`, 1.8 GHz sauf test1) :

  | fichier | résultat |
  |---|---|
  | `live_test1` | 2.45 GHz, noyé dans le Wi-Fi |
  | `live_test2_1800` | le sujet bouge : PRÉSENCE / MOUVEMENT |
  | `apnee_test_50cm_ram` | apnée puis respiration, non confirmée |
  | `rythme15_50cm_1800` | cadencé 15/min à 50 cm : RESPIRATION 80 %, latence 35 s (copié dans `tests/data`) |
  | `vide_chambre_1800` | salle vide 75 s : 0 % d'alerte |
  | `resp_chambre` | **respiration spontanée ~50 cm détectée** : 89 %, 20.5/min, SNR médian 13.4 dB, latence 27.6 s |

- Calibration : `calibration/lab_1800.json`, régénérée. Elle contient les seuils par échelle et les constantes de bruit de la fuite, mais **ne repose que sur 75 s de salle vide**. Les seuils y sont donc bornés par les valeurs par défaut. **Il faut au moins 10 min de « vide ».**

# Ce qui a été fait à la session précédente (tout est sur `main`)

Détails et chiffres : rapport §9.1–9.11.

1. **Réactivité.** Au SNR actuel (~13 dB sur 20 s), confirmer la *périodicité* en moins de ~25 s est impossible. Ce qui a été ajouté :
   - `radar/dsp/fast.py` : indices causaux à 10 Hz.
     - *Présence* : énergie 0.12–1 Hz sur 4 s, normalisée par le bruit attendu en salle vide = bruit thermique suivi en continu + k·|C|², où k est le bruit de la fuite en dBc.
     - *Activité* : énergie 1.5–15 Hz sur 1 s.
   - Nouvel état **PRESENCE** (« signe de vie »). Sur les enregistrements, la **1re alerte arrive en 4–5 s**, avec 0 % d'alerte en salle vide.
   - Détecteur **multi-échelle** 8/12/20 s, avec des seuils par échelle (18 / 15.5 / 10 dB, calibrables) et une contrainte de rythme stable. Les échelles courtes sont neutralisées pendant un mouvement franc. Au SNR du labo, seule la fenêtre de 20 s confirme ; en simulation, dès 20–25 dB, on passe de 22 s à 14–18 s.
   - `pipeline.Processor` : logique commune au temps réel et à `radar evaluate`, qui rejoue exactement ce que voit l'UI et rapporte la 1re alerte et l'échelle de confirmation.
   - UI : jauges présence/activité, bande « mouvement thoracique » de 30 s qui défile à 60 images/s, SNR par échelle.
2. **Mode SFCW, amplitude seule** (`radar/sfcw/`, commande `radar sfcw run|bench|record|analyze`).
   - Principe : |S_k| est insensible à la phase aléatoire du LO, et la fuite sert de référence. C'est la voie radiale avec en plus de la diversité de fréquence et de la distance.
   - Réglages : `calib_mode = manual` (vérifié dans le driver : en auto, calibration de quadrature TX à chaque saut du LO TX > 100 MHz), LO écrits via les attributs IIO en cache.
   - **Détection des buffers périmés par marquage** : un pas sur deux, le LO RX est décalé de 125 kHz. Plus un test de stabilité entre les deux moitiés du buffer.
   - Traitement : fond MTI, (A−B)/B, FFT sur les pas → profil. **Portée non ambiguë c/(4Δf) = 3.75 m** pour Δf = 20 MHz, résolution 15 cm (1.2–2.18 GHz, 50 pas). Ensuite SNR respiratoire par case, et zone en mouvement.
   - Simulation : localisation à ±3 cm, marcheur rejeté, **victime immobile séparée d'un marcheur**.
   - **Faux Pluto** (`radar/sfcw/scene.py`) : tout le code matériel a tourné dessus. **Jamais testé sur le vrai Pluto.**
3. **5.8 GHz.** `sdr.chip: ad9363|ad9364` validé. `configs/lab_5800.yaml` est prêt. `radar check` affiche la plage de LO du firmware. `radar scan` balaie les fréquences (bruit ambiant, et fuite avec `--tx`).
4. **Module 60 GHz** (`radar/mmwave.py`).
   - Décodeur TinyFrame commun MR60BHA2 / LD6002 (0x0A14 respiration, 0x0A15 cœur, 0x0A16 distance, 0x0F09 présence), plus un repli sur les journaux ESPHome.
   - `radar run --mr60 COMx` (LD6002 : `--mr60-baud 1382400`). Croquis de relais pour le XIAO ESP32C6 : `firmware/mr60bha2_passthrough`.
   - Pendant un enregistrement Pluto, les mesures du module vont dans `<fichier>.mr60.json` : **rythme de référence** pour `evaluate` et l'IA.
   - Recommandation : MR60BHA2 (24,90 $). Il ne traverse pas les décombres.
5. **`radar protocole`** : séances d'enregistrement guidées (`protocols/labo.yaml`, `planche.yaml`, `vide_long.yaml`), en CW ou SFCW, avec `--mr60` en option. Un seul moteur continu ; en fin de séance, manifeste `data/sessions/`, calibration de séance et évaluation.
6. **Phase ou micro-Doppler** (`radar compare`, rapport §9.11).
   - Le détecteur utilise la **phase linéarisée** (projection radiale/tangentielle). À 1.8 GHz, la respiration ne décale le spectre que de ~0.05 Hz (β ≈ 0.2) : le micro-Doppler ne la voit pas.
   - Sur le réel : phase linéarisée AUC **1.00** (92–94 % de détection) contre 0.63–0.73 pour le micro-Doppler, qui répond aux mouvements.
   - En simulation, la bascule se fait entre 5.8 et 24 GHz (à 60 GHz, micro-Doppler et vraie phase gagnent).
   - Le spectrogramme micro-Doppler signé est affiché en direct (±12.5 Hz, cm/s), et ses features sont enregistrées.
7. **IA.** Corrections : le semi-synthétique à la porteuse réelle du fond (il utilisait 3.5 GHz), λ randomisée (1.8/2.45/5.8 GHz), sélection du modèle sur le synthétique en pré-entraînement (avant : AUC NaN → **aucun `best.pt`**), `report.json` en JSON strict. Vérifié sur CPU (250 pas, AUC synthétique 0.66). **Le vrai pré-entraînement n'a pas été lancé** : il se fait sur la RTX 5090 locale.
8. Divers :
   - `radar check --stability` donne les constantes de bruit de la fuite en dBc ; sur la salle vide : −84.2 / −65.0.
   - Corrigés : arrondi JSON qui écrasait les petites valeurs, attribut `hidden` ignoré en CSS, interblocage d'enregistrement.
   - `.gitignore` ajouté. Le cache pip (108 Mo) et les `__pycache__` ont été retirés de l'index : **le `git pull` les supprime de la copie locale**, ce qui est sans conséquence.

# Contraintes

- **N'entre aucun mot de passe à la place de l'utilisateur** : le SSH root du Pluto (déverrouillage AD9364), c'est lui qui le fait.
- **Puissance d'émission ≤ ~5 mW** (TX 0 dB) : l'utilisateur se tient à ~50 cm. 5.8 GHz : 25 mW PIRE max.
- Git : travaille sur une branche, commits clairs, pas de force-push sur main.
- Ne touche pas à `IoT_radar-main/`.
- Avant de lancer une commande qui ouvre le Pluto, vérifie qu'aucune autre ne tourne : un seul programme à la fois sur le Pluto.

# Ce qu'il faut faire maintenant, avec le radar (dans cet ordre)

Pour chaque étape, lance la commande, **lis la sortie et juge** : un résultat qui contredit la simulation est une information à comprendre, pas un bug à masquer. Corrige le code si besoin, avec des tests, et note les résultats réels dans le rapport, dans une nouvelle section §10.

1. **Remise en route** : `git pull`, `pip install pyserial`, `pytest tests -q`, puis `radar.bat --config configs\lab.yaml check --stability 60`.
   - Relever la fuite (dBFS), le débit, et les constantes dBc de la fuite (comparer à −84.7 / −65.3).
2. **CW, nouvelle UI** : `radar.bat --config configs\lab.yaml run --calibration calibration\lab_1800.json`.
   - Vérifier : PRÉSENCE en ~5 s quand l'utilisateur s'assoit ; VIDE quand il sort ; MOUVEMENT quand il bouge ; la bande défilante ; le spectrogramme micro-Doppler (un geste vers le radar doit apparaître en fréquences positives ; si c'est l'inverse, le signe IQ est inversé : à corriger et à documenter) ; `proc_ms` raisonnable.
   - Points sensibles : les seuils de présence (6 dB) et d'activité (8 dB) sont réglés sur 75 s de vide ; les fausses alertes sont possibles si l'environnement a changé.
3. **Données de calibration** : `radar.bat --config configs\lab.yaml protocole --plan vide_long` (10 min, pièce vide), puis `--plan labo` (~20 min).
   - Ensuite `radar calibrate` sur toutes les séances, avec `--out calibration\lab_1800_v2.json`, et `radar evaluate` sur des séances **non utilisées** pour calibrer : fausses alarmes par heure, taux de détection, latences, erreur de rythme si le module 60 GHz est là.
4. **SFCW sur le vrai Pluto**, le plus risqué car jamais testé sur matériel. D'abord `radar.bat --config configs\lab.yaml sfcw bench` :
   - Latence réelle d'écriture des LO sur NCM. Hypothèse de simulation : 1.9 ms. Si c'est ≫ 5 ms, la cadence s'effondre.
   - Lectures par pas : le marquage doit donner ~K+2, sans échecs. Des échecs systématiques indiquent un problème de marquage ou de niveau : regarder les amplitudes de tonalité et le plancher dans `is_fresh`.
   - Cadence de balayage (il faut ≥ 2 balayages/s, idéalement 4–5), `kernel_buffers=1` accepté ou non, répétabilité d'amplitude, plancher dans le domaine distance.
   - Si c'est trop lent : réduire `n_steps` ou augmenter `f_step` (portée c/4Δf), essayer `--fastlock`, et en dernier recours piloter le balayage sur le Pluto lui-même (script sur le Zynq).
   - Ensuite `sfcw run` : l'utilisateur se place à 0.5, 1 puis 2 m, et la distance affichée doit suivre. Tester aussi le bouton « Figer le fond ». Enregistrer avec `sfcw record` ou `protocole --mode sfcw`, puis `sfcw analyze`.
   - Surveiller : les fantômes (produits croisés avec les murs), la saturation ADC à la résonance des antennes (`bench` [1]), et la restauration de `calib_mode` à la fermeture.
5. **Planche de bois** : `radar protocole --plan planche` en CW, puis avec `--mode sfcw`, puis avec `--mr60 COMx` si le module est arrivé. Pertes attendues : négligeables à 1.8 GHz, ~8–10 dB à 60 GHz.
6. **5.8 GHz** (si l'utilisateur déverrouille : `fw_setenv attr_name compatible`, `fw_setenv attr_val ad9364`, `reboot`) :
   - `--config configs\lab_5800.yaml` avec `check`, puis `scan` avec et sans `--tx`, puis `run` et `compare`.
   - Avec les antennes d'origine, s'attendre à un résultat mauvais. Recalibrer à 5.8 GHz : le bruit de phase croît avec la fréquence.
7. **Antennes** : quand les 2 × LP0965 log-périodiques (0.85–6.5 GHz, ~25 $) seront là, les placer à 20–50 cm l'une de l'autre, avec un écran entre les deux, et refaire les étapes 1 à 4.
   - Objectif : une fuite ≲ −30 dBFS au lieu de −9, et des constantes dBc qui baissent. Les échelles courtes du multi-échelle devraient alors commencer à confirmer (latence < 20 s).
8. **IA** sur la 5090, quand il y aura au moins 3 sessions « vide » et 3 sessions « respiration » :
   - `..\.venv\Scripts\python.exe ai\train.py pretrain`, puis `finetune --init ai\runs\pretrain_<date>\best.pt`.
   - Comparer à `classical_baseline_real_val` dans `report.json`. L'IA n'est adoptée que si elle bat le classique sur des sessions jamais vues.

# Limites connues / pistes

- Latence de confirmation de la respiration bornée par le SNR (~13 dB sur 20 s) : c'est la fuite qu'il faut réduire (antennes).
- Un ventilateur oscillant reste confondu avec une respiration ; l'amplitude en mm le sépare en simulation. L'amplitude en mm est peu fiable à 1.8 GHz (pas de vérité terrain) : le module 60 GHz peut en servir.
- Le CW ne donne pas la distance ; le SFCW, oui (à valider sur matériel).
- Piste forte à moyen terme : un **2e RX** (mode 2r2t, capricieux sur Rev.C) comme canal de référence, pour annuler la dérive commune et obtenir l'angle.

Commence par lire `IoT_radar_v2/RAPPORT_ETAT_DES_LIEUX.md` (§8 et §9), `IoT_radar_v2/README.md`, puis `radar/dsp/fast.py`, `radar/dsp/detector.py`, `radar/pipeline.py`, `radar/sfcw/sources.py` et `radar/sfcw/bench.py`. Lance les tests, puis suis l'ordre ci-dessus avec l'utilisateur. Termine par un résumé clair avec les résultats réels et ce qui reste à faire.
