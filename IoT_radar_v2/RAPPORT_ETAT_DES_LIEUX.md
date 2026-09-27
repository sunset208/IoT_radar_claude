# État des lieux — radar de détection de respiration (v1) et refonte v2

*27/09/2026 — analyse du dépôt `IoT_radar-main`, refonte dans `IoT_radar_v2`.
Aucun matériel n'était visible et rien n'a été entraîné : tous les chiffres
ci-dessous viennent de **simulations** (un simulateur physique plus réaliste
que celui de la v1, écrit pour l'occasion). Ils valident la chaîne, pas les
performances réelles.*

---

## 1. Le projet v1 en deux phrases

Un PlutoSDR émet une onde continue (3.5 GHz, décalée de 500 Hz en bande de
base). La réception est décimée à 2 kHz, filtrée (passe-haut 0.05 Hz), puis une
FFT de 8192 points toutes les 0.41 s alimente un score qui fusionne un test de
Fisher (puissance dans la bande respiratoire vs une bande de référence) et une
autocorrélation de la phase. Le volet IA entraîne un autoencodeur Conv2D avec
une tête de classification sur des spectrogrammes en dB de 8192 × 32.

## 2. Problèmes trouvés dans la v1 (du plus grave au moins grave)

| # | Problème | Conséquence |
|---|---|---|
| 1 | **Simulation irréaliste** : le clutter statique est placé à 0 Hz. En réalité la fuite TX→RX et les murs renvoient la tonalité émise, donc à **f_offset**. | La simulation masquait le problème n°2. Mesuré : avec un clutter placé à f_offset, **le test de Fisher donne p < 0.01 sur 100 % des trames en salle vide**. Le score reste bloqué à 0.5 et c'est l'ACF seule qui décide. |
| 2 | **Test de Fisher mal posé**. Le passe-haut 0.05 Hz retire le DC, pas la raie à 500 Hz. Le lobe principal de la fenêtre de Hann (±2 bins = ±0.5 Hz) tombe **dans** la bande respiratoire 0.1–0.8 Hz autour de la porteuse. Les bins fenêtrés sont corrélés, donc l'hypothèse χ²/F est fausse. | La composante spectrale du score ne porte aucune information. |
| 3 | **Résolution** : n_fft = 8192 à 2 kHz donne une fenêtre de 4.1 s et δf = 0.24 Hz. La bande 0.1–0.8 Hz ne couvre que 3 bins et une fenêtre ne contient même pas un cycle à 6 resp/min. | Impossible d'estimer un rythme. Le signal est suréchantillonné ×1000, ce qui coûte du CPU pour rien. |
| 4 | **Buffer TX cyclique discontinu** : 16384 échantillons à 2 MS/s pour 500 Hz, soit 4.096 périodes. | Saut de phase toutes les 8.2 ms, raies parasites tous les 122 Hz. |
| 5 | **Temps réel fragile** : `rx()` et le traitement sont dans le même générateur, à côté de l'interface matplotlib (GIL). Aucune détection de perte d'échantillons. | Une perte d'échantillons fait sauter la phase, ce qui ressemble à un mouvement. Ces pertes passent inaperçues et ne sont pas marquées dans les enregistrements. |
| 6 | Les boucles de *tracking* DC/quadrature de l'AD9363 restent actives. | Elles corrigent le signal près du DC. |
| 7 | **Format d'enregistrement** : spectrogramme en dB (la phase est perdue), 8192 × N en float64 (~19 Mo pour 2 min). Sur ces 8192 bins, ~7 servent (autour de 500 Hz). | Pour l'IA, 99.9 % de l'entrée est du bruit hors bande. |
| 8 | **Volet IA** : split aléatoire **par fenêtre**, donc des fenêtres voisines d'un même enregistrement tombent en train et en val (fuite). Val = 27 fenêtres. 80 % de la classe 1 (431/538). Un seul environnement. | Les 92.6 % de précision en validation sont à comparer aux 80 % qu'on obtient en répondant toujours « respiration ». Ils sont probablement optimistes à cause de la fuite. L'autoencodeur passe l'essentiel de sa capacité à reconstruire du bruit hors bande. |
| 9 | Divers : le README dit 2.4 GHz alors que la config est à 3.5 GHz. Le mode clutter `mean` est une boucle Python par échantillon. La « portée » affichée vient d'une équation radar à B_eff = 50 Hz arbitraire. | Doc incohérente, calcul inutilement lent, chiffre de portée sans valeur. |

Point positif : le code v1 est propre et bien commenté. Il est aussi
récupérable : les fichiers `.iq` v1 contiennent l'écho intact, et
`radar convert-v1` les convertit au format v2 (testé).

## 3. Ce que fait la v2

| Sujet | v1 | v2 |
|---|---|---|
| Porteuse décalée | 500 Hz, buffer TX discontinu | 10 kHz, **nombre entier de périodes** (vérifié par config) |
| Décimation | IIR Chebyshev en forme b/a, 2 kHz | NCO → **CIC dont les zéros tombent sur le DC récepteur et l'image TX** (> 80 dB, testé) → IIR elliptiques en SOS → **50 Hz** |
| Coût CPU front-end | élevé | **~5 %** d'un cœur à 1 MS/s |
| Acquisition | même thread que le traitement | **thread RX dédié**, file, détection des pertes, reset propre, marquage dans les enregistrements |
| Clutter statique | passe-haut sur le DC (sans effet sur l'écho à f_offset) | l'écho est ramené à 0 Hz puis retiré par tendance et passe-haut 0.06 Hz. Base **radiale / tangentielle** au clutter |
| Détecteur | Fisher (invalide) × ACF | **2 voies sommées, chacune normalisée par un plancher coloré local**. Robuste à la dérive de phase LO : 99e percentile en salle vide < 9 dB même avec 0.03 rad de dérive (testé) |
| Fenêtre | 4.1 s | **20 s** glissante, décision toutes les 0.5 s, hystérésis |
| Sorties | score | **VIDE / MOUVEMENT / RESPIRATION**, rythme (resp/min), **amplitude en mm** (arc-tangente + ajustement de cercle géométrique, ±20 %), cœur (expérimental), alerte « rythme trop régulier » |
| Enregistrement | spectrogramme dB + IQ 2 kHz filtré | **slow-time complexe 50 Hz avant tout filtrage** (400 o/s), pertes, annotations horodatées, config complète, IQ brut en option |
| Calibration | — | `radar calibrate` : seuils pour un taux de fausse alarme visé, calculés sur des enregistrements « vide » ; régression logistique validée par sessions |
| Évaluation | — | `radar evaluate` : % de temps en détection, latence, erreur de rythme, fausses alarmes, sur fichiers ou scénarios simulés |
| Diagnostic | — | `radar check` : niveaux ADC, fuite, DC, image, plancher, **bruit de phase équivalent en µm**, débit réel |
| Interface | matplotlib + lanceur pygame | web locale (PC, téléphone ou tablette) : état, forme d'onde en mm, spectre, constellation IQ, spectrogramme, historique, santé du système, enregistrement étiqueté |
| Simulation | clutter à 0 Hz, sinus parfait | fuite, échos et DC récepteur réalistes, dérive LO, respiration asymétrique à rythme variable, cœur, mouvements, marcheur, ventilateur, apnées, quantification ADC |
| Tests | smoke test du modèle | 15 tests pytest (front-end, détection, fausses alarmes, machine à états, format, conversion v1) |

## 4. Résultats (simulation uniquement)

* `radar evaluate --sim 6` : 42 sessions de 2 min, paramètres tirés au hasard,
  y compris des cas extrêmes (SNR −5 dB, dérive LO forte).
  * Respiration présente : **83 % des sessions détectées**, latence médiane
    **21 s** (fenêtre de 20 s + confirmation), erreur de rythme **~1 resp/min**.
    Les ratés sont les cas à SNR ≤ −4 dB ou à dérive LO 40 dB au-dessus de la cible.
  * Sans respiration (vide, marcheur, mouvements) : **0 % de fausse alarme**.
  * **Ventilateur oscillant** : détecté comme respiration à fort SNR. C'est une
    ambiguïté de fond pour un radar CW mono-antenne : un mouvement lent et
    périodique ressemble à une respiration. Voir §5.
* Session « normale » (SNR 15–25 dB) : 99 % du temps en RESPIRATION, 0 % de
  fausse alarme sur la salle vide.

**La grande inconnue, c'est le matériel.** Le facteur limitant réel sera
probablement la **stabilité de phase entre TX et RX du Pluto multipliée par la
fuite TX→RX**, et non le bruit thermique. La simulation montre qu'une dérive de
quelques mrad sur une fuite forte suffit à noyer une respiration faible si on
ne la traite pas (d'où les 2 voies et le plancher coloré). Cette dérive se
mesure en une minute, **même antennes débranchées** :
`radar check --stability 60` donne un « bruit équivalent en µm » dans la bande
0.1–0.8 Hz, à comparer au 1–10 mm d'une respiration.

## 5. Limites connues (v2)

* **Mouvements périodiques mécaniques** (ventilateur oscillant, rideau) :
  confondus avec une respiration. Pistes : amplitude en mm (un ventilateur
  bouge de plusieurs cm), forme d'onde asymétrique, battement cardiaque, IA.
  En contexte de secours, mieux vaut une fausse alerte qu'un raté : l'interface
  **avertit** au lieu de rejeter.
* **Un radar CW ne mesure pas la distance** : on ne localise pas, et deux
  personnes se mélangent. Il faudrait du FMCW/SFCW (le Pluto est limité à
  ~20 MHz de bande, soit ~7.5 m de résolution) ou plusieurs antennes (Pluto
  rév. C, 2T2R) pour l'angle.
* Le rythme cardiaque est expérimental (0.1–0.5 mm à la paroi thoracique).
* Tout est validé en simulation. Les seuils par défaut devront être
  **recalibrés sur de vraies salles vides**.

## 6. Comment faire quelque chose de bien : plan proposé

### 6.1 Matériel (en premier)

1. **Fréquence : quitter 3.5 GHz.** La bande 3.4–3.8 GHz est attribuée aux
   opérateurs 5G. Y émettre hors cage de Faraday n'est pas autorisé (à vérifier
   auprès de l'encadrant ou de l'ARCEP), et les stations 5G voisines peuvent
   polluer le récepteur. Utiliser la **bande ISM 2.4 GHz** (2400–2483.5 MHz,
   limites de puissance de la réglementation SRD), qui pénètre mieux les
   décombres, ou 5.8 GHz, plus sensible (λ plus court) mais qui pénètre moins.
   C'est une ligne dans la config : `sdr.f_c`.
2. **Antennes** : deux antennes directives (patch ou Yagi, +8 à 12 dBi),
   séparées ou blindées pour **réduire la fuite TX→RX**. La fuite fixe la
   dynamique et, multipliée par la dérive LO, le plancher de bruit.
3. `radar check` puis `radar check --stability 60`, d'abord antennes
   débranchées, puis branchées en salle vide. Régler `rx_gain_db` pour une
   crête ADC entre −20 et −6 dBFS.
4. Si la dérive est forte : laisser chauffer le Pluto 10 min avant les mesures
   (la dérive thermique décroît), éviter les courants d'air sur la carte, et
   envisager une référence d'horloge externe plus stable (TCXO, connecteur
   CLK_IN sur les Pluto rév. C/D).

### 6.2 Protocole de données (le vrai goulot d'étranglement)

* **Sessions** (`--tag`) : une session = un lieu, un montage et une date. Les
  splits se font **par session**, jamais par fenêtre.
* Au moins **40 % du temps en salle vide**, dans chaque environnement : c'est
  ce qui fixe le taux de fausse alarme.
* Positifs variés : 1, 2 et 4 m ; de face, de dos, de côté ; derrière du bois,
  du placo, des briques, des parpaings ; respiration normale, lente, rapide,
  apnées (utiliser le bouton « apnée » de l'interface).
* Confondeurs, étiquetés 0 : ventilateur, rideau, quelqu'un qui marche derrière
  le radar, porte, machine.
* **Vérité terrain du rythme** : ceinture respiratoire ou, gratuitement, un
  téléphone posé sur le thorax (accéléromètre, appli *phyphox*), démarré en
  même temps.
* Ordre de grandeur visé : **≥ 30 sessions, ≥ 5 sujets, ≥ 10 h**.

### 6.3 Détecteur classique

`radar calibrate data/recordings --pfa 0.01`, puis `radar evaluate` sur des
sessions **non utilisées** pour la calibration. Chiffres à mettre dans le
rapport S7 : probabilité de détection par session, **fausses alarmes par
heure**, latence, erreur de rythme par rapport à la ceinture, courbe ROC.

### 6.4 IA (voir `ai/README.md`)

Pré-entraînement sur le simulateur, puis affinage sur des données réelles et
**semi-synthétiques** (vrai fond vide + cible simulée), puis comparaison au
détecteur classique **sur les mêmes sessions tenues à l'écart**. On n'adopte
l'IA que si elle fait mieux. La 5090 est largement suffisante : le calcul n'est
pas le problème, **les données** le sont.

## 7. Ce qui n'a pas pu être fait

* Aucun test sur le vrai matériel : aucun PlutoSDR n'était visible (seul un
  adaptateur série FTDI FT230X sur COM3, silencieux en écoute passive).
* Aucun entraînement, comme demandé. Les scripts ont été vérifiés par un unique
  passage avant (`--check`), sans aucune mise à jour de poids.
* Les données v1 (`AICalibration/data`) ne sont pas dans le dépôt (ignorées par
  Git). Si les `.iq` existent ailleurs : `radar convert-v1 <dossier>`.
* L'ancien checkpoint `best.pt` (autoencodeur v1) est incompatible avec la
  nouvelle entrée et n'a pas été repris.

---

## 8. Premiers essais sur le vrai Pluto (27/09/2026, chambre anéchoïque encombrée)

* **Connexion sans driver** : l'interface réseau du Pluto est passée de RNDIS à
  **NCM** (`usb_ethernet_mode = ncm` dans `config.txt` du Pluto), mode que
  Windows 11 gère nativement. Sauvegarde de l'ancien fichier :
  `tools/pluto_config_backup_rndis.txt`. Pluto Rev.C, firmware v0.39, débit
  1 MS/s tenu sans perte.
* **2.45 GHz est inutilisable ici** : le Wi-Fi et le Bluetooth des appareils de
  la pièce envoient des pics jusqu'à −8 dBFS, émission coupée. Le balayage
  400–3700 MHz montre une bande calme partout sauf à 800 MHz (réseau mobile)
  et 2.45 GHz.
* **Les antennes d'origine** résonnent vers 0.87–0.92 GHz et 1.7–2.0 GHz
  (couplage TX→RX 25 dB plus fort qu'à 2.45 GHz). Choix retenu :
  **1.8 GHz, TX 0 dB (≈5 mW), RX 20 dB** (`configs/lab.yaml`).
* **Ce qui limite, c'est le bruit de phase de la fuite TX→RX**, comme prévu au
  §4. Les antennes sont à 2 cm l'une de l'autre, donc fuite à −9 dBFS. Une
  barrette de RAM glissée entre les deux la ramène à −19 dBFS.
* **Mouvements du sujet** : très nettement visibles (20 à 37 dB au-dessus du bruit).
* **Respiration cadencée à 15/min, sujet à 50 cm** : pic à **0.244 Hz
  (14.6/min)**, +25 dB au-dessus du bruit sur la voie radiale.
* **Défaut corrigé grâce à ces mesures** : le plancher coloré (médiane
  ±0.15 Hz) était plus étroit que la raie sur 20 s, donc la raie gonflait son
  propre plancher et le SNR mesuré plafonnait à ~6 dB. Il est remplacé par un
  plancher **CFAR en anneau** (garde 0.1 Hz). Réglages associés : seuil de
  10 dB, `on_count` passé à 5.
* **Résultat après correction** : RESPIRATION **80 % du temps**, latence 35 s,
  rythme médian 15.1/min. Cet enregistrement sert maintenant de test de
  non-régression (`tests/data`).
* **Pas encore détectée** : la respiration spontanée, non cadencée, d'un sujet
  qui bouge un peu (tests 2 et 3).
* **Suite** : enregistrer une salle vide pour calibrer les seuils, puis
  écarter physiquement les antennes (rallonges SMA ≥ 30 cm).
* **Salle vide (sujet sorti, 75 s)** : **0 % de fausse alarme, 0 % de
  MOUVEMENT**, SNR de 4 à 7 dB (sous le seuil). La voie tangentielle est 16 à
  26 dB plus bruitée que la radiale : signature de la dérive LO.
* **Calibration** (`calibration/lab_1800.json`) : seuil **9.8 dB** (proche du
  défaut de 10 dB), AUC du SNR seul **0.99** entre vide et respiration.
* **Bruit de phase de la fuite : 0.55 mrad rms** dans 0.1–0.8 Hz. Pour une
  cible ~40 dB sous le clutter, cela équivaut à **~0.7 mm** de déplacement :
  une respiration ample est détectable, une respiration calme est limite.
  D'où l'intérêt d'éloigner les antennes l'une de l'autre.
