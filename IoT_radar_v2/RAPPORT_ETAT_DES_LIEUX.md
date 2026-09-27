# État des lieux — radar de détection de respiration (v1) et refonte v2

*27/09/2026 — analyse du dépôt `IoT_radar-main`, refonte dans `IoT_radar_v2`.
Les §1–7 datent d'avant les essais matériels ; §8 : premiers essais sur le
vrai Pluto ; **§9 : deuxième séance** (indices rapides, multi-échelle, mode
SFCW, 5.8 GHz, module 60 GHz, antennes, séances guidées).
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

---

## 9. Deuxième séance (27/09/2026, nuit) : réactivité, SFCW, 5.8 GHz, 60 GHz

*Travail fait sans accès au Pluto (session distante) : tout ce qui touche au
matériel est codé, testé en simulation et sur les enregistrements du 27/09,
puis à valider sur place (§9.10).  37 tests pytest, dont 11 pour le SFCW.*

### 9.1 `resp_chambre` : la respiration spontanée est détectée

Sujet assis à ~50 cm, respiration libre (non cadencée), 1.8 GHz, 90 s.

| grandeur | valeur |
|---|---|
| RESPIRATION (régime établi) | **89 %** du temps, latence 27.6 s |
| rythme | **20.5 /min** (écart interquartile 20.1–20.8) |
| SNR du pic (fenêtre 20 s) | médiane 13.4 dB, p10 9.6 dB, max 20.2 dB |
| bruit de phase de la fuite | 0.62 mrad rms (0.55 en salle vide) |
| voie porteuse | **radiale** (la voie tangentielle reste au niveau de la salle vide) |
| mouvements | au début et à la fin (installation / départ) : état MOUVEMENT |

C'est le premier enregistrement de respiration **non cadencée** détecté : les
échecs des tests 2 et 3 venaient des mouvements du sujet, pas de la
respiration libre.  Le 15/min cadencé et le 20/min spontané passent tous deux
avec les mêmes seuils, et la salle vide reste muette.  Limite : le SNR
(~13 dB sur 20 s) laisse peu de marge ; tout gain matériel se traduira
directement en réactivité (§9.2) et en portée.

### 9.2 Réactivité

**Constat.**  La latence (~30 s) est bornée par le SNR, pas par le logiciel :
une raie à 13 dB sur 20 s tombe à ~9 dB sur 8 s, alors que le maximum de la
salle vide monte à 16.7 dB sur 8 s (moins de moyennage).  Aucun réglage ne
permet de détecter *la périodicité* plus vite au SNR actuel.  En revanche, on
peut détecter **la présence d'un être vivant** bien plus vite.

**Indices rapides (10 Hz, `radar/dsp/fast.py`).**  Filtres causaux, calculés
à chaque bloc de 0.1 s :

* *présence* : énergie 0.12–1 Hz sur 4 s de chaque voie (radiale /
  tangentielle), rapportée au bruit attendu **en salle vide** = bruit
  thermique (suivi en continu, 20e percentile glissant sur 60 s) + **bruit
  multiplicatif de la fuite** k·|C|².  Les constantes k se mesurent en salle
  vide : **−65.3 dBc de bruit de phase** (0.54 mrad, cohérent avec les
  0.55 mrad de `radar check`) et **−84.7 dBc de bruit d'amplitude** — la voie
  radiale est 20 dB plus propre que la tangentielle, ce qui explique après
  coup pourquoi elle porte la cible.  Sous H0, l'indice vaut ≈ 0 dB quel que
  soit le montage, et écarter les antennes (|C|² plus faible) le rend
  automatiquement plus sensible ;
* *activité* : énergie 1.5–15 Hz sur 1 s → mouvement franc en < 1 s.

Nouvel état **PRÉSENCE — signe de vie** (présence ≥ 6 dB pendant 1.5 s), entre
VIDE et RESPIRATION.

| enregistrement (1.8 GHz sauf test1) | 1re alerte | RESPIRATION confirmée |
|---|---|---|
| test1 (2.45 GHz, Wi-Fi) | 16.1 s | — |
| test2 (sujet qui bouge) | **4.2 s** | — (PRÉSENCE 62 %, MOUVEMENT 38 %) |
| apnée puis respiration | **4.2 s** | — (PRÉSENCE 63 %) |
| cadencé 15/min | **5.4 s** | 35.1 s |
| `resp_chambre` | **5.4 s** | 27.6 s |
| salle vide (75 s) | aucune | — (p99.9 de la présence : 3.1 dB) |

**Détecteur multi-échelle.**  Fenêtres 8, 12 et 20 s analysées en parallèle,
chacune avec son seuil (calibrable), les courtes exigeant en plus un rythme
stable (±0.04 Hz) et l'absence de mouvement franc.  Au SNR du labo, seule la
fenêtre de 20 s confirme ; dès que le SNR augmente, les courtes prennent le
relais.  Simulation (1.8 GHz, 12 séances de 90 s par point) :

| SNR cible (dB) | latence médiane, 20 s seule | multi-échelle | 1re alerte |
|---|---|---|---|
| 10 | 22.1 s | 22.1 s | 7.1 s |
| 15 | 22.1 s | 22.1 s | 5.4 s |
| 20 | 22.1 s | **17.9 s** | 5.4 s |
| 25–30 | 22.1 s | **14.1 s** | 5.4 s |

Fausses respirations (simulation, 8 × 120 s, dérive LO jusqu'à 10× celle du
Pluto) : marcheur 0 %, mouvements 2.3 %, vide 1.8 %, **identiques au
mono-échelle** (les 1.8 % en salle vide viennent de la dérive extrême, pas des
échelles courtes).

**Affichage.**  Jauges présence/activité et bande « mouvement thoracique »
de 30 s qui défilent en continu (60 images/s, données à 10 Hz), SNR de chaque
échelle, flux WebSocket sans perte.  Corrigés au passage : arrondi JSON à 4
décimales qui écrasait les petites valeurs (forme d'onde en paliers), attribut
`hidden` ignoré, interblocage possible en démarrant un enregistrement pendant
un autre.

**Évaluation = temps réel.**  `radar evaluate` rejoue désormais exactement la
chaîne du tableau de bord (`pipeline.Processor`) et rapporte le délai de 1re
alerte et l'échelle qui a confirmé.

### 9.3 Mode à fréquence balayée (SFCW), amplitude seule

**Pourquoi l'amplitude.**  Sur l'AD9363/9364, la phase TX/RX est aléatoire
après chaque changement de LO (deux PLL, pas de MCS sur l'AD9363, pas de
boucle RF interne comme sur le bladeRF de PMC7865734).  Mais elle est commune
à tous les échos d'un pas : |S_k| = |L_k + T_k| n'en dépend pas.  La fuite L,
qui domine, sert de référence interférométrique (comme en OCT spectrale) :
|L + T| ≈ |L| + Re(T·e^{−j∠L}).  C'est **exactement la voie radiale du CW**
(la plus propre, §9.2), avec trois avantages :

1. **diversité de fréquence** : la phase cible/fuite tourne d'un pas à
   l'autre, il n'y a plus de position « aveugle » ;
2. **séparation en distance** : la fuite (case 0), les murs et une personne
   qui marche ailleurs ne polluent plus la case de la victime ;
3. le bruit d'amplitude commun à tout un balayage (puissance TX…) tombe dans
   la case 0.

Contreparties : mesures réelles → spectre symétrique → portée non ambiguë
**c/(4Δf)** et non c/(2Δf) (3.75 m pour Δf = 20 MHz) ; cadence limitée par le
temps de changement de LO ; rapport cyclique < 100 % (~5–10 dB d'intégration
perdus face au CW) ; échos fantômes (produits croisés cible × réflecteurs
fixes) ~25 dB sous la cible.

**Mise en œuvre (`radar/sfcw/`).**

* `calib_mode = manual` : vérifié dans le source du driver AD9361, en mode
  auto chaque saut du LO TX de plus de 100 MHz déclenche une calibration de
  quadrature TX, **attendue** par l'écriture du LO (lent, émission parasite) ;
* écriture directe des attributs de LO (canaux IIO en cache) ;
* **buffers périmés repérés par marquage** : un pas sur deux, le LO RX est
  décalé de 125 kHz, la tonalité tombe donc alternativement à 250 et 125 kHz
  en bande de base ; un buffer n'est accepté que si la bonne tonalité domine
  l'autre de 26 dB, le plancher de 30 dB, **et** que ses deux moitiés ont la
  même amplitude à 10 % près (rejette le buffer « à cheval » sur le
  changement de LO, piège trouvé en simulation) ;
* traitement : fond adaptatif (MTI, τ = 30 s), modulation relative
  (A − B)/B (la réponse des antennes se simplifie), FFT sur les pas, puis
  par case de distance : SNR respiratoire, énergie de mouvement rapportée
  au plancher des cases calmes ; une respiration candidate située dans une
  zone en mouvement étalée (> 0.8 m) est rejetée (état MOUVEMENT) ;
* un **faux Pluto** (`radar/sfcw/scene.py`) reproduit latences, buffers
  noyau périmés, TX/RX désaccordés et phase aléatoire par pas : tout le code
  matériel tourne et se teste ici.

**Résultats en simulation** (1.2–2.18 GHz, 50 pas de 20 MHz, 15 cm de
résolution, 4 balayages/s) :

| scénario | résultat |
|---|---|
| respiration à 0.8 / 1.2 / 1.6 / 2.0 / 2.8 / 3.0 m, cible 35–50 dB sous la fuite | RESPIRATION 100 %, distance à ±3 cm |
| salle vide | 0 % |
| marcheur (1.5–3.5 m) | MOUVEMENT 100 %, 0 % de fausse respiration |
| **victime immobile à 1.0 m + marcheur de 1.5 à 3.5 m** | **RESPIRATION à 1.00 m, 100 % du temps** (le CW ne peut pas séparer les deux) |
| code Pluto sur faux Pluto | 0 pas raté, 4 lectures/pas (2 périmés + 1 à cheval + 1 frais), 3.8 balayages/s avec 1.9 ms par écriture de LO (hypothèse) |

**À mesurer chez vous** : `radar sfcw bench` (latence réelle d'écriture des LO
sur le lien NCM, lectures par pas, cadence, **répétabilité d'amplitude**
et plancher en distance → cible la plus faible détectable).  Leviers si la
cadence est trop basse : moins de pas (Δf plus grand, portée plus courte),
fastlock (`--fastlock` mesure le gain ; 8 profils seulement), ou, plus tard,
piloter le balayage directement sur le Pluto (script shell sur le Zynq :
écritures sysfs en µs au lieu d'allers-retours réseau).

### 9.4 Passage à 5.8 GHz

* Déverrouillage AD9364 (70 MHz–6 GHz, 56 MHz de bande) par l'utilisateur,
  en SSH root (README, « Passer à 5.8 GHz ») ; le mode 2r2t est réputé
  capricieux sur les Rev.C et n'est pas nécessaire ici.
* `sdr.chip: ad9364` dans la config ; la validation refuse > 3.8 GHz sur une
  AD9363 en indiquant la marche à suivre ; au démarrage, la plage de LO
  réellement annoncée par le firmware est vérifiée (`radar check` l'affiche).
* `configs/lab_5800.yaml` (CW à 5.8 GHz + SFCW 5.2–6.0 GHz).  `radar scan`
  (bruit ambiant, et fuite TX→RX avec `--tx`) pour choisir une fréquence
  calme : drones FPV, télépéage DSRC 5.795–5.815 GHz, Wi-Fi.
* Attendu : λ = 5.2 cm → pour le même déplacement, **3.2× plus de phase
  (+10 dB de modulation)** ; le bruit de phase des PLL croît aussi (~+10 dB),
  mais en millimètres équivalents il est inchangé.  En simulation, la latence
  médiane tombe à 10.6 s (les échelles courtes confirment).
* **Avec les antennes d'origine, 5.8 GHz sera mauvais** (résonances 0.9 et
  1.7–2.0 GHz) : il faut des antennes adaptées (§9.6).  Constantes de bruit
  de la fuite à recalibrer à 5.8 GHz (`radar calibrate` sur un « vide »).
* Réglementation : 5725–5875 MHz, 25 mW PIRE (ERC/REC 70-03) ; le Pluto sort
  quelques mW à 5.8 GHz, on reste en dessous même avec un patch de 8 dBi.

### 9.5 La phase est-elle la seule information utilisable ?

Non.  Le radar mesure un écho complexe en fonction du temps (et, en SFCW, de
la fréquence).  Ce qu'on peut en tirer :

| information | ce qu'elle donne | état dans le projet |
|---|---|---|
| **phase** (angle de l'écho) | déplacement, sensibilité 4π/λ — la plus fine | utilisée ; polluée par la dérive TX/RX (voie tangentielle) |
| **amplitude / voie radiale** \|L+T\| | même mouvement vu par interférence avec la fuite, **insensible à la dérive LO** (20 dB plus propre au labo) | porte la cible dans tous les enregistrements réels ; base du SFCW |
| **Doppler / micro-Doppler** (spectre du slow-time 0–25 Hz) | respiration 0.1–0.8 Hz, cœur ~1–2 Hz, gestes > 1.5 Hz, marche (~12 Hz à 1.8 GHz pour 1 m/s) : nature du mouvement | indice d'activité ; spectrogramme = bonne entrée pour l'IA (humain vs ventilateur) |
| **distance** | sépare victime, sauveteurs, murs ; localise | **SFCW** (15 cm sur 1 GHz, implémenté) ; FMCW/OFDM dans la bande instantanée : 7.5 m (20 MHz) ou 2.7 m (56 MHz, AD9364), trop grossier dans une pièce ; CW à deux tonalités : il faudrait Δf ~10 MHz, donc ≥ 20 MS/s, que l'USB 2.0 ne tient pas en continu |
| **angle** (2 RX espacées de λ/2) | direction de la cible, séparation de deux personnes ; **bonus : le 2e RX partage le LO → canal de référence qui annule la dérive commune** | nécessite le mode 2r2t (capricieux sur Rev.C) ; piste forte à moyen terme |
| **polarisation** | antennes croisées : moins de fuite directe ; le corps dépolarise un peu | à essayer avec des patchs polarisés circulairement (§9.6) |
| **radar passif** (Wi-Fi, 5G existants) | pas d'émission, sensibilité correcte en intérieur (littérature « Wi-Fi sensing ») | demande 2 RX (référence + surveillance) ; aucun illuminateur sous des décombres : hors sujet pour le secours |
| **statistiques temporelles** (asymétrie inspiration/expiration, variabilité du rythme) | distinguer un humain d'une machine | alerte « rythme trop régulier » ; à confier à l'IA |

### 9.6 Antennes et placement

**Ce que font les systèmes commerciaux.**  Les détecteurs de victimes
(LEADER Scan/Multisearch, Geotech RD-400, GSSI, LifeLocator…) sont des radars
**UWB impulsionnels ou GPR basse fréquence, posés au contact de la surface**.
LEADER annonce la respiration jusqu'à ~10 m à découvert et à travers ~50 cm
de béton ; le RD-400 affiche la **distance** de la victime (mouvement et
respiration sur deux traces), soit exactement l'idée du mode SFCW.  Trois
leçons : basse fréquence et large bande (pénétration + distance), antennes
directives au contact (pas de réflexion air/surface, énergie couplée dans le
milieu), et découplage TX/RX soigné.

**Le point faible actuel** : deux fouets omnidirectionnels à 2 cm l'un de
l'autre, c'est le pire couplage possible (ils rayonnent l'un dans l'autre),
d'où la fuite à −9 dBFS et le bruit multiplicatif qui borne le SNR.  La
barrette de RAM glissée entre eux a déjà gagné 10 dB.

**Propositions (prix vérifiés le 27/09/2026)** :

| antenne | bande | gain | prix | pour |
|---|---|---|---|---|
| **LP0965** log-périodique sur PCB (conçue par Kent Electronics, vendue par Ettus/NI) | 0.85–6.5 GHz | 5–6 dBi, directive | ~25 $ pièce (eBay) ; en stock chez Passion Radio (FR, 24–48 h) ; connecteur SMA à souder | **premier choix** : couvre 1.8, 2.45 et 5.8 GHz et le SFCW large bande ; en prendre 2 |
| log-périodique PCB générique 740–6000 MHz | 0.74–6 GHz | 6–7 dBi | ~20–40 $ (Amazon/AliExpress) | équivalent moins cher, qualité variable |
| patch FPV 5.8 GHz | 5.65–5.95 GHz | 8–14 dBi | ~9 $ (eBay) | 5.8 GHz seulement ; souvent **polarisé circulairement** : prendre TX RHCP + RX LHCP (une réflexion inverse le sens de rotation ; bonus : moins de fuite directe) |
| Vivaldi UWB (RFSPACE/WiMo) | 0.43–6 GHz | ~6 dBi | ~374 € (WiMo) | trop cher ; des Vivaldi PCB génériques existent à bas prix mais sans garantie |
| Nooelec UWB Surveyor | 0.7–10 GHz | 3 dBi, omni | ~45 € | omnidirectionnelle : ne réduit pas la fuite, déconseillée ici |

**Placement bistatique** :

1. deux antennes directives côte à côte, **20 à 50 cm d'écart**, orientées
   vers le thorax (rallonges SMA ; ~0.5 dB de pertes à 1.8 GHz pour 50 cm) ;
2. un **écran entre les deux** : plaque métallique reliée à la masse ou
   absorbant (mousse chargée), qui dépasse de 10–20 cm vers l'avant ;
3. antennes et câbles **fixés** (ruban adhésif, support rigide) : toute
   vibration module la fuite, donc le bruit ;
4. contrôle : `radar check` (fuite en dBFS : viser −30 dBFS ou moins au lieu
   de −9) puis un enregistrement « vide » et `radar calibrate` (les constantes
   de bruit de la fuite doivent baisser d'autant) ;
5. sur des décombres : antennes posées au contact de la surface, comme les
   systèmes commerciaux.

**Test à travers la planche** : `radar protocole --plan planche` enchaîne
salle vide, 50 cm avec et sans planche, 1 m avec et sans planche, planche
seule.  Pertes attendues pour ~1 cm de bois sec (ε ≈ 2, tan δ ≈ 0.05), aller-
retour : ~0.3 dB à 1.8 GHz (négligeable, la planche ajoute surtout un écho
fixe), ~1 dB à 5.8 GHz, **~8–10 dB à 60 GHz** (plus les réflexions aux
interfaces).

### 9.7 Module 60 GHz bon marché

| | **Seeed XIAO MR60BHA2** | Hi-Link HLK-LD6002 |
|---|---|---|
| prix | **24,90 $** (kit avec XIAO ESP32C6, boîtier, USB-C) | ~10–20 $ (module nu) |
| mesures | respiration, cœur, distance, présence | respiration, cœur, distance, phases |
| portée annoncée | ≤ 1.5 m (0.4–2 m) | 1.5 m |
| liaison | USB-C via l'ESP32C6 (croquis de relais fourni) | UART **1 382 400 bauds**, 3.3 V / 600 mA |
| protocole | TinyFrame, types 0x0A14/15/16… | **le même** |

**Recommandation : le MR60BHA2.**  Même famille de radar et même protocole
que le LD6002, mais prêt à brancher : alimentation, USB et boîtier sont
inclus.  Le LD6002 est moins cher, mais il faut un adaptateur UART rapide
(le FTDI FT230X convient, pas un CP210x) et une alimentation 3.3 V / 600 mA
séparée (la broche 3.3 V d'un adaptateur USB ne fournit pas ce courant).

Le kit sort d'usine avec un firmware ESPHome qui envoie les mesures en Wi-Fi.
Il suffit de téléverser le croquis `firmware/mr60bha2_passthrough` (Arduino
IDE, 5 minutes) pour recevoir les trames brutes sur l'USB.
`radar run --mr60 COMx` affiche alors le module à côté du Pluto.  Le lecteur
reconnaît aussi les journaux texte du firmware d'origine (mode « esphome »).

**Compromis.**  À 60 GHz, λ = 5 mm : le module est très sensible (cœur
compris), mais il **ne traverse pas les décombres**.  Le béton absorbe
plusieurs dizaines de dB par centimètre à 60 GHz ; une planche de 1 cm coûte
~10 dB aller-retour, ce qui peut encore passer à 50 cm.  Il ne remplace donc
pas le Pluto pour le secours. En revanche, il apporte deux choses :

1. un **point de comparaison** peu cher pour le rapport ;
2. une **vérité terrain du rythme** à courte distance. Pendant un
   enregistrement Pluto, ses mesures sont sauvegardées dans
   `<fichier>.mr60.json`, et `radar evaluate` / l'IA s'en servent comme
   rythme de référence. C'est plus pratique qu'une ceinture respiratoire.

### 9.8 Données : `radar protocole`

Les données restent le vrai goulot d'étranglement (§6.2). La calibration du
27/09 repose sur **75 s de salle vide** : les percentiles hauts sont donc mal
estimés, et `radar calibrate` borne désormais ses seuils par les valeurs par
défaut quand il y a moins de 10 min de « vide ».

La commande `radar protocole` guide une séance complète. Un seul moteur
tourne en continu, et chaque étape (consigne, étiquette, durée, annotations
« apnée » / « reprise ») devient un enregistrement étiqueté. En fin de
séance, elle écrit un manifeste, une calibration de séance et une
évaluation. Elle fonctionne en CW (`--mode cw`) comme en SFCW
(`--mode sfcw`), avec le module 60 GHz en option.

Trois plans sont fournis : `labo` (~20 min, positions variées), `planche`
et `vide_long` (10 min).

### 9.9 IA

Toujours rien d'entraîné : pas de GPU dans la session distante, et la RTX
5090 est chez l'utilisateur. La chaîne a en revanche été corrigée sur trois
points :

1. **Semi-synthétique** : la cible simulée ajoutée aux fonds vides réels
   utilisait la longueur d'onde de `ai/config.yaml` (3.5 GHz), alors que ces
   fonds ont été enregistrés à 1.8 GHz. La profondeur de modulation était
   donc fausse d'un facteur 2. On prend maintenant la porteuse de chaque
   enregistrement.
2. **Synthétique** : λ est tiré parmi 1.8, 2.45 et 5.8 GHz, pour que le même
   modèle serve après le passage à 5.8 GHz.
3. **Données réelles** : les enregistrements SFCW sont ignorés proprement, et
   le rythme du module 60 GHz sert de vérité terrain.

Le pré-entraînement sur simulateur est pertinent et peu coûteux (quelques
dizaines de minutes sur la 5090). Mais un affinage n'aura de sens qu'avec
plusieurs séances réelles (≥ 3 sessions « vide » et ≥ 3 sessions
« respiration », dans des lieux et montages différents).

### 9.10 À valider sur place (ordre conseillé)

1. `python -m pytest tests -q` (37 tests).
2. `radar.bat --config configs\lab.yaml run --calibration calibration\lab_1800.json` :
   jauges de présence/activité, bande défilante, état PRÉSENCE en ~5 s.
3. `radar.bat --config configs\lab.yaml sfcw bench` : latence des LO,
   cadence, répétabilité.  Puis `sfcw run`, et bouger devant les antennes.
4. `radar.bat --config configs\lab.yaml protocole --plan vide_long`, puis
   `--plan labo` : de vraies données de calibration.
5. `radar.bat --config configs\lab.yaml protocole --plan planche` (CW, puis
   `--mode sfcw`, puis avec `--mr60 COMx` si le module est là).
6. Déverrouillage AD9364, `check`, `scan` et `run` avec `configs\lab_5800.yaml`,
   en sachant que les antennes d'origine sont inadaptées à 5.8 GHz.
7. Commande d'antennes : 2 × LP0965, puis refaire les étapes 2 à 4 avec
   20–50 cm d'écart et un écran entre les deux antennes.
