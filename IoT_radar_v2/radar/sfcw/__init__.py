"""Mode radar à fréquence balayée (SFCW), **amplitude seule**, pour le PlutoSDR.

Pourquoi l'amplitude seule
--------------------------
Sur l'AD9363/9364, la phase relative TX/RX est aléatoire après chaque
changement de LO (deux PLL indépendantes, pas de MCS sur l'AD9363, pas de
boucle RF interne comme sur le bladeRF de PMC7865734).  Mais cette phase est
**commune** à tous les échos d'un même pas : S_k = e^{jφ_k}·(L_k + T_k), donc
|S_k| = |L_k + T_k| ne dépend pas de φ_k.  La fuite TX→RX L, qui domine,
sert de **référence interférométrique** (comme en OCT spectrale) ::

    |L + T| ≈ |L| + Re(T·e^{−j∠L}),   T/L ∝ e^{−j2π f_k (τ_T − τ_L)}

Une transformée de Fourier sur les pas de fréquence de (|S_k| − fond)/fond
donne un profil en distance relatif au trajet de fuite.  La variation lente
d'une case de distance donne la respiration, avec :

* **diversité de fréquence** — la phase cible/fuite tourne d'un pas à
  l'autre : plus de « points aveugles » de la voie radiale du CW ;
* **séparation en distance** — la fuite (case 0), les murs et une personne qui
  marche ailleurs ne polluent plus la case de la cible ;
* le bruit d'amplitude commun à tout un balayage tombe dans la case 0.

Contreparties : les mesures sont réelles → portée non ambiguë c/(4·Δf)
(3.75 m pour Δf = 20 MHz) ; cadence limitée par le temps de changement de LO
(``radar sfcw bench`` le mesure) ; rapport cyclique < 100 % (perte de ~5–10 dB
d'intégration par rapport au CW).

Modules
-------
``dsp``     traitement distance × temps (fond, profil, énergie, respiration)
``scene``   simulateur de balayages + faux Pluto (tests sans matériel)
``sources`` sources : Pluto, simulation, relecture
``engine``  moteur temps réel + enregistrement (``data/sfcw``)
``bench``   mesure du temps de changement de fréquence et de la répétabilité
"""
