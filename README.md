# Notes JS post réunion du 8 / 11/ 2024

1- À l'avenir, il faut enregistrer et transcrire les réunions.

2- À propos du LSTM qui semble avoir un bon score de classification:
- ne serait-il pas utile d'appliquer nos régresseurs _a priori_ sur les prédictions du modèle (dans la version random forest plus que dans la version linéaire) afin de savoir ce que le modèle a appris? Ce n'est pas la peine de s'enthousiasmer si le modèle classe simplement à partir de la longueur totale de la trajectoire, ou la vitesse moyenne..

# Stratégies d'étude

## Corrélation trajectoires / diagnostic neuro

### Approche paramétrique

Regarder les mushroom in view
Regarder les paramètres extraits par le NB, puis les mettre dans un MLP / faire du Linear dessus, voire ajouter des paramètres. C'est fait avec les moyennes et variances de 6 paramètres de trajectoire dans le NB extract_explicit_features.

On remarque que sans stratégie de data augmentation, cela semble complexe de ne pas overfit. Le seul moyen d'avoir une loss correcte et donc une bonne précision sur le train était de renormaliser les données, ce que j'ai fait malgré ma réticence initiale (peur de perdre des infos). Finalement, malgré avoir testé plusieurs hyperparamètres, aucun résultat probant sur un set de validation.



### Machine Learning

- Utilisation d'un CNN 1D
- Utilisation d'un CNN "naîf" 2D sur une image de la trajectoire. C'est quelque chose qui est fait dans le cadre de l'analyse temps-fréquence de signaux par exemple.
Pour le CNN 2D, on peut générer plusieurs type d'image, notamment en jouant sur les paramtères suivants :
- Trajectoire seule
- Présence ou non de champignons
- Présence ou non d'indicateurs temporels (timestamps? Gradient de couleur)

## Prédiction de trajectoires

# Utilisation du code

# Remarks 
Subjects "Fred" and "margaux" were deleted due to incomplete data, making dataloader fail when confronted to those profiles.

