A double random forest tennis prediction model boasting 63% accuracy
Built using the scikitlearn module in python
Utilizes data compiled by Jeff Sackman https://github.com/JeffSackmann

Explanation:
tennisdatagenerator.py first uses the atp data from Jeff Sackman to create tennis.csv and stats.csv
tennispredictionmodel.py trains a random forest regressor on stats.csv and trains a random forest classifier on tennis.csv. Historic data from two players is fed into these forests to obtain a prediction of the match winner.
