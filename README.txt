Tennis Prediction Model
Written by Dylan Webb 
V 1.0 -- 11/28/2020
V 1.1 -- 12/4/2020
  added a round robin function which takes in a single column of player names and outputs the results of a round robin tournament among them
  added functionality for outputting just tournament winner rather than the winners of every round in the predictTournament function
  cosmetic changes

A random forest tennis prediction model boasting up to 64% accuracy in predicting the winner of a given match
Built using the scikitlearn module in python
Utilizes data compiled by Jeff Sackman https://github.com/JeffSackmann

Explanation:
tennisstatsgenerator.py first uses the atp data from Jeff Sackman to create tennis.csv and stats.csv
tennispredictionmodel.py trains a random forest regressor on stats.csv and trains a random forest classifier on tennis.csv.
Historic data from two players is fed into these forests to obtain a prediction of the match winner.
