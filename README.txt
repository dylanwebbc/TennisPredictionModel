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

Methodology:
  Select ATP data from Jeff Sackman is stored in the tennis.csv file
  More stats from each match are extrapolated from tennis.csv and stored in the stats.csv file
  A random forest regressor is trained on stats.csv and a random forest classifier is trained on tennis.csv
  Historic data for two specific players is fed into the regressor to predict the following stats for the upcoming match
    -proportion of returns won on the opponent’s first serve (1stRnWon%),
    -proportion of returns won on the opponent’s second serve (2ndRnWon%)
    -proportion of serves won (svWon%)
  These predicted stats are fed into the classifier to obtain a prediction of the match winner

First-time Setup:
  Download Jeff Sackman's tennis_atp data and unzip, placing the tennis_atp-master folder with tennispredictionmodel.py and tennisstatsgenerator.py
  Run updateFiles(current year, "anything", fullUpdate = True) in tennispredictionmodel.py
  Follow the instructions for prediction

Instructions for Prediction:
  If you need to add the data from the most recent tournament, run updateFiles(year, "Tournament Name") in tennispredictionmodel.py
  Create a new tournament.csv file with two columns titled "player1" and "player2" and enter the names of opposing players in upcoming matches
    (format the names as "first last" as shown in the wimbledon2019_round1.csv file)
  Run predictTournament(pd.read_csv("tournament.csv"), Tournament Year, "Tournament Name") in tennispredictionmodel.py
