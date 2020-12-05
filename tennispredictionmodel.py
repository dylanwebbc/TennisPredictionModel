#TENNIS PREDICTION MODEL
#WRITTEN BY DYLAN WEBB 11.28.20

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#ROUND ROBIN PREDICTION

def roundRobin(nameList, year, tourney):
  #Important: nameList has just one column, unlike the usual "names" used below
  print("Round Robin")
  df = nameList 
  df["score"] = 0

  for i in range(len(df) - 1):

    #match player against players they haven't yet played
    names = pd.DataFrame(df["name"])
    for j in range(i + 1):
      names = names.drop(j)
    names.columns = ["player1"]
    names["player2"] = df["name"].iloc[i]
    outcome = predictionModel(names, year, tourney)

    #sort results into the score column
    for j in range(len(df) - (i + 1)):
      for k in range(len(df)):
        if df["name"].iloc[k] == outcome["predicted_winner"].iloc[j]:
          df.loc[k, "score"] += 1

  #print results
  pd.set_option('display.max_rows', None)
  df.index += 1
  df = df.sort_values(by = "score", ascending = False)
  print(df)

#TOURNAMENT WINNER PREDICTION

def predictTournament(round1, year, tourney, winner = False):
  pd.set_option("display.max_rows", None)

  prediction = predictionModel(round1, year, tourney)

  r = 0
  while len(prediction) > 1:
    r += 1
    if winner == False:
      print("\nRound", r)
      print(prediction)
      
    nextRound = pd.DataFrame(np.zeros((int(len(prediction)/2),2))).astype(str)
    nextRound.columns = ["player1", "player2"]
    for i in range(len(prediction)):
      if i % 2 == 0:
        nextRound.loc[int(i/2), "player1"] = prediction["predicted_winner"].iloc[i]
      else:
        nextRound.loc[int((i-1)/2), "player2"] = prediction["predicted_winner"].iloc[i]

    prediction = predictionModel(nextRound, 2019, "Wimbledon")

  print("\nFinal Round")
  print(prediction)

#DOUBLE FOREST TENNIS PREDICTION MODEL

def predictionModel(names, year, tourney):

  #Generate features for regressor forest
  features1 = approximateFeatures(names["player1"].iloc[0], names["player2"].iloc[0],
                               year, tourney, "2ndRnWon%")
  features2 = approximateFeatures(names["player1"].iloc[0], names["player2"].iloc[0],
                               year, tourney, "svWon%")
  features3 = approximateFeatures(names["player1"].iloc[0], names["player2"].iloc[0],
                               year, tourney, "1stRnWon%")
  
  for i in range(1, len(names)):
    features1 = features1.append(approximateFeatures(names["player1"].iloc[i], names["player2"].iloc[i], 
                                                  year, tourney, "2ndRnWon%"), ignore_index = True) 

    features2 = features2.append(approximateFeatures(names["player1"].iloc[i], names["player2"].iloc[i], 
                                                  year, tourney, "svWon%"), ignore_index = True) 

    features3 = features3.append(approximateFeatures(names["player1"].iloc[i], names["player2"].iloc[i], 
                                                  year, tourney, "1stRnWon%"), ignore_index = True)

  #classifier forest takes input from regressor forest to predict the winner
  #run 15 times total to capture average forest vote
  p = np.zeros((len(names)))
  for i in range(15):
    p = np.add(p, forestClassify(pd.DataFrame(forestRegress(features1, features2, features3, year))))
  p /= 15

  #stores predictions in a dataframe of names to return
  prediction = pd.DataFrame(np.zeros((len(names),2)))
  prediction.columns = ["predicted_winner","likelihood"]
  for i in range(len(names)):
    if p[i] > .5:
      prediction.loc[i, "predicted_winner"] = names["player1"].iloc[i]
      prediction.loc[i, "likelihood"] = p[i]
    else:
      prediction.loc[i, "predicted_winner"] = names["player2"].iloc[i]
      prediction.loc[i, "likelihood"] = 1 - p[i]
  
  return prediction

#FEATURE GENERATOR

def approximateFeatures(player, opponent, year, tourney, stat):
  df = pd.read_csv("tennis.csv")
  stats = pd.read_csv("stats.csv")

  yearMax = df["year"] < year
  yearMin = df["year"] >= year - 5
  past_mask = yearMin & yearMax
  df = df[past_mask]

  yearMax = stats["year"] < year
  yearMin = stats["year"] >= year - 5
  past_mask = yearMin & yearMax
  stats = stats[past_mask]

  playerW = df["winner_name"] == player
  playerL = df["loser_name"] == player

  opponentW = df["winner_name"] == opponent
  opponentL = df["loser_name"] == opponent

  #average values to fill in case of missing data
  #this is caused by players who don't usually compete in Grand Slams
  meanOffset = .9
  varOffset = .7

  playerWStat = np.mean(stats[stat + "player_w"]) * meanOffset
  playerLStat = np.mean(stats[stat + "player_l"]) * meanOffset
  playerWVar = np.mean(stats[stat + "pVariance_w"]) * varOffset
  playerLVar = np.mean(stats[stat + "pVariance_l"]) * varOffset

  opponentWStat = np.mean(stats[stat + "opponent_w"]) * meanOffset
  opponentLStat = np.mean(stats[stat + "opponent_l"]) * meanOffset
  opponentWVar = np.mean(stats[stat + "oVariance_w"]) * varOffset
  opponentLVar = np.mean(stats[stat + "oVariance_l"]) * varOffset

  pCommonOpponentWStat = np.mean(stats[stat + "playerCommon_w"]) * meanOffset
  pCommonOpponentLStat = np.mean(stats[stat + "playerCommon_l"]) * meanOffset
  pCommonOpponentWVar = np.mean(stats[stat + "pCommonVar_w"]) * varOffset
  pCommonOpponentLVar = np.mean(stats[stat + "pCommonVar_l"]) * varOffset

  oCommonOpponentWStat = np.mean(stats[stat + "opponentCommon_w"]) * meanOffset
  oCommonOpponentLStat = np.mean(stats[stat + "opponentCommon_l"]) * meanOffset
  oCommonOpponentWVar = np.mean(stats[stat + "oCommonVar_w"]) * varOffset
  oCommonOpponentLVar = np.mean(stats[stat + "oCommonVar_l"]) * varOffset

  #Calculate Player Statistics
  win = df[playerW]["w_" + stat]
  lose = df[playerL]["l_" + stat]

  if len(win) > 0:
    playerWStat = np.mean(win)
    playerWVar = np.var(win)
  if len(lose) > 0:
    playerLStat = np.mean(lose)
    playerLVar = np.var(lose)

  #Calculate Opponent Statistics
  win = df[opponentW]["w_" + stat]
  lose = df[opponentL]["l_" + stat]

  if len(win) > 0:
    opponentWStat = np.mean(win)
    opponentWVar = np.var(win)
  if len(lose) > 0:
    opponentLStat = np.mean(lose)
    opponentLVar = np.var(lose)

  #find common opponents
  opponents1 = pd.DataFrame(pd.concat([df[playerW]["loser_name"], df[playerL]["winner_name"]]))
  opponents2 = pd.DataFrame(pd.concat([df[opponentW]["loser_name"], df[opponentL]["winner_name"]]))
  opponents1.drop_duplicates(keep="first", inplace = True)
  opponents2.drop_duplicates(keep="first", inplace = True)
  
  commonOpponents = pd.merge(opponents1, opponents2)

  #calculate Player Common Opponent Statistics
  winTotal = np.empty(0)
  loseTotal = np.empty(0)
  
  for i in range(len(commonOpponents)):
    commonOpponentW = df["winner_name"] == commonOpponents.iloc[i][0]
    commonOpponentL = df["loser_name"] == commonOpponents.iloc[i][0]

    playerW_mask = playerW & commonOpponentL
    playerL_mask = playerL & commonOpponentW
    
    winTotal = np.append(winTotal, df[playerW_mask]["w_" + stat].values)
    loseTotal = np.append(loseTotal, df[playerL_mask]["l_" + stat].values)

  if winTotal.size > 0:
    pCommonOpponentWStat = np.mean(winTotal)
    pCommonOpponentWVar = np.var(winTotal)
  if loseTotal.size > 0:
    pCommonOpponentLStat = np.mean(loseTotal)
    pCommonOpponentLVar = np.var(loseTotal)

  #calculate Opponent Common Opponent Statistics
  winTotal = np.empty(0)
  loseTotal = np.empty(0)
  
  for i in range(len(commonOpponents)):
    commonOpponentW = df["winner_name"] == commonOpponents.iloc[i][0]
    commonOpponentL = df["loser_name"] == commonOpponents.iloc[i][0]

    opponentW_mask = opponentW & commonOpponentL
    opponentL_mask = opponentL & commonOpponentW

    winTotal = np.append(winTotal, df[opponentW_mask]["w_" + stat].values)
    loseTotal = np.append(loseTotal, df[opponentL_mask]["l_" + stat].values)

  if winTotal.size > 0:
    oCommonOpponentWStat = np.mean(winTotal)
    oCommonOpponentWVar = np.var(winTotal)
  if loseTotal.size > 0:
    oCommonOpponentLVar = np.var(loseTotal)
    oCommonOpponentLStat = np.mean(loseTotal)
  
  #label encode surface depending on tournament
  surface = 1
  if tourney == "Roland Garros":
    surface = 0
  elif tourney == "Wimbledon":
    surface = .5

  #store results and return
  results = pd.DataFrame([[surface, playerWStat, playerLStat, opponentWStat, opponentLStat, 
                           pCommonOpponentWStat, pCommonOpponentLStat, oCommonOpponentWStat, oCommonOpponentLStat,
                           playerWVar, playerLVar, opponentWVar, opponentLVar,
                           pCommonOpponentWVar, pCommonOpponentLVar, oCommonOpponentWVar, oCommonOpponentLVar]])

  results.columns = ["surface", stat + "player_w", stat + "player_l", stat + "opponent_w", stat + "opponent_l", 
                     stat + "playerCommon_w", stat + "playerCommon_l", stat + "opponentCommon_w", stat + "opponentCommon_l",
                     stat + "pVariance_w", stat + "pVariance_l", stat + "oVariance_w", stat + "oVariance_l",
                     stat + "pCommonVar_w", stat + "pCommonVar_l", stat + "oCommonVar_w", stat + "oCommonVar_l"]

  return results

#RANDOM FOREST 1 - REGRESSOR

def forestRegress(in1, in2, in3, year):
  df = pd.read_csv("stats.csv")
  predicted = pd.DataFrame()

  yearMax = df["year"] < year
  yearMin = df["year"] >= year - 5
  past_mask = yearMin & yearMax

  df = df[past_mask]

  stats = ["2ndRnWon%", "svWon%", "1stRnWon%"]

  X = df.drop(["year", "outcome", "2ndRnWon%", "svWon%", "1stRnWon%"], axis = 1)
  y = df[stats]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

  for i in range(len(stats)):
    stat = stats[i]

    statList = ["surface", stat + "player_w", stat + "player_l", stat + "opponent_w", stat + "opponent_l", 
        stat + "playerCommon_w", stat + "playerCommon_l", stat + "opponentCommon_w", stat + "opponentCommon_l",
        stat + "pVariance_w", stat + "pVariance_l", stat + "oVariance_w", stat + "oVariance_l",
        stat + "pCommonVar_w", stat + "pCommonVar_l", stat + "oCommonVar_w", stat + "oCommonVar_l"]

    sX_train = X_train[statList]
    sy_train = y_train[stat]

    forest = RandomForestRegressor(warm_start = True, oob_score = True, 
                                  min_samples_leaf = 6, n_estimators = 200, max_depth = 150)
    forest.fit(sX_train, sy_train.values.ravel())

    if i == 0:
      tempPrediction = pd.DataFrame(forest.predict(in1))
    elif i == 1:
      tempPrediction = pd.DataFrame(forest.predict(in2))
    else:
      tempPrediction = pd.DataFrame(forest.predict(in3))

    tempPrediction.columns = [stat]

    if stat == "2ndRnWon%":
      predicted = tempPrediction
    else:
      predicted = predicted.join(tempPrediction)

  return predicted

#RANDOM FOREST 2 - CLASSIFIER

def forestClassify(in_test):
  df = pd.read_csv("tennis.csv")

  win = df[["w_2ndRnWon%","w_svWon%","w_1stRnWon%"]]
  win.columns = ["2ndRnWon%","svWon%","1stRnWon%"]

  lose = df[["l_2ndRnWon%","l_svWon%","l_1stRnWon%"]]
  lose.columns = ["2ndRnWon%","svWon%","1stRnWon%"]
                  
  #data
  X = pd.concat([win,lose])

  #target
  y = pd.concat([pd.DataFrame(np.ones((len(win),1))),pd.DataFrame(np.zeros((len(lose),1)))])

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

  forest = RandomForestClassifier(warm_start = True, oob_score = True)

  forest.fit(X_train, y_train.values.ravel())
  predicted = forest.predict(in_test)

  return predicted

#TESTING FUNCTIONALITY
pd.set_option('display.max_rows', None)

#input test matches and winner will be predicted
names = pd.DataFrame([["Roger Federer", "Novak Djokovic"],["Roger Federer", "Rafael Nadal"]])
names.columns = ["player1", "player2"]
print(predictionModel(names, 2019, "Wimbledon"))

#input entire csv of first round of Wimbledon to predict on
print(predictionModel(pd.read_csv("wimbledon2019_round1.csv"), 2019, "Wimbledon"))

#Predicts entire tournament from first round using progressive predictions
predictTournament(pd.read_csv("wimbledon2019_round1.csv"), 2019, "Wimbledon")

#Conducts a round robin and prints how many matches won
roundRobin(pd.read_csv("Top30.csv"), 2020, "Australian Open")

#Testing Prediction Model on 2019 (the rest of the code)
df = pd.read_csv("tennis.csv")
mask = df["year"] == 2019
df = df[mask]

#create dataframe of player names
names = pd.DataFrame()
names["player1"] = df["winner_name"]
names["player2"] = df["loser_name"]
names["tourney"] = df["tourney_name"]

#randomly rearrange dataframe
for i in range(len(names)):
  p = np.random.random()
  if p < .5:
    temp = names['player1'].iloc[i]
    names["player1"].iloc[i] = names["player2"].iloc[i]
    names["player2"].iloc[i] = temp

#predict results using prediction model accross the various tournaments
mask = names["tourney"] == "Australian Open"
results = predictionModel(names[mask], 2019, "Australian Open")
mask = names["tourney"] == "Roland Garros"
results = results.append(predictionModel(names[mask], 2019, "Roland Garros"), ignore_index = True)
mask = names["tourney"] == "Wimbledon"
results = results.append(predictionModel(names[mask], 2019, "Wimbledon"), ignore_index = True)
mask = names["tourney"] == "US Open"
results = results.append(predictionModel(names[mask], 2019, "US Open"), ignore_index = True)

#compare predictions to actual match results and report accuracy
accuracy = 0
for i in range(len(results)):
  if results["predicted_winner"].iloc[i] == df["winner_name"].iloc[i]:
    accuracy += 1
accuracy /= len(results)

#accuracy usually between 60% and 63%
print("Proportion of Accurate Predictions:", accuracy)
