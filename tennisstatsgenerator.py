#TENNIS STATS GENERATOR
#WRITTEN BY DYLAN WEBB 11.28.20

import numpy as np
import pandas as pd

#Generate statistics summarizing past player performance
def generateStats(player, opponent, year, stat):
  df = pd.read_csv("tennis.csv")

  yearMax = df["year"] < year
  yearMin = df["year"] >= year - 5
  past_mask = yearMin & yearMax
  
  df = df[past_mask]

  playerW = df["winner_name"] == player
  playerL = df["loser_name"] == player

  opponentW = df["winner_name"] == opponent
  opponentL = df["loser_name"] == opponent

  pCommonOpponentWStat = 0
  pCommonOpponentLStat = 0
  pCommonOpponentWVar = 0
  pCommonOpponentLVar = 0

  oCommonOpponentWStat = 0
  oCommonOpponentLStat = 0
  oCommonOpponentWVar = 0
  oCommonOpponentLVar = 0

  #Calculate Player Statistics
  playerWStat = np.mean(df[playerW]["w_" + stat])
  playerLStat = np.mean(df[playerL]["l_" + stat])

  playerWVar = np.var(df[playerW]["w_" + stat])
  playerLVar = np.var(df[playerL]["l_" + stat])

  #Calculate Opponent Statistics
  opponentWStat = np.mean(df[opponentW]["w_" + stat])
  opponentLStat = np.mean(df[opponentL]["l_" + stat])

  opponentWVar = np.var(df[opponentW]["w_" + stat])
  opponentLVar = np.var(df[opponentL]["l_" + stat])

  #find common opponents
  opponents1 = pd.DataFrame(pd.concat([df[playerW]["loser_name"], df[playerL]["winner_name"]]))
  opponents2 = pd.DataFrame(pd.concat([df[opponentW]["loser_name"], df[opponentL]["winner_name"]]))
  opponents1.drop_duplicates(keep = "first", inplace = True)
  opponents2.drop_duplicates(keep = "first", inplace = True)
  
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

  if winTotal.size > 0 and loseTotal.size > 0:
    pCommonOpponentWStat = np.mean(winTotal)
    pCommonOpponentLStat = np.mean(loseTotal)

    pCommonOpponentWVar = np.var(winTotal)
    pCommonOpponentLVar = np.var(loseTotal)
  else:
    playerWStat = float("NaN")

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

  if winTotal.size > 0 and loseTotal.size > 0:
    oCommonOpponentWStat = np.mean(winTotal)
    oCommonOpponentLStat = np.mean(loseTotal)

    oCommonOpponentWVar = np.var(winTotal)
    oCommonOpponentLVar = np.var(loseTotal)
  else:
    playerWStat = float("NaN")
  
  results = pd.DataFrame([[playerWStat, playerLStat, opponentWStat, opponentLStat, 
                           pCommonOpponentWStat, pCommonOpponentLStat, oCommonOpponentWStat, oCommonOpponentLStat,
                           playerWVar, playerLVar, opponentWVar, opponentLVar,
                           pCommonOpponentWVar, pCommonOpponentLVar, oCommonOpponentWVar, oCommonOpponentLVar]])

  results.columns = [stat + "player_w", stat + "player_l", stat + "opponent_w", stat + "opponent_l", 
                     stat + "playerCommon_w", stat + "playerCommon_l", stat + "opponentCommon_w", stat + "opponentCommon_l",
                     stat + "pVariance_w", stat + "pVariance_l", stat + "oVariance_w", stat + "oVariance_l",
                     stat + "pCommonVar_w", stat + "pCommonVar_l", stat + "oCommonVar_w", stat + "oCommonVar_l"]

  return results

#CREATE TENNIS.CSV FILE
#Cleans up data for calculations
df = pd.read_csv("tennis_atp-master/atp_matches_2003.csv")
df["year"] = 2003
for year in range(2004,2020):
  file = "tennis_atp-master/atp_matches_" + str(year) + ".csv"
  
  newdf = pd.read_csv(file)
  newdf["year"] = year
  df = df.append(newdf, ignore_index=True)

mask1 = df["tourney_name"] == "Australian Open"
mask2 = df["tourney_name"] == "Roland Garros"
mask3 = df["tourney_name"] == "US Open"
mask4 = df["tourney_name"] == "Wimbledon"

mask = mask1 | mask2 | mask3 | mask4

df = df[mask]

#Feature engineering
df["w_2ndIn"] = df["w_svpt"] - df["w_1stIn"]
df["l_2ndIn"] = df["l_svpt"] - df["l_1stIn"]

df["w_svWon%"] = (df["w_1stWon"] + df["w_2ndWon"]) / df["w_svpt"]
df["l_svWon%"] = (df["l_1stWon"] + df["l_2ndWon"]) / df["l_svpt"]

df["w_1stRnWon%"] = (df["l_1stIn"] - df["l_1stWon"]) / df["l_1stIn"]
df["l_1stRnWon%"] = (df["w_1stIn"] - df["w_1stWon"]) / df["w_1stIn"]

df["w_2ndRnWon%"] = (df["l_2ndIn"] - df["l_2ndWon"]) / df["l_2ndIn"]
df["l_2ndRnWon%"] = (df["w_2ndIn"] - df["w_2ndWon"]) / df["w_1stIn"]

#label encode surface
df["surface"] = df["surface"].astype("category")
df["surface"] = df["surface"].cat.codes
df["surface"] /= 2

#remove unecessary columns and output tennis.csv
df = df[["year", "tourney_name", "surface", "winner_name", "loser_name",
         "w_svWon%", "w_1stRnWon%", "w_2ndRnWon%",
         "l_svWon%", "l_1stRnWon%", "l_2ndRnWon%"]]

df.dropna(how = 'any', inplace = True)
df.to_csv('tennis.csv', index = False)

#CREATE STATS.CSV FILE
#remove first five years for generateStats function
year_mask = df["year"] >= 2008
df = df[year_mask]

outcome = pd.DataFrame(np.zeros((len(df),8)))
outcome.columns = ["player1", "player2", "year", "surface",
                   "2ndRnWon%", "svWon%", "1stRnWon%", "outcome"]

#randomly rearrange data frame and assign statistics
for i in range(len(df)):
  outcome.loc[i, "year"] = df["year"].iloc[i]
  outcome.loc[i, "surface"] = df["surface"].iloc[i]
  p = np.random.random()
  if p < .5:
    outcome.loc[i, "outcome"] = 1
    outcome.loc[i, "player1"] = df["winner_name"].iloc[i]
    outcome.loc[i, "player2"] = df["loser_name"].iloc[i]
    outcome.loc[i, "2ndRnWon%"] = df["w_2ndRnWon%"].iloc[i]
    outcome.loc[i, "svWon%"] = df["w_svWon%"].iloc[i]
    outcome.loc[i, "1stRnWon%"] = df["w_1stRnWon%"].iloc[i]
  else:
    outcome.loc[i, "player1"] = df["loser_name"].iloc[i]
    outcome.loc[i, "player2"] = df["winner_name"].iloc[i]
    outcome.loc[i, "2ndRnWon%"] = df["l_2ndRnWon%"].iloc[i]
    outcome.loc[i, "svWon%"] = df["l_svWon%"].iloc[i]
    outcome.loc[i, "1stRnWon%"] = df["l_1stRnWon%"].iloc[i]

#run generate stats function on every player matchup
stat1 = generateStats(outcome["player1"].iloc[0], outcome["player2"].iloc[0],
                      outcome["year"].iloc[0], "2ndRnWon%")
stat2 = generateStats(outcome["player1"].iloc[0], outcome["player2"].iloc[0],
                      outcome["year"].iloc[0], "1stRnWon%")
stat3 = generateStats(outcome["player1"].iloc[0], outcome["player2"].iloc[0],
                      outcome["year"].iloc[0], "svWon%")

for i in range(1,len(df)):
  stat1 = stat1.append(generateStats(outcome["player1"].iloc[i], outcome["player2"].iloc[i], 
                                    outcome["year"].iloc[i], "2ndRnWon%"), ignore_index = True) 

  stat2 = stat2.append(generateStats(outcome["player1"].iloc[i], outcome["player2"].iloc[i], 
                                    outcome["year"].iloc[i], "1stRnWon%"), ignore_index = True)

  stat3 = stat3.append(generateStats(outcome["player1"].iloc[i], outcome["player2"].iloc[i], 
                                    outcome["year"].iloc[i], "svWon%"), ignore_index = True)

#combine data frames and output stats.csv
outcome = outcome.drop(["player1", "player2"], axis = 1)
stats = outcome.join(stat1)
stats = stats.join(stat2)
stats = stats.join(stat3)

stats.dropna(how = 'any', inplace = True)
stats.to_csv('stats.csv', index = False)
