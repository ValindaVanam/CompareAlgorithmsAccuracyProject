#Name: Valinda Vanam 
#Student ID: 700703487
#Email: vxv34870@ucmo.edu
# Importing libs

import os
import sqlite3
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from model import ML, plot_confusion_matrix

warnings.simplefilter("ignore")

# loading dataset from the sqlite file
with sqlite3.connect(os.path.join('Datasets', 'database.sqlite')) as db:
    countryData = pd.read_sql_query("SELECT * from Country", db)
    matchData = pd.read_sql_query("SELECT * from Match", db)
    leagueData = pd.read_sql_query("SELECT * from League", db)
    teamData = pd.read_sql_query("SELECT * from Team", db)

# displaying top-5 rows of all the dataset
print('\nDisplaying top-5 rows of Countries data')
display(countryData.head())

print('\nDisplaying top-5 rows of Match data')
display(matchData.head())

print('\nDisplaying top-5 rows of League data')
display(leagueData.head())

print('\nDisplaying top-5 rows of Team data')
display(teamData.head())

# descriptive analysis of the datasets
print('\nDisplaying descriptive analysis of Match data')
display(matchData.describe())

print('\nDisplaying descriptive analysis of League data')
display(leagueData.describe())

# displaying the rows and columns of the datasets
print('\nDisplaying the rows and columns')
print('\tCountries data: ', countryData.shape)
print('\tLeague data: ', leagueData.shape)
print('\tMatch data: ', matchData.shape)
print('\tTeam data: ', teamData.shape)

# renaming the column name
countryData = countryData.rename(columns={'id': 'country_id', 'name': 'country_name'})
leagueData = leagueData.rename(columns={'name': 'league_name'})

# merging the datasets
leagues = countryData.merge(leagueData, on='country_id')
tempData = matchData[matchData.league_id.isin(leagues['id'])]

# feature extraction
features = ['id', 'league_id', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'season']
Dataset = tempData[features]
Dataset["total_goals"] = Dataset['home_team_goal'] + Dataset['away_team_goal']
Dataset.dropna(inplace=True)

# Adding results column of the match
Dataset["result"] = np.where(Dataset['home_team_goal'] == Dataset['away_team_goal'], 0,
                             np.where(Dataset['home_team_goal'] > Dataset['away_team_goal'], 1, -1))

# displaying the final dataset
print('\nDisplaying the final dataset ')
display(Dataset.head())

# shape of the final dataset
print(f'\nDisplaying the rows and columns count of the final dataset : {Dataset.shape}')

# separating the dataset based on country name
newDataset = pd.merge(Dataset, leagues, left_on='league_id', right_on='id')
newDataset = newDataset.drop(['id_x', 'id_y', 'country_id'], axis=1)

belgium = newDataset[newDataset.country_name == 'Belgium']
english = newDataset[newDataset.country_name == "England"]
spanish = newDataset[newDataset.country_name == "Spain"]
german = newDataset[newDataset.country_name == "Germany"]

# visualization: goals scored by each country
plt.title("Plotting goals scored by each country")
seasons = Dataset['season'].unique()
plt.xticks(range(len(seasons)), seasons)
plt.style.use('ggplot')
plt.xlabel("Seasons")
plt.ylabel("Total Goals Scored")

plt.plot(english.groupby('season').total_goals.sum().values, 'ro-', german.groupby('season').total_goals.sum().values,
         'gv-', spanish.groupby('season').total_goals.sum().values, 'bp-')
plt.legend(["English Premier League", "German Bundesliga", "Spanish La Liga"])
plt.show()

# visualization: Average goals scored by each country
plt.xticks(range(len(seasons)), seasons)
plt.xlabel("Season")
plt.style.use('ggplot')
plt.title("Average goals scored by each country")
plt.ylabel("Average goals")

plt.plot(english.groupby('season').total_goals.mean().values, 'ro-', german.groupby('season').total_goals.mean().values,
         'gv-', spanish.groupby('season').total_goals.mean().values, 'bp-')
plt.legend(["English Premier League", "German Bundesliga", "Spanish La Liga"])
plt.show()

# visualization: goals scored by each country in home matches
plt.title("Plotting goals scored by each country in home matches")
seasons = Dataset['season'].unique()
plt.xticks(range(len(seasons)), seasons)
plt.style.use('ggplot')
plt.xlabel("Seasons")
plt.ylabel("Home-goals")

plt.plot(english.groupby('season').home_team_goal.sum().values, 'ro-',
         german.groupby('season').home_team_goal.sum().values,
         'gv-', spanish.groupby('season').home_team_goal.sum().values, 'bp-')
plt.legend(["English Premier League", "German Bundesliga", "Spanish La Liga"])
plt.show()

# visualization: goals scored by each country in away matches
plt.title("Plotting goals scored by each country in away matches")
seasons = Dataset['season'].unique()
plt.xticks(range(len(seasons)), seasons)
plt.style.use('ggplot')
plt.xlabel("Seasons")
plt.ylabel("Away-goals")

plt.plot(english.groupby('season').away_team_goal.sum().values, 'ro-',
         german.groupby('season').away_team_goal.sum().values,
         'gv-', spanish.groupby('season').away_team_goal.sum().values, 'bp-')
plt.legend(["English Premier League", "German Bundesliga", "Spanish La Liga"])
plt.show()

# visualization: home to away goals ratio scored by each country
plt.title("Plotting home to away goals ratio scored by each country")
seasons = Dataset['season'].unique()
plt.xticks(range(len(seasons)), seasons)
plt.style.use('ggplot')
plt.xlabel("Seasons")
plt.ylabel("Home to Away goals ratio")

plt.plot(english.groupby('season').home_team_goal.sum().values / english.groupby('season').away_team_goal.sum().values,
         'ro-',
         german.groupby('season').home_team_goal.sum().values / german.groupby('season').away_team_goal.sum().values,
         'gv-',
         spanish.groupby('season').home_team_goal.sum().values / spanish.groupby('season').away_team_goal.sum().values,
         'bp-')
plt.legend(["English Premier League", "German Bundesliga", "Spanish La Liga"])
plt.show()

# loading statistics dataset downloaded from http://football-data.co.uk/data.php
stats = pd.DataFrame()
for files in glob(os.path.join('Datasets', "*.csv")):
    stats = pd.concat([stats, pd.read_csv(files)], ignore_index=True)

# extracting features from the statistics dataset
features = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
            'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
stats = stats[features]

# displaying top-5 rows of statistics dataset
print('\nDisplaying top-5 rows of statistics data')
display(stats.head())

# shape of the statistics dataset
print(f'\nDisplaying the rows and columns count of the statistics dataset :{stats.shape}')

# preparing dataset for Machine learning process
DatasetML = pd.DataFrame(columns=('Team', 'HGS', 'AGS', 'HAS', 'AAS', 'HGC', 'AGC', 'HDS', 'ADS'))

avg_home_scored = stats.FTHG.sum() / stats.shape[0]
avg_away_scored = stats.FTAG.sum() / stats.shape[0]

temphome = stats.groupby('HomeTeam')
tempaway = stats.groupby('AwayTeam')

DatasetML['Team'] = temphome.sum().reset_index().HomeTeam.values
DatasetML['HGS'] = temphome.FTHG.sum().values
DatasetML['HGC'] = temphome.FTAG.sum().values
DatasetML['AGS'] = tempaway.FTAG.sum().values
DatasetML['AGC'] = tempaway.FTHG.sum().values
DatasetML['HAS'] = (DatasetML.HGS / 304.0) / avg_home_scored
DatasetML['AAS'] = (DatasetML.AGS / 304.0) / avg_away_scored
DatasetML['HDS'] = (DatasetML.HGC / 304.0) / avg_away_scored
DatasetML['ADS'] = (DatasetML.AGC / 304.0) / avg_home_scored

# displaying the top-5 rows of ML dataset
print('\nDisplaying top-5 rows of ML dataset')
display(DatasetML.head())

featureDataset = stats[['HomeTeam', 'AwayTeam', 'FTR', 'HST', 'AST', 'HC', 'AC']]
f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for i, data in featureDataset.iterrows():
    f_HAS.append(DatasetML[DatasetML['Team'] == data['HomeTeam']]['HAS'].values[0])
    f_HDS.append(DatasetML[DatasetML['Team'] == data['HomeTeam']]['HDS'].values[0])
    f_AAS.append(DatasetML[DatasetML['Team'] == data['AwayTeam']]['AAS'].values[0])
    f_ADS.append(DatasetML[DatasetML['Team'] == data['AwayTeam']]['ADS'].values[0])

featureDataset['HAS'] = f_HAS
featureDataset['HDS'] = f_HDS
featureDataset['AAS'] = f_AAS
featureDataset['ADS'] = f_ADS

featureDataset["Result"] = np.where(featureDataset['FTR'] == 'H', 1, np.where(featureDataset['FTR'] == 'A', -1, 0))

homeTeamId = []
awayTeamId = []
for i, data in featureDataset.iterrows():
    try:
        homeTeamId.append(teamData[teamData['team_long_name'] == data['HomeTeam']]['team_api_id'].values[0])
    except:
        try:
            homeTeamId.append(
                teamData[teamData['team_long_name'].str.contains(data['HomeTeam'])]['team_api_id'].values[0])
        except:
            homeTeamId.append(None)
    try:
        awayTeamId.append(teamData[teamData['team_long_name'] == data['AwayTeam']]['team_api_id'].values[0])
    except:
        try:
            awayTeamId.append(
                teamData[teamData['team_long_name'].str.contains(data['AwayTeam'])]['team_api_id'].values[0])
        except:
            awayTeamId.append(None)

featureDataset['HomeTeam'] = homeTeamId
featureDataset['AwayTeam'] = awayTeamId

# dropping the NA values
featureDataset.dropna(inplace=True)

# displaying the top-5 rows of ML Feature dataset
print('\nDisplaying top-5 rows of ML Feature dataset')
display(featureDataset.head())

# shape of the statistics dataset
print(f'\nDisplaying the rows and columns count of the feature dataset dataset :{featureDataset.shape}')

# Splitting the data into a train and test set
x = featureDataset[['HomeTeam', 'AwayTeam', 'HAS', 'HDS', 'AAS', 'ADS', 'HST', 'AST', 'HC', 'AC']]
y = featureDataset.Result
index = np.random.rand(len(featureDataset)) < 0.8
X_train, X_test, Y_train, Y_test = x[index], x[~index], y[index], y[~index]
print(f'The shape of the train and test datasets are {X_train.shape} and {X_test.shape}')

# defining Machine learning algorithms
ml = ML()

# training the models
print("\n Training initiated")
ml.train(X_train, Y_train)

# predicting the class for the test data
print("\n prediction initiated")
results = ml.identify(X_test, Y_test)

# displaying the accuracy data
# getting the classification performance report
for name, report in results[2].items():
    print(f'classification performance report of {name}')
    print(report)

# getting the Accuracy score
print("\nAccuracy score of the models")
for name, score in results[0].items():
    print(f'Accuracy Score of {name} is {score}')

# plotting accuracy comparison
plt.bar(list(results[0].keys()), list(results[0].values()), facecolor='red')
plt.xlabel("Algorithms")
plt.ylabel('Accuracy')
plt.title("Comparison of Algorithms accuracy")
plt.show()

# piloting confusion matrix
for name, cnf_matrix in results[1].items():
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=list(set(Y_train)), title=f'Confusion matrix of {name}')
    plt.show()
