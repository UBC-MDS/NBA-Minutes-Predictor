# Project Proposal: Predicting the Court Minutes of Players in NBA

<br>

## Motivation and Purpose

<br>



The motivation behind this project is to help make decisions when placing bets on daily fantasy sports. In other words, we are applying "machine-learning" algorithms to help predict player performance. However, since there are many performance measures, we will only be focusing on predicting the number of minutes that each player will play on the respective teams. It has been shown that the number of minutes played correlates well with the number of points scored, and thus the total score of a team can be estimated from such. 
<br>
In addition to predicting match outcomes, player minutes can also be used to help viewers decide if they want to buy tickets to the game. Some viewers are big fans of certain players.
<br>
The daily fantasy sports market represents billions of dollars in investment, and is expected to continue growing with a rule change allowing Sports Betting at the state level by the US Supreme Court in 2018.

<br>

## Primary Research Question

<br>

How can we use historical player data to predict the number of minutes that players will play in upcoming NBA matches?

<br>

## Origin of Data

<br>

Here is the dataset we will derive our model from:
<br>
https://www.kaggle.com/pablote/nba-enhanced-stats#2012-18_playerBoxScore.csv
<br>
To download the data without a Kaggle API key, use the following link:
<br>
https://github.com/jnederlo/nba_data/blob/master/2012-18_playerBoxScore.csv


<br>

## Data Analysis Plan

<br>

First we will clean the data, and transform all data labels to be quantitative. 
<br>
Then we will create new features based off the data (e.g. 5, 10, and 20 game rolling averages for various stats). 
<br>
Next, we will split the data into training and testing sets (3:1 ratio). Then we plan on analyzing the data and pick parameters that would make the most sense in predicting our target.
<br>
Last, we will try different approaches in for our predictive model (decision tree, logisitic regression) and see which one performs the best. 


 ## Discuss One EDA Table and Figure
 
 <br>

Before doing our EDA analysis, we calculated the rolling averages of the number of court minutes in the past 5, 10 and 20 games for each player and added them as additional features. Then we split the data into train/test sets.
<br>
For our EDA table, we calculated the correlation between them. We included these values as additional parameters to derive our model. 
<br>
For our figure, we plan on making a residual error plot. This will help us determine if our model has systematic errors in predicting the number of minutes players play. In addition, we can plot our predictions against actuals, alongside the players 5 game average. We will be using the 5 game average as the benchmark to measure our predictions. 

## Presentation of Results

<br>

A table can be created for each player on the two teams

| Player ID | Predicted Minutes |
| :--- | ---: |
|Player1 ID | x_1 minutes |
|Player2 ID | x_2 minutes |
|Player3 ID | x_3 minutes |
|Player4 ID | x_4 minutes |
..etc

more results can be derived. We are not sure which ones yet.



