# Project Proposal: Predicting the Court Minutes of Players in NBA

<br>

## Motivation and Purpose

<br>

The motivation behind this project is to help make decisions when placing bets on NBA matches. In other words, we are applying "machine-learning" algorithms to help predict the outcomes. However, since match results are decided by many factors, we will only be focusing on predicting the number of minutes that each player will play on the respective teams. It has been shown that the number of minutes played correlates well with the number of points scored, and thus the total score of a team can be estimated from such. 

In addition to predicting match outcomes, player minutes can also be used to help viewers decide if they want to buy tickets to the game. Some viewers are big fans of certain players.

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

## Data Analysis Plan

<br>

First we will clean the data, and transform all data labels to be quantitative. Then we will split the data into training and testing sets (3:1 ratio). Then we plan on analyzing the data and pick parameters that would make the most sense in predicting our target. We will also create new parameters based off the data (e.g. weighted averages that are not already present). Afterwards, we will try different approaches in for our predictive model (decision tree, logisitic regression) and see which one performs the best. 


 ## Discuss One EDA Table and Figure
 
 <br>

Before doing our EDA analysis, we split the data into train/test sets.
For our EDA table, we will calculate the rolling averages of the number of court minutes in the past 5, 10 and 20 games for each player and calculated the correlation between them. We included these values as additional parameters to derive our model. 
<br>
For our figure, we plan on making a residual error plot. This will help us determine if our model has systematic errors in predicting the number of minutes players play.  

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





