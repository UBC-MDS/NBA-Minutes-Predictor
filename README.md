# Project Proposal: Predicting the Court Minutes of Players in the NBA

- author: Jack Tan
- contributor: Jarvis Nederlof

## Motivation and Purpose


The motivation behind this project is to help make decisions when placing bets on daily fantasy sports. In other words, we are applying "machine-learning" algorithms to help predict player performance. However, since there are many performance measures, we will only be focusing on predicting the number of minutes that each player will play in an upcoming game. We will focus on predicting minutes played as it represents one of the most important stats when it comes to predicting a players fantasy point production. The daily fantasy sports market represents billions of dollars in investment, and is expected to continue growing with a rule change allowing Sports Betting at the state level by the US Supreme Court in 2018, which makes predicting individual player performance valuable.

In addition to predicting player performance, player minutes can also be used to help viewers decide if they want to purchase tickets to an upcoming game. Some viewers are big fans of certain players so knowing if they will play a little or a lot could help them to make a better purchasing decision.

## Primary Research Question


How can we use historical player data to predict the number of minutes that players will play in upcoming NBA matches?

Our Hypothesis is that certain historical stats of a player can be used to make predictions that are better than just using the players' previous n-games average minutes.

## Origin of Data


Our dataset is sourced from Kaggle [here](https://www.kaggle.com/pablote/nba-enhanced-stats#2012-18_playerBoxScore.csv).

The data can be downloaded without Kaggle authentication [here](https://github.com/jnederlo/nba_data/blob/master/2012-18_playerBoxScore.csv).

## Data Analysis Plan


- First we will clean the data, and transform all data labels to be quantitative. 
- Then we will create new features based on the data (e.g. 5, 10, and 20 game rolling averages for various stats). 
- Next, we will split the data into training and testing sets (3:1 ratio). 
- Then we analyze the data and pick parameters that would make the most sense in predicting our target (i.e. by looking at correlations, etc.). This is an iterative process.
- Last, we will try different approaches for our predictive model (decision tree, logisitic regression, etc.) and see which one performs the best. The best performing model (or model ensemble) will be reported in our results.

## EDA Analysis and Figures


Before doing our EDA analysis, we calculated the rolling averages of the number of court minutes in the past 5, 10 and 20 games for each player and added them as additional features. This was necessary to do before splitting as it represents combining multiple player rows together into new feature columns. Then we split the data into train/test sets.

For our EDA table, we calculated the correlation between the features and our target (actual minutes played). The resulting correlations will give us clues as to what features are likely to be important when predicting the minutes in a players' upcoming game. 

For our resulting figures, we plan on making a residual error plot. This will help us determine if our model has systematic errors in predicting the number of minutes players played. In addition, we can plot our predictions against actual minutes played, alongside the players' 5 game average. We will be using the 5 game average as the benchmark to measure our predictions as highly correlated with actual minutes and represents a good benchmark to try to beat.

## Presentation of Results


We plan to present our results by comparing how our model performed vs. the players 5 game average and their actual minutes played. We will pick snapshots of certain players for certain games to highlight the value our predictions hold. These snapshots will be represented as a table like the example below:

| Player Name | 5-Game Average Minutes | Predicted Minutes | Actual Minutes |
| :--- | ---: | ---: | ---: |
|Player1 Name | x_1 minutes | x_1 minutes | x_1 minutes |
|Player2 Name | x_2 minutes | x_2 minutes | x_2 minutes |
|Player3 Name | x_3 minutes | x_3 minutes | x_3 minutes |
|Player4 Name | x_4 minutes | x_4 minutes | x_4 minutes |
..etc

Additionally, we will display a table outlining the results of different models compared to the results of our best derived model.

Finally, more presentations will be developed throughout the completion of this project.



