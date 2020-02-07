# NBA Minutes Predictor

- Authors: Jarvis Nederlof, Roc Zhang, Jack Tan

## About

We have built a regression model using a light gradient boosting model to predict the number of expected minutes an NBA basketball player will play in an upcoming game. Our final model performed well on an unseen test data set, achieving mean squared error of 38.24 with a coefficient of determination of 0.65. Both metrics showed better performance compared to a players 5-game average minutes played (our evaluation metric) of 50.24 and 0.55, $MSE$ and $R^2$ respectively. The results represent significant value in the context of Daily Fantasy Sports, and the prediction model could be used as is. However, we note possible areas of further improvement that, if explored, could provide improved predictions, and more value.

The data set used in this project is of the NBA Enhanced Box Score and Standings (2012 - 2018) created by Paul Rossotti, hosted on [Kaggle.com](https://www.kaggle.com/pablote/nba-enhanced-stats#2012-18_playerBoxScore.csv). It was sourced using APIs from [xmlstats](https://erikberg.com/api). A copy of this dataset is hosted on a separate remote repository located [here](https://github.com/jnederlo/nba_data) to allow easy download with authenticating a Kaggle account. The particular data file used can be accessed [here](https://github.com/jnederlo/nba_data/blob/master/2012-18_playerBoxScore.csv). Each row in the data set represents a player's box score statistics for a particular game. The box score statistics are determined by statisticians working for the NBA. There were 151,493 data examples (rows).

## Report

The final report can be found [here](https://github.com/UBC-MDS/DSCI_522_group408/blob/master/report.pdf).

## Usage

You can run this analysis a few different ways. Start by cloning/downloading this repository, and navigate to the root of the project using the command line.

### Run with Docker

To run the analysis using Docker type the following (fill <PATH_ON_YOUR_COMPUTER> with the absolute path to the root of this project on your computer):

```
> docker run --rm -v <PATH_ON_YOUR_COMPUTER>:/home/nba_minutes jnederlo/nba_minutes make -C '/home/nba_minutes' all
```

To clean up the analysis type:

```
> docker run --rm -v <PATH_ON_YOUR_COMPUTER>:/home/nba_minutes jnederlo/nba_minutes make -C '/home/nba_minutes` clean
```

The Docker container is hosted on Docker Hub and can be viewed [here](https://hub.docker.com/repository/docker/jnederlo/nba_minutes/general). The `Dockerfile` can be viewed [here](https://github.com/UBC-MDS/DSCI_522_group408/blob/master/Dockerfile).

### Run with Make

Alternatively, you can use `make` commands from the root of the directory of this project to reproduce the analysis. The commands are listed as follows:  

```
##### General commands #####
# Run the whole workflow
make all

# Clean all of the workflow outputs
make clean

##### Run the workflow one at a time in order #####
# Download the data and save to file
make data/2012-18_playerBoxScore.csv

# Wrangle and preprocess the data - generate features and save data to a file
make data/player_data_ready.csv

# Run the Exploratory Data Analysis (EDA) - save results in a file
make results/EDA-correl_df_neg_9.csv results/EDA-correl_df_pos_20.csv results/EDA-feat_corr.png results/EDA-hist_y.png

# Train the models and make predictions - generate figures for final report
make results/modelling-gbm_importance.png results/modelling-residual_plot.png results/modelling-score_table.csv 

# Generate the final report
make report.pdf
```

You can rewiew the `Makefile` [here](https://github.com/UBC-MDS/DSCI_522_group408/blob/master/Makefile).

If running locally, and not with Docker, make sure you have the required dependencies installed.

## Dependencies

 - Python 3.7.5 and Python packages:
	 - pandas==0.25.2
	 - numpy==1.17.2
	 - docopt==0.6.2
	 - requests==2.20.0
	 - tqdm==4.41.1
	 - selenium==3.141.0
	 - altair==4.0.1
	 - scikit-learn==0.22.1
	 - matplotlib==3.1.2
	 - selenium==3.141.0
	 - termcolor==1.1.0
	 - jupyterlab==1.2.3
	 - lightgbm==2.3.1
	 - xgboost==0.90
 - R version 3.6.1 and R packages:
	 - tidyverse==1.2.1
	 - docopt==0.6.2
 - System requirement:
	 - ChromeDriver==79.0.3945.36 # $ brew cask install chromedriver
	 [click here for more information](https://altair-viz.github.io/user_guide/saving_charts.html)
	 - Latex (TeX Live 2019)
	 [click here for more information](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex)

## Licence

The NBA Minutes Predictor materials here are licensed under the MIT License. If re-using/re-mixing please provide attribution and link to this repository.