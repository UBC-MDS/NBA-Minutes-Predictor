# NBA Minutes Predictor

- authors: Jarvis Nederlof, Roc Zhang, Jack Tan

## About

We have built a regression model using a light gradient boosting model to predict the number of expected minutes an NBA basketball player will play in an upcoming game. Our final model performed well on an unseen test data set, achieving mean squared error of 38.24 with a coefficient of determination of 0.65. Both metrics showed better performance compared to a players 5-game average minutes played (our evaluation metric) of 50.24 and 0.55, $MSE$ and $R^2$ respectively. The results represent significant value in the context of Daily Fantasy Sports, and the prediction model could be used as is. However, we note possible areas of further improvement that, if explored, could provide improved predictions, and more value.

The data set used in this project is of the NBA Enhanced Box Score and Standings (2012 - 2018) created by Paul Rossotti, hosted on [Kaggle.com](https://www.kaggle.com/pablote/nba-enhanced-stats#2012-18_playerBoxScore.csv). It was sourced using APIs from [xmlstats](https://erikberg.com/api). A copy of this dataset is hosted on a separate remote repository located [here](https://github.com/jnederlo/nba_data) to allow easy download with authenticating a Kaggle account. The particular data file used can be accessed [here](https://github.com/jnederlo/nba_data/blob/master/2012-18_playerBoxScore.csv). Each row in the data set represents a player's box score statistics for a particular game. The box score statistics are determined by statisticians working for the NBA. There were 151,493 data examples (rows).

## Report

The final report can be found [here](https://github.com/UBC-MDS/DSCI_522_group408/blob/master/report.pdf).

## Usage

To replicate the analysis, clone this GiHub repository, install the [dependencies](#dependencies) listed below, and run the following scripts via the command line/terminal from the root of the directory of this project:

```
# Download the data and save to file
> Rscript scripts/01-data_download.r --url=https://raw.githubusercontent.com/jnederlo/nba_data/master/2012-18_playerBoxScore.csv --out_file=data/adfasdf.csv
```
```
# Wrangle and preprocess the data - generate features and save data to a file
> python scripts/02-data_preproc.py --input_path_file=data/2012-18_playerBoxScore.csv --save_folder=data
```
```
# Run the Exploratory Data Analysis (EDA) - save results in a file
> python scripts/03-EDA.py --input_path_file=data/player_data_ready.csv --save_folder=results
```
```
# Train the models and make predictions - generate figures for final report
> python scripts/04-model_fit.py --file_name=player_data_ready.csv --save_folder=results
```
```
# Generate the final report
> jupyter nbconvert --to pdf --template report.tplx report.ipynb
```
__A Quick Note__: _To generate the final report requires various latex installs. In a future release we will add a makefile to run the script, and will wrap the depencies in a docker container which should alleviate the task of reproducing the results and running the scripts without errors. Stay tuned for future realeses planned in the coming weeks._

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
	 - plotly-orca==1.2.1
	 - plotly==4.3.0
	 - selenium==3.141.0
	 - termcolor==1.1.0
	 - ChromeDriver==79.0.3945.36
	 - jupyterlab==1.2.3
 - R version 3.6.1 and R packages:
	 - tidyverse==1.2.1
	 - docopt==0.6.2


## Licence

The NBA Minutes Predictor materials here are licensed under the MIT License. If re-using/re-mixing please provide attribution and link to this repository.