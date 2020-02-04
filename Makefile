# author: Roc Zhang
# date: 2020-01-28

# This file runs all of the analysis sequentially, to reproduce all of the code and findings contained within this repository.
# `Make all` will run all of the scripts and generate the final report.
# 'Make clean` will remove all of the generate files (images, results, report).

# the `all` command that runs the whole workflow
all : data/2012-18_playerBoxScore.csv data/player_data_ready.csv results/EDA-correl_df_neg_9.csv results/EDA-correl_df_pos_20.csv results/EDA-feat_corr.png results/EDA-hist_y.png results/modelling-gbm_importance.png results/modelling-residual_plot.png results/modelling-score_table.csv report.pdf

# Download the data and save to file
data/2012-18_playerBoxScore.csv : scripts/01-data_download.r
	Rscript scripts/01-data_download.r --url=https://raw.githubusercontent.com/jnederlo/nba_data/master/2012-18_playerBoxScore.csv --out_file=data/2012-18_playerBoxScore.csv

# Wrangle and preprocess the data - generate features and save data to a file
data/player_data_ready.csv : data/2012-18_playerBoxScore.csv data/2012-18_playerBoxScore.csv
	python scripts/02-data_preproc.py --input_path_file=data/2012-18_playerBoxScore.csv --save_folder=data

# Run the Exploratory Data Analysis (EDA) - save results in a file
results/EDA-correl_df_neg_9.csv results/EDA-correl_df_pos_20.csv results/EDA-feat_corr.png results/EDA-hist_y.png : scripts/03-EDA.py data/player_data_ready.csv
	python scripts/03-EDA.py --input_path_file=data/player_data_ready.csv --save_folder=results

# Train the models and make predictions - generate figures for final report
results/modelling-gbm_importance.png results/modelling-residual_plot.png results/modelling-score_table.csv : data/player_data_ready.csv
	python scripts/04-model_fit.py --file_name=player_data_ready.csv --save_folder=results

# Generate the final report
report.pdf : report.tplx report.ipynb report.bib
	jupyter nbconvert --to pdf --template report.tplx report.ipynb

# the `clean` command that cleans all outputs 
clean : 
	rm -f data/*.csv
	rm -f results/*.png
	rm -f results/*.csv
	rm -f report.pdf
