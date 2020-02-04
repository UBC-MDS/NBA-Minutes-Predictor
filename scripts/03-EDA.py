# author: Roc Zhang
# date: 2020-01-25

"""
This script takes preprocessed data from `data` folder, performs EDA and saves the result tables and figures to `results` folder.

Both the input file path+name and the save folder are required as inputs.

Usage: 03-EDA.py --input_path_file=<file_name> --save_folder=<save_folder>

Options:
--input_path_file=<file_name>         path and file name of the input preprocessed data
--save_folder=<save_folder>	    folder to save the output table and figures

Example: python scripts/03-EDA.py --input_path_file=data/player_data_ready.csv --save_folder=results
"""

# Loading the required packages
# Data proc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Plot
import altair as alt
import matplotlib.pylab as pl

# Other Packages
from docopt import docopt
import sys
import os
from termcolor import colored

# Ignore warnings from packages
import warnings
warnings.simplefilter("ignore")

opt = docopt(__doc__)

def main(input_path_file, save_folder):
	# Load the preprocessed data from csv
	# e.g. 'player_data_ready.csv'
	
	# Validate the file-path to load file
	path_str = str(input_path_file)
	if os.path.exists(path_str) == False:
		print(colored('ERROR: Path to file is not valid!', 'red'))
	try:
		df = pd.read_csv(path_str)
		print(colored('\nData loaded successfully!', 'green'))
	except:
		print(colored("ERROR: Data can't be loaded!", 'red'))
		raise


	# Validate the save_foler directory exists or make folder
	if os.path.exists(str(save_folder)) == False:
		try:
			os.makedirs(save_folder)
		except:
			print(colored('ERROR: Path to save directory is not valid!', 'red'))
			raise
	
	#######################################
	########### EDA starts here ###########
	#######################################

	# Remove Unnecessary Columns
	info_cols = ['playDispNm', 'gmDate', 'teamAbbr']
	df = df.drop(columns=info_cols)

	# Use only train split to perform EDA
	df_train, df_test = train_test_split(df, test_size=0.2)
	print(colored('Train test split finished!', 'green'))

	# Make and save histogram of the target - playMin
	ax = df_train['playMin'].hist(bins=55, grid=False)
	pl.suptitle("Histogram of the target - playMin")
	fig = ax.get_figure()
	fig.savefig(str(save_folder)+'/EDA-hist_y.png')
	print(colored('EDA-hist_y.png successfully saved!', 'green'))

	# Calculate the correlations of the features against the target
	correlations = {}
	for col in df_train:
		if col == 'playMin':
			continue
		try:
			correlations[col] = round(np.corrcoef(df_train[col], df_train['playMin'])[0][1], 3)
		except:
			continue	
	correl_df = pd.DataFrame.from_dict(correlations, orient='index')
	correl_df.columns = ['corr w/ target']
	correl_df = correl_df.sort_values('corr w/ target', ascending=False)

	assert len(correl_df) == df_train.shape[1] - 2, "Correlation table is not correctly calculated!" # `playPos` and target are not included

	# Save the top positively / negatively correlated features
	correl_df_pos_20 = correl_df.iloc[:20, :].copy()
	correl_df_neg_9 = correl_df.iloc[-9:, :].sort_values('corr w/ target').copy()

	correl_df_pos_20.to_csv(str(save_folder)+'/EDA-correl_df_pos_20.csv')
	print(colored('EDA-correl_df_pos_20.csv successfully saved!', 'green'))
	correl_df_neg_9.to_csv(str(save_folder)+'/EDA-correl_df_neg_9.csv')
	print(colored('EDA-correl_df_neg_9.csv successfully saved!', 'green'))

	# make and save the visualization of feature importance
	correl_df.reset_index(inplace=True)
	correl_df.columns = ['stat', 'correlation']
	sort = list(correl_df.reset_index()['index'])
	# Base bar chart
	c1 = alt.Chart(correl_df).mark_bar(size=1, color='black').encode(
	    alt.X('correlation:Q',
	          title='Correlation',
	          scale=alt.Scale(zero=False, domain=[-.3, 1])),
	    alt.Y('stat:N', title="", sort=sort))
	# Base circle chart
	c2 = alt.Chart(correl_df).mark_circle(color='black', size=420).encode(
	    alt.X('correlation:Q', scale=alt.Scale(zero=False, domain=[-.4, 1])),
	    alt.Y('stat:N', sort=sort))
	# Base text chart
	c3 = alt.Chart(correl_df).mark_text(color='white', size=8).encode(
	    alt.X('correlation:Q', scale=alt.Scale(zero=False, domain=[-.4, 1])),
	    alt.Y('stat:N', sort=sort),
	    text=alt.Text('correlation:Q', format='.2f'))
	# Final chart object
	correl_loli = (c1 + c2 + c3).properties(
	        title='Feature Correlation with Target',
	        width=400
	    ).configure_title(
	        fontSize=20,
	        font='Courier',
	        anchor='start')
	
	# Save chart object
	correl_loli.save(str(save_folder)+'/EDA-feat_corr.png', scale_factor=.50)
	print(colored('EDA-feat_corr.png successfully saved!', 'green'))

	print(colored('\nEDA complete!', 'green'))

if __name__ == "__main__":
	main(opt["--input_path_file"], opt["--save_folder"])
