# author: Roc Zhang
# date: 2020-01-25

"""
This script takes raw data from the 'data' folder in the project repository to perform data preprocessing work.
Data preprocessing mainly includes making rolling and ewm features used for predicting 'playMin'.

Both the input file path+name and the save folder are required as inputs.

Usage: 02-data_preproc.py --input_path_file=<file_name> --save_folder=<save_folder>

Options:
--input_path_file=<file_name>         path and file name of the input data to be pre-processed
--save_folder=<save_folder>	    folder to save the processed output data

Example: python scripts/02-data_preproc.py --input_path_file=data/2012-18_playerBoxScore.csv --save_folder=data
"""

# Loading the required packages
# Data proc
import pandas as pd
pd.set_option('mode.chained_assignment', None) # turn off warning message of SettingWithCopyWarning 
import numpy as np
# Other Packages
from docopt import docopt
from tqdm import tqdm
import sys
import os
from termcolor import colored
# Ignore warnings from packages in models
import warnings
warnings.simplefilter("ignore")

opt = docopt(__doc__)

def main(input_path_file, save_folder):
	# Load the original data from csv
	# e.g. '2012-18_playerBoxScore.csv'
	
	print(colored("\nWARNING: This script takes about 2 minutes to run\n", 'yellow'))

	# Validate the file-path to load file
	path_str = str(input_path_file)
	if os.path.exists(path_str) == False:
		print(colored('ERROR: Path to file is not valid!', 'red'))
	try:
		df = pd.read_csv(path_str)
		print(colored('Data loaded successfully!', 'green'))
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
	#### Data preprocessing starts here####
	#######################################

	df = df.dropna()

	# make `playRat` feature
	df['playRat'] = (df['playPTS'] + 
					(df['playBLK'] * 2) +
					(df['playTO'] * -0.5) +
					(df['playSTL'] * 2) +
					(df['playAST'] * 1.5) +
					(df['playTRB'] * 1.25))

	# test 'playRat' column
	assert df['playRat'].isna().sum() == 0, colored("ERROR: NaN value detected in 'playRat' column!!", 'red')

	# filter columns for modeling
	cols_to_use = ['gmDate', 'teamAbbr',  'teamLoc', 'teamRslt', 'playDispNm', 'playMin', 'playRat', 'playPos', 'playStat']
	df = df[cols_to_use].copy()
	df['gmDate'] = pd.to_datetime(df['gmDate']).copy() # format datetime for sorting

	# replace categorical values with numbers to apply `rolling` to them
	rep_dict = {'teamLoc': {'Home':1, 'Away':0},
				'playStat': {'Starter':1, 'Bench':0}}
	df = catg_num_replace(rep_dict, df)

	# test on the categorical value replacement
	assert list(df['teamLoc'].unique()) == [0,1], colored("ERROR: Categorical value replacement failed!", 'red')

	# make input variables for making rolling features and ewm features
	cols_keep = ['playDispNm', 'gmDate', 'teamAbbr', 'playMin', 'teamLoc', 'playStat', 'playPos']
	cols_roll = ['playMin','playRat']
	windows = [5, 20]
	ewm_alpha = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
	agg_funcs = ['median']

	df_org = df.copy()
	df = pd.DataFrame() 

	# iterate through names to make new df with rolling and ewm features
	name_list = list(df_org['playDispNm'].unique())

	print(colored("\nData processing in progress:", 'green'))
	for name in tqdm(name_list):
		thisguy = df_org.query("playDispNm == @name").sort_values('gmDate', ascending=True)
		if len(thisguy) < 30: # ignoring players that have less than 30 games' record
			continue
		cols_created = []
		
		# make rolling features
		cols_created_rolling, thisguy = make_rolling_features(cols_roll, windows, agg_funcs, thisguy)
		cols_created.extend(cols_created_rolling)
		# test on making rolling features
		assert len(cols_created_rolling) == len(cols_roll) * len(windows) * len(agg_funcs), "Number of created rolling features is wrong!"
		assert thisguy.isna().sum().sum() == 0, "NaN value detected when making rolling features!"

		# make ewm features
		cols_created_ewm, thisguy = make_ewm_features(cols_roll, ewm_alpha, thisguy)
		cols_created.extend(cols_created_ewm)
		# test on making rolling features
		assert len(cols_created_ewm) == len(cols_roll) * len(ewm_alpha) + len(ewm_alpha), "Number of created ewm features is wrong!"
		assert thisguy.isna().sum().sum() == len(ewm_alpha), "Number of ewm features containing NaN should be the same as the length of ewm_alpha!" #ewm_std features should have 1 NaN value for the first row


		# shift created features by 1 row so that it means the "last n games"          
		thisguy_result = meaningful_shift(cols_created, cols_keep, thisguy)
		# test on the meaningful shift
		assert len(thisguy) == len(thisguy_result), "Number of rows changed when making the meaningful shift!"
		assert thisguy[cols_created[3]].iloc[3] == thisguy_result[cols_created[3]].iloc[4], "Meaningful shift fails!"

		# append this guy's result table into df
		df = pd.concat((df, thisguy_result), axis=0, ignore_index=True).copy()

		
	df = df.dropna().copy()

	# wrangling part ends, save the result dataframe
	df.to_csv(str(save_folder)+'/player_data_ready.csv', index=False)
	print(colored('Data successfully saved!', 'green'))

	print(colored('\nData preprocessing complete!', 'green'))
##################################
######## Define Functions ########
##################################
def catg_num_replace(rep_dict, df_input):
	"""
	Replace categorical values with numbers to apply `rolling` to them.

	Parameters:
	-----------
	rep_dict -- (dict) the dictionary used for replacement. format: {'column_name':{categorical_level: number_to_replace}}
	df_input -- (pd DataFrame) the input dataframe which contains the categorical features to be replaced.
	
	Return:
	-----------
	df_output -- (pd DataFrame) the output dataframe
	"""
	df_output = df_input.copy()
	for x in rep_dict.keys():
		df_output[x] = df_input[x].apply(lambda y: rep_dict[x][y])

	return df_output


def make_rolling_features(cols_roll, windows, agg_funcs, df_input):
	"""
	Make and add rolling features to the input dataframe given the columns, windows and aggregate function. 
	And record features created.

	Parameters:
	-----------
	cols_roll -- (list) a list of column names used to make rolling features
	windows -- (list) a list of int used as the window of making rolling features
	agg_funcs -- (list) a list of function names used as aggregate function for making rolling features
	df_input -- (pd DataFrame) the input dataframe which contains the features used for making rolling features
	
	Return:
	-----------
	cols_created_this_roll -- (list) a list of column names of the created rolling features
	df_output -- (pd DataFrame) the output dataframe containing all the rolling features
	"""
	cols_created_this_roll = []
	df_output = df_input.copy()
	for col in cols_roll:
		for t in windows:
			for fun in agg_funcs:
				new_col = col+'_last'+str(t)+'_'+fun
				cols_created_this_roll.append(new_col)
				df_output.loc[:, new_col] = getattr(df_input[col].rolling(t, min_periods=1), fun)().copy()

	return cols_created_this_roll, df_output


def make_ewm_features(cols_roll, ewm_alpha, df_input):
	"""
	Make mean ewm features based on cols_roll, ewm_alpha.
	Make std ewm features for 'playMin'. 
	Return list of column names containing all the ewm features created.

	Parameters:
	-----------
	cols_roll -- (list) a list of column names used to make ewm features
	ewm_alpha -- (list) a list of numbers used as the alpha for makign ewm features
	df_input -- (pd DataFrame) the input dataframe which contains the features used for making ewm features
	
	Return:
	-----------
	cols_created_ewm -- (list) a list of column names of the created ewm features
	df_output -- (pd DataFrame) the output dataframe containing all the ewm features
	"""
	cols_created_ewm = []
	df_output = df_input.copy()
	for col in cols_roll:
		for alpha in ewm_alpha:
			new_col_mean = col+'_ewm_0'+str(alpha-int(alpha))[2:] # create ewm feature name
			cols_created_ewm.append(new_col_mean)
			df_output.loc[:, new_col_mean] = df_input[col].ewm(alpha=alpha, min_periods=1).mean()
			
			if col == 'playMin':
				new_col_std = col+'_ewm_std_0'+str(alpha-int(alpha))[2:] # create ewm feature name
				df_output.loc[:, new_col_std] = df_input[col].ewm(alpha=alpha, min_periods=1).std()
				cols_created_ewm.append(new_col_std)
	
	return cols_created_ewm, df_output


def meaningful_shift(cols_created, cols_keep, df_input):
	"""
	Shift created features by 1 row so that it means the "last n games".
	
	Parameters:
	-----------
	cols_created -- (list) a list of column names containing all the created rolling and ewm features
	cols_keep -- (list) a list of names of the columns that should be kept in use without shifting
	df_input -- (pd DataFrame) the input dataframe for shifting
	
	Return:
	-----------
	df_output -- (pd DataFrame) the output dataframe containing unshifted cols_keep and shifted rolling and ewm features
	"""

	cols_created.append('gmDate')
	merge_temp = df_input[cols_created].copy().set_index('gmDate').shift(1, axis = 0).reset_index().copy()
	df_output = pd.merge(df_input[cols_keep], merge_temp, how='left', on='gmDate')

	return df_output


if __name__ == "__main__":
	main(opt["--input_path_file"], opt["--save_folder"])
