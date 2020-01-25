# Author: Jack Tan
# Contributer: Jarvis Nederlof
# Date: 2020-01-25

"""
This script takes processed data from the 'data' folder in the project repostiory
and creates various models to predict the 'playMin' feature using other features.
Types of model produced includes: baseline, linear regression, XGBoost and LGBM.
Afterwards, the scripts test the models and calculates the MSE and coefficient of
determination on test set. Residual plots are created for all models and a feature
importance plot is created for the GBM model. These figures are then saved accordingly.

Both the file name and the save folder are required as inputs.

Usage: 04-model_fit.py --file_name=<file_name> --save_folder=<save_folder>

Options:
--file_name=<file_name>         File name of the processed features and targets
--save_folder=<save_folder>	Folder to save all figures and csv files produced

Example: python scripts/04-model_fit.py --file_name=player_data_ready.csv --save_folder=results
"""

# Loading the required packages
# Models
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
# Plotting
import altair as alt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# Numerical Packages
import numpy as np
import pandas as pd
# SKLearn Packages
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Binary Model Save
from pickle import dump
from docopt import docopt
# Other Packages
from termcolor import colored
import sys
import os
# Ignore warnings from packages in models
import warnings
warnings.simplefilter("ignore")

opt = docopt(__doc__)

def main(file_name, save_folder):
	# Load the processed data from csv
	# e.g. 'player_data_ready.csv'
	
	print(colored("\nWARNING: This script takes about 1 minute to run\n", 'yellow'))
	
	# Validate the file-path to load file
	path_str = str('../data/' + file_name)
	if os.path.exists(path_str) == False:
		path_str = str('data/' + file_name)
	try:
		data = pd.read_csv(path_str)
		print(colored('Data loaded successfully!', 'green'))
	except:
		print(colored('ERROR: Path to file is not valid!', 'red'))
		raise

	# Validate the save_foler directory exists or make folder
	if os.path.exists(save_folder) == False:
		if os.path.exists(str('../' + save_folder) == False):
			try:
				os.makedirs(save_folder)
			except:
				print(colored('ERROR: Path to save directory is not valid!', 'red'))
				raise

	# Preprocess the Data
	X_train, y_train, X_test, y_test = preprocess(data)

	# Fit the models
	lgb = lgbm_model(X_train, y_train, X_test, y_test)
	xgb = xgboost_model(X_train, y_train, X_test, y_test)
	lm = linear_model(X_train, y_train, X_test, y_test)
	base = baseline_model()

	# Setup plot figure
	fig = make_subplots(rows=4, cols=1, subplot_titles=("GBM Model",
														"XGB Model",
														"Linear Regression Model",
														"Baseline Model"))

	# Get the predictions and score each model
	results = {}
	for i, model in enumerate([(lgb, 'lgbm'), (xgb, 'xgboost'), (lm, 'linaer regression'), (base, 'base model')]):
		# Get the predictions
		preds = predictions(model[0], X_test)
		# Get the scoring metrics
		results[model[1]] = scoring(preds, y_test)
		# Calculate the model residuals
		resid_df = calc_residuals(preds, y_test, model[1])
		# Add to plot figure
		fig.add_trace(go.Scatter(x=resid_df.loc[:, 'x'],
								 y=resid_df.loc[:, 'Mean Residual'],
								 mode='markers'),
								 row=i+1,
								 col=1)

	# Get the model results into a dataframe
	df = pd.DataFrame(data=results, index=['MSE', 'Coefficient of Determination'])
	
	# Save the results dataframe
	save_results(df, save_folder)

	# Save the plotting figure
	plot_figure(fig, save_folder)

	# Save the feature importance from LGBM model
	feature_importance(lgb, X_test, save_folder)

class baseline_model:
	def predict(self, X):
		predictions = X.loc[:, 'playMin_last5_median']
		return predictions

def preprocess(data):
	"""
	Preprocess the data by dropping certain columns not suitable to train on.
	Split the data into training and testing splits.

	Return a tuple of the training and testing splits.

	Parameters:
	-----------
	data -- (pd DataFrame) The loaded data.
	"""
	# Removing columns that can't be interpretted
	data = data.drop(
		columns=['playDispNm', 'gmDate', 'teamAbbr', 'playPos'])

	# Test that target is in data
	assert 'playMin' in data.columns, 'No targets found'
	# print('Test 1 passed!')
	
	# Splitting data into training and testing
	X, y = data.loc[:, data.columns != 'playMin'], data['playMin']
	X_train, X_test, y_train, y_test = train_test_split(X,
														y,
														random_state=100,
														test_size=0.25)
	return X_train, y_train, X_test, y_test

def lgbm_model(X_train, y_train, X_test, y_test):
	"""
	Initialize and fit the LGBM model.

	Return the fitted model.
	
	Parameters:
	-----------
	X_train -- (pd DataFrame) The training data
	y_train -- (pd DataFrame) The training target
	X_test -- (pd DataFrame) The testing data
	Y_test -- (pd DataFrame) The testing target
	"""
	print(colored("\nTraining LGBM Model", 'cyan'))
	# LGBM MODEL:
	gbm = lgb.LGBMRegressor(num_leaves=31,
							learning_rate=0.1,
							n_estimators=60)
	gbm.fit(X_train, y_train,
			eval_set=[(X_test, y_test)],
			eval_metric='l2',
			verbose=False)
	print(colored('Finished Training LGBM Model\n', 'green'))
	return gbm


def xgboost_model(X_train, y_train, X_test, y_test):
	"""
	Initialize and fit the XG Boost model.

	Return the fitted model.
	
	Parameters:
	-----------
	X_train -- (pd DataFrame) The training data
	y_train -- (pd DataFrame) The training target
	X_test -- (pd DataFrame) The testing data
	Y_test -- (pd DataFrame) The testing target
	"""
	print(colored("Training XGBoost Model", 'cyan'))
	# XGBoost MODEL:
	params = {'n_estimators': 60,
			  'max_depth': 5,
			  'booster': 'gbtree', # gbtree or dart seem best
			  'learning_rate': .1,
			  'gamma': 0,  # default 0 - larger gamma means more conservative model
			  'reg_lambda': 1,  # default 1 - L2 regularization
			  'reg_alpha': 0,  # default 0 - L1 regularization
			  'objective': 'reg:squarederror',
			  'eval_metric': ['rmse'],
			  'verbosity': 0}
	xgb = XGBRegressor(**params).fit(X_train, y_train)
	print(colored('Finished Training XGBoost Model\n', 'green'))
	return xgb

def linear_model(X_train, y_train, X_test, y_test):
	"""
	Initialize and fit the Linear Regression model.

	Return the fitted model.
	
	Parameters:
	-----------
	X_train -- (pd DataFrame) The training data
	y_train -- (pd DataFrame) The training target
	X_test -- (pd DataFrame) The testing data
	Y_test -- (pd DataFrame) The testing target
	"""
	print(colored("Training Linear Regression Model", 'cyan'))
	# Linear Regression Model
	lr_model = LinearRegression()
	lr_model.fit(X=X_train, y=y_train)
	print(colored('Finished Training Linear Regression Model\n', 'green'))
	return lr_model

def predictions(model, X_test):
	"""
	Call the predict method on each model.

	Return the model's predictions as an array.

	Parameters:
	-----------
	model -- (model object) the trained model
	X_test -- (pd DataFrame) the testing data
	"""
	return model.predict(X_test)

def scoring(pred, y_test):
	"""
	Get the mean squared error and the r2 score for each
	model's predictions.

	Return the scoring metrics in a tuple.

	Parameters:
	-----------
	model -- (model object) the trained model
	X_test -- (pd DataFrame) the testing data
	"""
	# Calculating MSE and r2 score for different models
	mse = round(mean_squared_error(y_test, pred), 2)
	r2 = round(r2_score(y_test, pred), 2)
	assert mse > 0, 'Mean squared error should be greater than 0!'
	# print('Test 2 passed!')
	return mse, r2

def calc_residuals(pred, y_test, model_name):
	"""
	Get the residual error scores for each trained model's predictions.

	Return a dataframe of the model's binned residuals.

	Parameters:
	-----------
	pred -- (list) the model's predictions
	y_test -- (pd DataFrame) the testing targets
	model_name -- (str) the name of the model
	"""
	# Calculating residuals for different models
	residual = pred - y_test
	if model_name == 'base model':
		df = pd.DataFrame(data={'x': pred, 'y': residual}).groupby(['x']).mean().reset_index()
	else:
		df = pd.DataFrame(data={'x': pred, 'y': residual}).sort_values(by='x')
	preds_df = pd.melt(df,
					   id_vars=['x'],
					   value_vars=['y'],
					   value_name='Mean Residual')
	
	# Bin the dataframe with a 0.1 bin_size on the actuals
	bins = np.arange(0, np.max(df.loc[:, 'x']), .1)
	preds_binned = preds_df.groupby(pd.cut(preds_df['x'], 
									bins,
									labels=False), 
									as_index=False).mean()
	return preds_binned

def save_results(df, save_folder):
	"""
	Save the model results metrics into a csv file.

	Parameters:
	-----------
	df -- (pd DataFrame) the models results metrics
	save_folder -- (str) the directory to save the results in
	"""
	try:
		df.to_csv(str('../' + save_folder + '/modelling-score_table.csv'))
	except:
		df.to_csv(str(save_folder + '/modelling-score_table.csv'))
		# print(colored('ERROR: Save folder is not valid!', 'red'))
		# raise
	print(colored(f'Saved Model Results in /{save_folder} directory', 'green'))

def plot_figure(fig, save_folder):
	"""
	Save the residual plots figures.

	Parameters:
	-----------
	fig -- (plotly figure object) the plotly residuals figure
	save_folder -- (str) the directory to save the results in
	"""
	# Update xaxis properties
	fig.update_xaxes(title_text="Model Prediction", row=4, col=1)

	# Update yaxis properties
	fig.update_yaxes(title_text="Residual", range=[-5, 5], row=1, col=1)
	fig.update_yaxes(title_text="Residual", range=[-5, 5], row=2, col=1)
	fig.update_yaxes(title_text="Residual", range=[-5, 5], row=3, col=1)
	fig.update_yaxes(title_text="Residual", range=[-5, 5], row=4, col=1)


	fig.update_layout(height=1000, width=800, showlegend=False)
	try:
		fig.write_image(str('../' + save_folder + '/modelling-residual_plot.png'))
	except:
		fig.write_image(str(save_folder + '/modelling-residual_plot.png'))
	print(colored(f'\nSaved Residuals Plot in /{save_folder} directory', 'green'))

def feature_importance(gbm, X_test, save_folder):
	"""
	Parameters:
	-----------
	gbm -- (model object) the LGBM model
	X_test -- (pd DataFrame) the testing data
	save_folder -- (str) the directory to save the results in
	"""
	# Feature Importance Plot
	feature_df = pd.DataFrame()
	feature_df['features'] = list(X_test.columns)
	feature_df['importance'] = gbm.feature_importances_
	feature_df.sort_values(by=['importance'], ascending=False, inplace=True)

	# Plot the feature importance
	gbm_features = alt.Chart(feature_df).mark_bar().encode(
		x='importance:Q',
		y=alt.Y('features:N', sort=alt.EncodingSortField(
			field='features', op='count', order='ascending'))
	).properties(
		title='Importance of Different Features'
	)
	try:
		gbm_features.save(str('../' + save_folder + '/modelling-gbm_importance.png'), scale_factor=5.0)
	except:
		gbm_features.save(str(save_folder + '/modelling-gbm_importance.png'), scale_factor=1)
	print(colored(f'\nSaved Features Importance Plot in /{save_folder} directory', 'green'))

if __name__ == "__main__":
	main(opt["--file_name"], opt["--save_folder"])
