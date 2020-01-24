# author: Jack Tan
# date: 2020-01-24

"""
This script takes processed data from the 'data' folder in the project repostiory
and creates various models to predict the 'playMin' feature using other features.
Types of model produced includes: baseline, linear regression, XGBoost and LGBM
Afterwards, the scripts test the models and calculates the MSE and coefficient of
determination. Residual plots are created for all models and a feature importance
plot is created for the GBM model.

Both the file name and the save folder are required as inputs.
Usage: model_fit.py --file_name=<file_name> --save_folder=<save_folder>
Options:
--file_name=<file_name>         file name of the processed features and targets
--save_folder=<save_folder>	    Folder to save all figures and csv files produced
Example: python data_download.py --file_name=player_data_ready.csv --out_file=img
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
# Creating classes


class baseline_model:
    def predict(self, X):
        predictions = X.loc[:, 'playMin_last5_median']
        return predictions

opt = docopt(__doc__)

def main(file_name, save_folder):
    # Load the processed data from csv
    # e.g. 'player_data_ready.csv'
    path = str('../data/' + file_name)
    try:
        all_data = pd.read_csv(path)
        print('Data loaded successfully!')
    except:
        print('Path to file is not valid!')
        raise
    # Removing columns that can't be interpretted
    all_data = all_data.drop(
        columns=['playDispNm', 'gmDate', 'teamAbbr', 'playPos'])


    assert 'playMin' in all_data.columns, 'No targets found'
    print('Test 1 passed!')
    # Splitting data into training and testing
    X, y = all_data.loc[:, all_data.columns != 'playMin'], all_data['playMin']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=100,
                                                        test_size=0.25)
    print('Finished Splitting Data')

    # LGBM MODEL:
    gbm = lgb.LGBMRegressor(num_leaves=31,
                            learning_rate=0.1,
                            n_estimators=60)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            verbose=False)
    print('Finished Fitting GBM Model')

    # XGBoost MODEL:
    params = {'n_estimators': 60,
              'max_depth': 5,
              'booster': 'gbtree',  # gbtree or dart seem best
              'learning_rate': .1,
              'gamma': 0,  # default 0 - larger gamma means more conservative model
              'reg_lambda': 1,  # default 1 - L2 regularization
              'reg_alpha': 0,  # default 0 - L1 regularization
              'objective': 'reg:squarederror',
              'eval_metric': ['rmse']}
    xgb_model = XGBRegressor(**params).fit(X_train, y_train)
    print('Finished Fitting XGB Model')

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X=X_train, y=y_train)
    print('Finished Fitting Linear Regression Model')

    # Baseline Model
    base_model = baseline_model()

    # Calculating test predictions for different models
    gbm_test = gbm.predict(X=X_test)
    xgb_test = xgb_model.predict(data=X_test)
    lr_test = lr_model.predict(X=X_test)
    base_test = base_model.predict(X=X_test)

    # Calculating MSE and r2 score for different models
    gbm_mse = mean_squared_error(y_test, gbm_test)
    gbm_r2 = r2_score(y_test, gbm_test)
    assert gbm_mse > 0, 'Mean squared error should be greater than 0!'
    print('Test 2 passed!')
    xgb_mse = mean_squared_error(y_test, xgb_test)
    xgb_r2 = r2_score(y_test, xgb_test)
    lr_mse = mean_squared_error(y_test, lr_test)
    lr_r2 = r2_score(y_test, lr_test)
    base_mse = mean_squared_error(y_test, base_test)
    base_r2 = r2_score(y_test, base_test)

    # Saving results as csv file
    my_df = pd.DataFrame(data={'Baseline': [base_mse, base_r2],
                               'Linear Regression': [lr_mse, lr_r2],
                               'XGB': [xgb_mse, xgb_r2],
                               'LGBM': [gbm_mse, gbm_r2]},
                         index=['MSE', 'Coefficient of Determination'])
    try:
        my_df.to_csv(str('../' + save_folder + '/score_table.csv'))
        print('Finished Calculating & Saving Prediction Errors')
    except:
        print('Save folder is not valid!')
        raise

    # Calculating residuals for different models
    residual_gbm = gbm_test - y_test
    residual_xgb = xgb_test - y_test
    residual_lr = lr_test - y_test
    residual_base = base_test - y_test
    base_df = pd.DataFrame(data={'x': base_test, 'y': residual_base}).groupby(
        ['x']).mean().reset_index()
    xgb_df = pd.DataFrame(
        data={'x': xgb_test, 'y': residual_xgb}).sort_values(by='x')
    gbm_df = pd.DataFrame(
        data={'x': gbm_test, 'y': residual_gbm}).sort_values(by='x')
    lr_df = pd.DataFrame(data={'x': lr_test, 'y': residual_lr}).sort_values(by='x')

    # Making dataframes of mean residuals over bins
    xgb_preds = pd.melt(xgb_df,
                        id_vars=['x'],
                        value_vars=['y'],
                        value_name='Mean Residual')

    # Bin the dataframe with a 0.1 bin_size on the actuals
    bins = np.arange(0, np.max(xgb_df.loc[:, 'x']), .1)
    xgb_binned = xgb_preds.groupby(pd.cut(xgb_preds['x'], bins,
                                          labels=False), as_index=False).mean()

    gbm_preds = pd.melt(gbm_df,
                        id_vars=['x'],
                        value_vars=['y'],
                        value_name='Mean Residual')

    # Bin the dataframe with a 0.1 bin_size on the actuals
    bins = np.arange(0, np.max(gbm_df.loc[:, 'x']), .1)
    gbm_binned = gbm_preds.groupby(pd.cut(gbm_preds['x'], bins,
                                          labels=False), as_index=False).mean()

    lr_preds = pd.melt(lr_df,
                       id_vars=['x'],
                       value_vars=['y'],
                       value_name='Mean Residual')

    # Bin the dataframe with a 0.1 bin_size on the actuals
    bins = np.arange(0, np.max(lr_df.loc[:, 'x']), .1)
    lr_binned = lr_preds.groupby(pd.cut(lr_preds['x'], bins,
                                        labels=False), as_index=False).mean()


    # Making figure for residual plots
    fig = make_subplots(rows=4, cols=1, subplot_titles=("Baseline Model",
                                                        "XGB Model",
                                                        "Linear Regression Model",
                                                        "GBM Model"))

    fig.add_trace(
        go.Scatter(x=base_df.loc[:, 'x'], y=base_df.loc[:, 'y'], mode='markers'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=xgb_binned.loc[:, 'x'], y=xgb_binned.loc[:, 'Mean Residual'], mode='markers'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=lr_binned.loc[:, 'x'],
                   y=lr_binned.loc[:, 'Mean Residual'], mode='markers'),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=gbm_binned.loc[:, 'x'], y=gbm_binned.loc[:, 'Mean Residual'], mode='markers'),
        row=4, col=1
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Model Prediction", row=4, col=1)

    # Update yaxis properties
    fig.update_yaxes(title_text="Residual", range=[-5, 5], row=1, col=1)
    fig.update_yaxes(title_text="Residual", range=[-5, 5], row=2, col=1)
    fig.update_yaxes(title_text="Residual", range=[-5, 5], row=3, col=1)
    fig.update_yaxes(title_text="Residual", range=[-5, 5], row=4, col=1)


    fig.update_layout(height=1000, width=800, showlegend=False)
    fig.write_image(str('../' + save_folder + '/residual_plot.png'))
    print('Finished Creating Residual Plots')

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
    gbm_features.save(str('../' + save_folder + '/gbm_importance.png'))


if __name__ == "__main__":
	main(opt["--file_name"], opt["--save_folder"])
