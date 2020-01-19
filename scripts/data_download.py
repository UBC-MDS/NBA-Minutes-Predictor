# author: Jarvis Nederlof
# date: 2020-01-15
#
# This script downloads the data for the project from a remote repository.
# The script takes no arguments.
#
# Usage: python data_download.py

import pandas as pd

def main():
	"""
	Download a dataset from a remote repository and save to the `data` directory.
	"""
	df = pd.read_csv('https://raw.githubusercontent.com/jnederlo/nba_data/master/2012-18_playerBoxScore.csv')
	df.to_csv('../data/2012-18_playerBoxScore.csv')


if __name__ == "__main__":
	main()