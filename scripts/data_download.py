# author: Jarvis Nederlof
# date: 2020-01-15

"""
This script downloads the data for the project from a remote repository.
It then saves the downloaded csv into a file. Both the repository url and
the file location to save the resulting csv are required as inputs.

Usage: data_download.py --url=<url> --out_file=<out_file>

Options:
--url=<url>					URL to data file on remote repository.
--out_file=<out_file>		Path (including filename and output type) of where to save the file - the script only supports csv filetypes.

Example: python data_download.py --url=https://raw.githubusercontent.com/jnederlo/nba_data/master/2012-18_playerBoxScore.csv --out_file=../data/2012-18_playerBoxScore.csv

"""
import pandas as pd
import requests
import os
from docopt import docopt

opt = docopt(__doc__)

def main(url, out_file):
	"""
	Download a dataset from a remote repository and save to the `data` directory.
	"""
	try:
		request = requests.get(url)
		request.status_code == 200
	except Exception as req:
		print("Bad url entered.")
		print(req)
	
	df = pd.read_csv(url)

	try:
		df.to_csv(out_file, index=False)
	except:
		os.makedirs(os.path.dirname(out_file))
		df.to_csv(out_file, index=False)


if __name__ == "__main__":
	main(opt["--url"], opt["--out_file"])