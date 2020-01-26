# author: Jarvis Nederlof, Roc Zhang
# date: 2020-01-25

"
This script downloads the data for the project from a remote repository.
It then saves the downloaded csv into a file. Both the repository url and
the file location to save the resulting csv are required as inputs.

Usage: 01-data_download.r --url=<url> --out_file=<out_file>

Options:
--url=<url>					URL to data file on remote repository.
--out_file=<out_file>		Path (including filename and output type) of where to save the file - the script only supports csv filetypes.

Example: Rscript scripts/01-data_download.r --url=https://raw.githubusercontent.com/jnederlo/nba_data/master/2012-18_playerBoxScore.csv --out_file=data/2012-18_playerBoxScore.csv
"-> doc

library(tidyverse)
library(docopt)

opt <- docopt(doc)

main <- function(url, out_file) {
	"
	Download a dataset from a remote repository and save to the `data` directory.
	"

	data <- read_csv(toString(url))
	print("Raw data successfully loaded!")
	  
	write_csv(data, path = toString(out_file))
	print("\nRaw data successfullly saved!")
}

main(opt["--url"], opt["--out_file"])
