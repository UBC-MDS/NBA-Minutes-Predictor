# author: Jarvis Nederlof
# date: 2020-02-06
# This file builds a Docker container to run the program from start to end for project reproducibility.
# The docker container is hosted on dockerhub here --> https://hub.docker.com/repository/docker/jnederlo/nba_minutes/general
# For usage instructions refer to the project README.

# Get base image from trusted source - versioned for reproducibility of languages and packages
FROM continuumio/anaconda3:2019.10

# Install chromium, zip tools, and latex
RUN apt-get update && \
	apt install -y chromium && \
	apt-get install -y libnss3 && \
	apt-get install unzip && \
	apt-get install -y texlive texlive-xetex texlive-fonts-recommended texlive-generic-recommended texlive-latex-extra cm-super

# Install chromedriver
RUN wget -q "https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip" -O /tmp/chromedriver.zip \
	&& unzip /tmp/chromedriver.zip -d /usr/bin/ \
	&& rm /tmp/chromedriver.zip && chown root:root /usr/bin/chromedriver && chmod +x /usr/bin/chromedriver

# Add the python required packages from conda
RUN conda install -c conda-forge lightgbm==2.3.0 && \
	conda install -c conda-forge xgboost==0.90 && \
	conda install -y -c conda-forge altair==4.0.1 && \
	conda install -y vega_datasets==0.7.0 && \
	conda install -y selenium==3.141.0 && \
	conda install -c anaconda docopt==0.6.2

# Add the python required packages not available in conda
RUN pip install tqdm==4.41.1 && \
	pip install termcolor==1.1.0

# Install R and the R packages with conda
RUN conda install -y -c r r-tidyverse==1.2.1 && \
	conda install -y -c conda-forge r-docopt==0.6.1

CMD [ "/bin/bash" ]