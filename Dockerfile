# Get base image from trusted source.
FROM continuumio/anaconda3:2019.10

# # Install R
# RUN apt-get update && apt-get install r-base r-base-dev -y

# Install chromium and zip tools
RUN apt-get update && apt install -y chromium && apt-get install -y libnss3 && apt-get install unzip

# Install chromedriver
RUN wget -q "https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip" -O /tmp/chromedriver.zip \
	&& unzip /tmp/chromedriver.zip -d /usr/bin/ \
	&& rm /tmp/chromedriver.zip && chown root:root /usr/bin/chromedriver && chmod +x /usr/bin/chromedriver

# Add the python required packages from conda
RUN conda install -c conda-forge lightgbm==2.3.0 && \
	conda install -c conda-forge xgboost==0.90 && \
	conda install -y -c conda-forge altair==4.0.1 && \
	conda install -y vega_datasets==0.7.0 && \
	conda install -y selenium==3.141.0

# Add the python required packages not available in conda
RUN pip install docopt==0.6.2 && \
	pip install tqdm==4.41.1 && \
	pip install termcolor==1.1.0

# Install the R packages with conda
RUN conda install -y -c r r-tidyverse==1.2.1 && \
	conda install -y -c conda-forge r-docopt==0.6.1

CMD [ "/bin/bash" ]