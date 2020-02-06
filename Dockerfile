FROM debian:buster-slim

# Set the environment for conda with python3.7
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Add missing and outdated linux packages
RUN apt-get update --fix-missing && \
	apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
	apt-get clean

# Intall python and anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -O ~/anaconda.sh && \
	/bin/bash ~/anaconda.sh -b -p /opt/conda && \
	rm ~/anaconda.sh && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc && \
	find /opt/conda/ -follow -type f -name '*.a' -delete && \
	find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
	/opt/conda/bin/conda clean -afy

RUN apt-get update && apt install -y chromium && apt-get install -y libnss3 && apt-get install unzip

# Install R and it's dependencies
# RUN apt-get install r-base r-base-dev -y \ 
# 	Rscript -e "install.packages('tidyverse')" \
# 	Rscript -e "install.packages('docopt')"

# Install chromedriver
RUN wget -q "https://chromedriver.storage.googleapis.com/79.0.3945.36/chromedriver_linux64.zip" -O /tmp/chromedriver.zip \
	&& unzip /tmp/chromedriver.zip -d /usr/bin/ \
	&& rm /tmp/chromedriver.zip && chown root:root /usr/bin/chromedriver && chmod +x /usr/bin/chromedriver

# Install altair and selenium
RUN conda install -y -c conda-forge altair && conda install -y vega_datasets && conda install -y selenium

# COPY . /opt/app
# WORKDIR /opt/app

# Add the python required packages
RUN pip install pandas==0.25.2 \
	pip install numpy==1.17.2 \
	pip install docopt==0.6.2 \
	pip install requests==2.20.0 \
	pip install tqdm==4.41.1 \
	pip install selenium==3.141.0 \
	pip install altair==4.0.1 \
	pip install scikit-learn==0.22.1 \
	pip install matplotlib==3.1.2 \
	pip install plotly==4.3.0 \
	pip install selenium==3.141.0 \
	pip install termcolor==1.1.0 \
	pip install ChromeDriver==79.0.3945.36 \
	pip install jupyterlab==1.2.3 \
	pip install xgboost==0.90 \
	pip install lightgbm==2.3.1


CMD [ "/bin/bash" ]