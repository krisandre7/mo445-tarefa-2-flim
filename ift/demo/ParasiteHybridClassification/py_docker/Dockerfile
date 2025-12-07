FROM tensorflow/tensorflow:latest-gpu-py3

# Environment variable
#--- Avoid the keyboard configuration interative question
ENV DEBIAN_FRONTEND noninteractive

# Install some dependencies
RUN apt-get update && apt-get install -y \
		vim \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* 

# Install usefull python packages
RUN apt-get update && apt-get install -y \
		python3-numpy \
		python3-scipy \
		python3-sklearn \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install --upgrade ipython && \
	pip3 --no-cache-dir install \
                nvidia-ml-py3 \
		Pillow

