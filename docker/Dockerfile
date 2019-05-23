FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    apt install -y libsm6 libxext6 && \
    # software-properties-common requried to be able to use add-apt-repository
    apt-get install -y apt-utils git python-virtualenv python3-dev python3-pip software-properties-common python-software-properties python-opencv

RUN apt-get clean autoclean && \
    apt-get autoremove -y

RUN apt-get install python3-pip cmake zlib1g-dev python3-tk python-opencv -y
RUN apt-get install libboost-all-dev -y
RUN apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran -y
RUN apt-get install libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev -y
RUN apt-get install dpkg-dev build-essential python3.5-dev libjpeg-dev  libtiff-dev libsdl1.2-dev libnotify-dev freeglut3 freeglut3-dev libsm-dev libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libgtk-3-dev libwebkitgtk-3.0-dev libgstreamer-plugins-base1.0-dev -y
RUN apt-get install libav-tools libsdl2-dev swig cmake -y
RUN apt-get install iputils-ping mongodb-clients -y

# Apt cleaning
RUN apt-get clean autoclean && \
    apt-get autoremove -y

# set environment variables
ENV PATH $PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/lib:$LD_LIBRARY_PATH

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision opencv-python scikit-image lmdb gym[atari] pycolab pymongo sacred GitPython visdom_observer
RUN pip3 install tensorflow-gpu
RUN pip3 install jupyter ipdb tqdm matplotlib pandas

RUN git clone https://github.com/openai/baselines && \
	cd baselines && \
	pip3 install -e . && \
	cd ..

ADD . graphrl
WORKDIR 'graphrl'
RUN pip3 install -e .
