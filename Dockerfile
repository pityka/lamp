FROM pityka/base-ubuntu-libtorch:torch190-jdk17

# RUN apt install -y openjdk-8-dbg
RUN apt-get update --fix-missing && apt-get install -y software-properties-common wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && apt-get update
RUN apt-get install -y nsight-compute-2021.1.1 nsight-systems-2021.1.3
WORKDIR /opt
COPY . .