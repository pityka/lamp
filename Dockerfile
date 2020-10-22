FROM pityka/base-ubuntu-libtorch:torch16

# RUN apt install -y openjdk-8-dbg
WORKDIR /opt
COPY . .