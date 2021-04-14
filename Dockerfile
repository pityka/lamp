FROM pityka/base-ubuntu-libtorch:torch181

# RUN apt install -y openjdk-8-dbg
WORKDIR /opt
COPY . .