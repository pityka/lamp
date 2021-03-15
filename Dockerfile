FROM pityka/base-ubuntu-libtorch:torch180

# RUN apt install -y openjdk-8-dbg
WORKDIR /opt
COPY . .