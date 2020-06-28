FROM pityka/base-ubuntu-libtorch:3

RUN apt install -y openjdk-8-dbg
WORKDIR /opt
COPY . .