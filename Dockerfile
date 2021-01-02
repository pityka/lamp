FROM pityka/base-ubuntu-libtorch:torch171

# RUN apt install -y openjdk-8-dbg
WORKDIR /opt
COPY . .