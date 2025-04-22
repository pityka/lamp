FROM pityka/base-ubuntu-libtorch:torch260_amd64

RUN  apt update  && apt-get install -y curl
WORKDIR /opt
RUN curl -L https://github.com/sbt/sbt/releases/download/v1.10.11/sbt-1.10.11.tgz | tar xzf -  
ENV PATH="$PATH:/opt/sbt/bin"
RUN sbt --allow-empty sbtVersion
COPY . .