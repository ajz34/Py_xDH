# https://hub.docker.com/r/jupyter/base-notebook/dockerfile

FROM python:3.8-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade && \
    apt-get install -yq --no-install-recommends git gcc && \
    rm -rf /var/lib/apt/lists/*

ENV WRK_DIR "/work"
ENV PKG_DIR="$WRK_DIR/pyxdh"

RUN mkdir $WRK_DIR
WORKDIR $WRK_DIR

RUN mkdir -p $PKG_DIR
RUN git clone https://github.com/ajz34/Py_xDH.git $PKG_DIR
RUN cd $PKG_DIR/ && git checkout --force origin/legacy
RUN cp $PKG_DIR/.pyscf_conf.py ~/.pyscf_conf.py

RUN pip --no-cache-dir install -r $PKG_DIR/requirements.txt

ENV PYTHONPATH $PKG_DIR:$PYTHONPATH

EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --allow-root
