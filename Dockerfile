# https://hub.docker.com/r/jupyter/base-notebook/dockerfile

FROM python:3.7-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade && \
    apt-get install -yq --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ENV WRK_DIR "/work"
ENV PKG_DIR="$WRK_DIR/pyxdh"

RUN mkdir $WRK_DIR
WORKDIR $WRK_DIR

RUN mkdir -p $PKG_DIR && \
    git clone https://github.com/ajz34/Py_xDH.git $PKG_DIR && \
    cp $PKG_DIR/.pyscf_conf.py ~/.pyscf_conf.py

RUN pip --no-cache-dir install -r $PKG_DIR/requirements.txt
RUN pip --no-cache-dir install -r $PKG_DIR/docs/requirements.yml
RUN pip --no-cache-dir install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install
RUN for extension in \
        nbextensions_configurator/config_menu/main \
        hide_input/main \
        scratchpad/main \
        collapsible_headings/main \
        toc2/main \
        nbextensions_configurator/tree_tab/main \
        codefolding/edit \
        jupyter-js-widgets/extension; \
    do jupyter nbextension enable $extension --sys-prefix; done
RUN mkdir -p ~/.jupyter/custom && \
    echo ".container { width:85%; }" > ~/.jupyter/custom/custom.css

ENV PYTHONPATH $PKG_DIR:$PYTHONPATH

EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --allow-root
