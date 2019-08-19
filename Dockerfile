FROM python:3.7-slim

RUN apt-get update
RUN apt-get -y install git

ENV APP_DIR "/app"
ENV WRK_DIR "/app/pyxdh"

RUN mkdir -p $WRK_DIR
RUN git clone https://github.com/ajz34/Py_xDH.git $WRK_DIR
RUN cp $WRK_DIR/.pyscf_conf.py ~/.pyscf_conf.py
RUN pip install -r $WRK_DIR/requirements.txt
RUN pip install -r $WRK_DIR/docs/requirements.yml

ENV PYTHONPATH $WRK_DIR:$PYTHONPATH

EXPOSE 8888

CMD cd $APP_DIR; jupyter notebook --ip=0.0.0.0 --allow-root
