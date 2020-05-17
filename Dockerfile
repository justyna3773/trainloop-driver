FROM python:3.6

ENV CODE_DIR /root/code
WORKDIR $CODE_DIR

RUN apt-get -y update && apt-get -y install ffmpeg \
  && mkdir -p $CODE_DIR

COPY requirements.txt $CODE_DIR
RUN pip3 install -r requirements.txt

COPY . $CODE_DIR
RUN pip3 install -e .

CMD /bin/bash
