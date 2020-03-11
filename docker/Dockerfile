FROM ubuntu:18.04

RUN apt-get update -y
RUN apt-get install -y git python3-pip curl
RUN python3 -m pip install --upgrade pip pytest-cov nbval pycodestyle \
      git+git://github.com/ubermag/ubermagutil.git \
      git+git://github.com/ubermag/discretisedfield.git \
      git+git://github.com/ubermag/micromagneticmodel.git

COPY . /usr/local/micromagnetictests/
WORKDIR /usr/local/micromagnetictests
RUN python3 -m pip install .
