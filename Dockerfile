FROM continuumio/miniconda3

ENV PATH /opt/conda/bin:$PATH
ENV LANG C

COPY environment.yml .
RUN conda env create -f environment.yml

RUN apt-get update && apt-get install -y libxrender-dev vim

RUN mkdir /src
WORKDIR /src