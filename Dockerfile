# Dcokerfile, Image, Container
FROM python:3.8

WORKDIR /app

ADD . /app

RUN echo "backend : Agg" >> matplotlibrc \
    && echo "font.family : Ricty Diminished" >> matplotlibrc

RUN pip install -r requirements.txt

CMD python main.py