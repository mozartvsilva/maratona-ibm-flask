FROM python:3.6-alpine

COPY . /flask
WORKDIR /flask

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ./scripts/run.sh
