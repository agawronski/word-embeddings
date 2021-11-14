FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en
EXPOSE 8501
EXPOSE 80
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
