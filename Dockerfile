FROM jjanzic/docker-python3-opencv:latest

COPY requirements.txt /
RUN pip install -r /requirements.txt

CMD ["python", "app.py"]