FROM jjanzic/docker-python3-opencv:opencv-4.1.0

COPY requirements.txt /
RUN pip install -r /requirements.txt

CMD ["python", "app.py"]