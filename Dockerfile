FROM tensorflow/tensorflow:2.7.0-gpu

RUN pip install keras numpy pandas matplotlib sklearn opencv-python-headless tables

WORKDIR /app/penguins

ENTRYPOINT /bin/bash

#ENTRYPOINT python penguindetector.py

