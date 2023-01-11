# Base image
# FROM python:3.9-slim # without cuda
FROM nvcr.io/nvidia/pytorch:22.12-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy important stuff
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# Set working directory and install reqs
WORKDIR /
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Which application do we want to run when the docker image is executed (train_model.py). The "-u" makes sure any print statements get redirected to the terminal.
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

