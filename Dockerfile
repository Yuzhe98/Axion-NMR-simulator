# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Metadata indicating an image maintainer.
LABEL maintainer="yuhzhang@uni-mainz.de"

# Install Git and other dependencies
RUN apt-get update && apt-get install -y git

## Pip dependencies
# Upgrade pip
RUN pip install --upgrade pip
# Install production dependencies
WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt




