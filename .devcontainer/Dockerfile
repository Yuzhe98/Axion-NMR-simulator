FROM python:3.13-slim

# Install Git and other dependencies
RUN apt-get update && apt-get install -y git

# Install Times New Roman and other Microsoft core fonts
RUN apt-get update && apt-get install -y \
    ttf-mscorefonts-installer \
    fontconfig \
    && fc-cache -f -v


# # install fontconfig
# RUN apt-get update; apt-get install -y fontconfig

# # refresh system font cache
# RUN fc-cache -f -v

# Install Python packages
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    numpy \
    matplotlib \
    scipy\
    pandas \
    uncertainties \
    h5py \
    astropy \
    BaselineRemoval \
    attr \
    numba
WORKDIR /workspace
