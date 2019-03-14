FROM pytorch/pytorch:0.4_cuda9_cudnn7
LABEL maintainer="nishimura"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libgtk2.0-dev \
        language-pack-ja-base \
        language-pack-ja \
        libblas-dev \
        liblapack-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install \
	torch torchvision \
	pydensecrf \
    opencv-python \
    imageio \
    scikit-image \
    keras \
    gputil \
    hyperdash \
    scipy \
    tqdm \
    jupyter_contrib_nbextensions \
    jupyter_nbextensions_configurator\
    pydot \
    staintools==0.1.1\
    pulp \
    tqdm



