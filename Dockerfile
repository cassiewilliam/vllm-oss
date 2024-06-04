# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

#################### MAIN IMAGE ####################
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS dev

# # Set the DEBIAN_FRONTEND variable to noninteractive to avoid interactive prompts
# ENV DEBIAN_FRONTEND=noninteractive

# # Preconfigure tzdata for US Central Time (build running in us-central-1 but this really doesn't matter.)
# RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
#     && echo 'tzdata tzdata/Zones/America select Chicago' | debconf-set-selections

# We install an older version of python here for testing to make sure vllm works with older versions of Python.
# For the actual openai compatible server, we will use the latest version of Python.
RUN apt-get update -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update -y \
    && apt-get install -y python3.10 python3.10-dev python3.10-venv

# RUN apt-get update -y \
#     && apt-get install -y python3.10 python3.10-dev

# Download and install pip for Python 3.10
RUN apt-get install -y curl && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py

RUN ln -fs /usr/bin/python3.10 /usr/bin/python && ln -fs /usr/bin/python3.10 /usr/bin/python3

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.2/compat/

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# install build dependencies
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

