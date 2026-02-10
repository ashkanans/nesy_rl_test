FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    git curl ca-certificates \
    build-essential pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=researcher
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace/nesy_rl

RUN python3 -m pip install -U pip setuptools wheel

# Only copy requirements for caching
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# MuJoCo Python bindings (official)
RUN python3 -m pip install --no-cache-dir mujoco

# D4RL
RUN python3 -m pip install --no-cache-dir "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"

# IQL (official JAX repo)
RUN git clone https://github.com/ikostrikov/implicit_q_learning /opt/implicit_q_learning \
    && python3 -m pip install --no-cache-dir -r /opt/implicit_q_learning/requirements.txt

# JAX with GPU support (match to CUDA in the base image)
# For CUDA 12.x, JAX docs recommend jax[cuda12]
RUN python3 -m pip install --no-cache-dir "jax[cuda12]"

# (No COPY . .) — you’ll mount the repo at runtime
ENV PYTHONPATH=/workspace/nesy_rl:${PYTHONPATH}

USER ${USERNAME}
CMD ["/bin/bash"]
