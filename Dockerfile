FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    git curl ca-certificates \
    build-essential pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create venv (so pip installs don't fight system python)
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install -U pip setuptools wheel

ARG USERNAME=researcher
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace/nesy_rl

COPY requirements.txt ./requirements.txt
COPY trajectory-transformer ./trajectory-transformer
RUN python -m pip install --no-cache-dir -r requirements.txt

# MuJoCo Python bindings (official)
RUN python -m pip install --no-cache-dir mujoco

# D4RL
RUN python -m pip install --no-cache-dir "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"

# IQL (official JAX repo)
RUN git clone https://github.com/ikostrikov/implicit_q_learning /opt/implicit_q_learning \
    && python -m pip install --no-cache-dir -r /opt/implicit_q_learning/requirements.txt

# JAX with GPU support (match to CUDA in the base image)
# For CUDA 12.x, JAX docs recommend jax[cuda12]
RUN python -m pip install --no-cache-dir "jax[cuda12]"

ENV PYTHONPATH=/workspace/nesy_rl:${PYTHONPATH}

USER ${USERNAME}
CMD ["/bin/bash"]
