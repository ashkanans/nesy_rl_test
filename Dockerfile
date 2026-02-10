FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    python3-pip \
    git curl ca-certificates \
    build-essential pkg-config graphviz \
    patchelf libgl1-mesa-dev libosmesa6-dev libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Create separate venvs for Torch and JAX to avoid CUDA/cuDNN conflicts
ENV VENV_TORCH=/opt/venv-torch
ENV VENV_JAX=/opt/venv-jax

RUN python3.10 -m venv ${VENV_TORCH} \
    && python3.10 -m venv ${VENV_JAX}

# Default to Torch venv
ENV PATH="${VENV_TORCH}/bin:$PATH"

RUN ${VENV_TORCH}/bin/python -m pip install -U pip setuptools wheel \
    && ${VENV_JAX}/bin/python -m pip install -U pip setuptools wheel

ARG USERNAME=researcher
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace/nesy_rl

COPY requirements.txt ./requirements.txt
COPY trajectory-transformer ./trajectory-transformer
COPY implicit_q_learning ./implicit_q_learning
RUN ${VENV_TORCH}/bin/python -m pip install --no-cache-dir -r requirements.txt

# MuJoCo Python bindings (official)
RUN ${VENV_TORCH}/bin/python -m pip install --no-cache-dir mujoco

# MuJoCo 2.1 for mujoco-py (used by D4RL/IQL)
RUN mkdir -p /opt/mujoco210 \
    && curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    | tar -xz -C /opt
ENV MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210
ENV LD_LIBRARY_PATH=/opt/mujoco210/bin:${LD_LIBRARY_PATH}
ENV D4RL_SUPPRESS_IMPORT_ERROR=1
ENV MUJOCO_GL=egl

# Preinstall mujoco-py with Cython<3 to avoid build failures
RUN ${VENV_TORCH}/bin/python -m pip install --no-cache-dir \
        "Cython<3" glfw imageio cffi fasteners lockfile \
    && ${VENV_TORCH}/bin/python -m pip install --no-cache-dir --no-deps "mujoco-py==2.1.2.14"

# D4RL
RUN ${VENV_TORCH}/bin/python -m pip install --no-cache-dir "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"

# IQL (vendored) in JAX venv
RUN ${VENV_JAX}/bin/python -m pip install --no-cache-dir "jax[cuda12]==0.6.2"
RUN ${VENV_JAX}/bin/python -m pip install --no-cache-dir "numpy==1.26.4"
RUN ${VENV_JAX}/bin/python -m pip install --no-cache-dir gym==0.23.1 mujoco
RUN ${VENV_JAX}/bin/python -m pip install --no-cache-dir \
        "Cython<3" glfw imageio cffi fasteners lockfile \
    && ${VENV_JAX}/bin/python -m pip install --no-cache-dir --no-deps "mujoco-py==2.1.2.14"
RUN ${VENV_JAX}/bin/python -m pip install --no-cache-dir "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"
RUN ${VENV_JAX}/bin/python -m pip install --no-cache-dir --no-build-isolation \
    -r /workspace/nesy_rl/implicit_q_learning/requirements.txt

ENV PYTHONPATH=/workspace/nesy_rl:${PYTHONPATH}

RUN chown -R ${USERNAME}:${USERNAME} /opt/venv-torch /opt/venv-jax

USER ${USERNAME}
CMD ["/bin/bash"]
