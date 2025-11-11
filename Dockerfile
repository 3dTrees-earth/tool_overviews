# Base image with CUDA support
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential git \
    libgl1 libglvnd0 libglx0 libegl1 libgles2 libglib2.0-0 \
 && ln -s /usr/bin/python3 /usr/local/bin/python \
 && python -m pip install --upgrade pip \
 && rm -rf /var/lib/apt/lists/*

RUN pip install \
    geopandas \
    "laspy[lazrs,laszip]" \
    numpy==1.23.5 \
    open3d==0.18.0 \
    shapely \
    pyproj \
    matplotlib \
    imageio \
    pydantic \
    pydantic-settings \
    tqdm \
    pytest

# Ensure EGL headless rendering and expose graphics capability
ENV OPEN3D_RENDERING_PROVIDER=egl
ENV EGL_PLATFORM=surfaceless
ENV MPLBACKEND=Agg
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

# Redirect cache directories to /tmp (always writable regardless of user)
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV XDG_CONFIG_HOME=/tmp/.config
ENV FONTCONFIG_CACHE=/tmp/.fontconfig
ENV MPLCONFIGDIR=/tmp/.matplotlib

WORKDIR /src
COPY ./src /src

# No ENTRYPOINT - let Galaxy control the command execution
CMD ["python", "run.py", "--help"]