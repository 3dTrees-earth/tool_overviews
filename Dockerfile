# IMPORTANT: as of May 2025, open3d is not available for python 3.13
# And there is also no chance to get the 3.12 running with linux, while mac works, though
# dockerfile
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential \
    libgl1 libegl1 libglib2.0-0 git xvfb \
 && ln -s /usr/bin/python3 /usr/local/bin/python \
 && python -m pip install --upgrade pip \
 && rm -rf /var/lib/apt/lists/*

# Optional: surfaceless EGL for headless Open3D
ENV EGL_PLATFORM=surfaceless
ENV MPLBACKEND=Agg

# Your Python deps
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

WORKDIR /src
COPY ./src /src

# Create a small entrypoint that starts Xvfb and forwards to the app
RUN install -D -m 0755 /dev/stdin /usr/local/bin/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
XVFB_DISPLAY="${XVFB_DISPLAY:-:99}"
XVFB_W="${XVFB_W:-1920}"
XVFB_H="${XVFB_H:-1080}"
XVFB_D="${XVFB_D:-24}"
XVFB_EXTRA_ARGS="${XVFB_EXTRA_ARGS:--ac +extension GLX +render -nolisten tcp}"

Xvfb "${XVFB_DISPLAY}" -screen 0 "${XVFB_W}x${XVFB_H}x${XVFB_D}" ${XVFB_EXTRA_ARGS} >/tmp/xvfb.log 2>&1 &
export DISPLAY="${XVFB_DISPLAY}"

exec "$@"
EOF

# Ensure LF endings (fix '/usr/bin/env: bash\r')
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "run.py"]