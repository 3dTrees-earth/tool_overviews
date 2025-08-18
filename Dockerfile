# IMPORTANT: as of May 2025, open3d is not available for python 3.13
# And there is also no chance to get the 3.12 running with linux, while mac works, though
FROM python:3.11


RUN apt-get update && apt-get install -y && \
    pip install --upgrade pip && \
    # undocumented dependency for open3d - try to install but don't fail if not available
    apt install -y libegl1 libgl1 libgomp1 || true && \
    apt install -y libgl1-mesa-glx || true

# These are the needed packages for running the overview tool
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

ENV EGL_PLATFORM=surfaceless

RUN mkdir -p /src && mkdir -p /in && mkdir -p /out
COPY ./src /src

# Debugging dependencies
RUN pip install ipython \
    debugpy


WORKDIR /src
CMD ["python", "run.py"]
