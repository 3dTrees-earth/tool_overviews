# tool_overviews
Galaxy implementation of the Overviews tool

# build the docker image
docker build -t tool-overviews:latest -f Dockerfile

# run the docker image
docker run --rm --gpus all \
  -v <input_dir>:/data:ro \
  -v <output_dir>:/out \
  tool-overviews:latest \
  python -u run.py \
  --dataset-path <you_las.laz> \
  --output-dir /out \
  --top-views-deg 90 \
  --cmap viridis