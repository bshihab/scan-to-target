# Scan to Target: pipeline + planning server. Research demo, not a medical device.
# Build:  docker build -t scan-to-target .
# Data:   mount a folder with data/ (scripts/fetch_data.sh) and out/
# Run:    docker run --rm -v "$PWD/data:/app/data" -v "$PWD/out:/app/out" scan-to-target subject sub-01 test
# Serve:  docker run --rm -p 127.0.0.1:8770:8770 -v "$PWD/out:/app/out" scan-to-target serve out/scenes/sub-01_ses-test --host 0.0.0.0
FROM python:3.13-slim
WORKDIR /app
RUN useradd --create-home planner
COPY pyproject.toml README.md ./
COPY s2t ./s2t
COPY viewer ./viewer
RUN pip install --no-cache-dir . && chown -R planner /app
USER planner
EXPOSE 8770
ENTRYPOINT ["s2t"]
CMD ["--help"]
