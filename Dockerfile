# Scan to Target: pipeline + planning server. Research demo, not a medical device.
# Build:   docker build -t scan-to-target .
# Data:    scripts/fetch_data.sh on the host, then mount data/ and out/ (paths are relative to /work)
# Run:     docker run --rm -v "$PWD/data:/work/data" -v "$PWD/out:/work/out" scan-to-target subject sub-01 test
# Check:   docker run --rm -v "$PWD/data:/work/data" -v "$PWD/out:/work/out" --entrypoint python scan-to-target validation/b1_dicom_vs_dcm2niix.py
# Serve:   docker run --rm -p 127.0.0.1:8770:8770 -v "$PWD/out:/work/out" scan-to-target serve out/scenes/sub-01_ses-test --host 0.0.0.0
#          (published on the host's loopback only; the server's Host check allows 127.0.0.1:8770 and localhost:8770)
FROM python:3.13-slim
RUN useradd --create-home planner
WORKDIR /src
COPY pyproject.toml README.md ./
COPY s2t ./s2t
RUN pip install --no-cache-dir .
WORKDIR /work
COPY validation ./validation
COPY scripts ./scripts
RUN chown -R planner /work
USER planner
EXPOSE 8770
ENTRYPOINT ["s2t"]
CMD ["--help"]
