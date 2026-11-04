# The Robin worker image: `robin-worker` with every model and head runtime of
# robin-models (OWL, BirdNET, Perch, and ONNX heads), for linux/amd64 on CPU.
#
# Build it with `pixi run -e test worker-image`, which first builds Robin's own wheels
# into `.worker-image/dist`. See docker/README.md.
#
# The image holds no credential and no profile. A job gets its AWS login from its job
# role, and its task from the work file that its command names.

FROM python:3.13-slim

# Robin's own wheels are pure Python, so they are built on the host. pip gets every
# other dependency here, for this image's platform and Python.
COPY .worker-image/dist/ /tmp/robin-wheels/

# Each Robin package is given as its wheel file, never by name. A package given as a
# file has only that file as its candidate, also where another package requires it, so
# a package of the same name on PyPI cannot replace it.
RUN pip install --no-cache-dir \
        /tmp/robin-wheels/robin_contracts-*.whl \
        /tmp/robin-wheels/robin_inference_engine-*.whl \
        /tmp/robin-wheels/robin_worker-*.whl \
        "$(ls /tmp/robin-wheels/robin_adapters-*.whl)[ondio]" \
        "$(ls /tmp/robin-wheels/robin_models-*.whl)[owl,birdnet,perch,onnx-head]" \
    && pip check \
    && rm -rf /tmp/robin-wheels

# The worker writes its scratch files here.
WORKDIR /work

ENTRYPOINT ["robin-worker"]
