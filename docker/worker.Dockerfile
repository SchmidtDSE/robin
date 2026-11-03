# The Robin worker image: `robin-worker` and the OWL model, for linux/amd64 on CPU.
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

RUN pip install --no-cache-dir --find-links /tmp/robin-wheels \
        robin-worker \
        robin-inference-engine \
        "robin-adapters[ondio]" \
        "robin-models[owl]" \
    && pip check \
    && rm -rf /tmp/robin-wheels

# The worker writes its scratch files here.
WORKDIR /work

ENTRYPOINT ["robin-worker"]
