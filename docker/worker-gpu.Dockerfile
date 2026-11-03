# The Robin worker image for GPU: the CPU image plus the CUDA and cuDNN libraries that
# TensorFlow needs, for linux/amd64.
#
# Build it with `pixi run -e test worker-image-gpu`, which builds the CPU image first.
# See docker/README.md.

ARG BASE=robin-worker:dev
FROM ${BASE}

# The same TensorFlow version as the CPU image, so pip adds only the nvidia-* wheels.
RUN TENSORFLOW=$(python -c "import importlib.metadata as m; print(m.version('tensorflow'))") \
    && pip install --no-cache-dir "tensorflow[and-cuda]==${TENSORFLOW}" \
    && pip check

# TensorFlow finds the CUDA libraries of the nvidia-* wheels only through
# LD_LIBRARY_PATH. Without it, TensorFlow runs on the CPU and gives no error.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_cupti/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cufft/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/curand/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cusolver/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cusparse/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nccl/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/nvjitlink/lib

# Check that each folder exists and holds a shared library. This does not import
# TensorFlow: an amd64 build on another machine cannot run TensorFlow's AVX code.
RUN python -c "\
import os, pathlib; \
folders = [pathlib.Path(f) for f in os.environ['LD_LIBRARY_PATH'].split(':')]; \
missing = [str(f) for f in folders if not any(f.glob('*.so*'))]; \
assert not missing, f'no CUDA library in: {missing}'; \
print(f'{len(folders)} CUDA library folders')"
