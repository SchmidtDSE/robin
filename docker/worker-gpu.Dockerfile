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
/usr/local/lib/python3.13/site-packages/nvidia/nvjitlink/lib:\
/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvcc/nvvm/lib64

# XLA compiles for the GPU with ptxas and libdevice, from the cuda_nvcc wheel. Perch
# is a jax2tf model with functions that XLA must compile, so on the GPU it needs them.
ENV PATH=/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvcc/bin:${PATH} \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvcc

# The NVIDIA container runtime mounts nvidia-smi only for the `utility` capability.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# The entrypoint logs the GPU's use once every ROBIN_GPU_LOG_SECONDS (60 by default; 0
# turns it off), as lines that start with `gpu:`, and then runs robin-worker.
COPY --chmod=755 <<'EOF' /usr/local/bin/robin-worker-gpu
#!/bin/sh
interval=${ROBIN_GPU_LOG_SECONDS:-60}
if [ "$interval" != 0 ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --format=csv,noheader -l "$interval" \
      --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
      | sed -u 's/^/gpu: /' &
  else
    echo "gpu: nvidia-smi is not available, so GPU use is not logged" >&2
  fi
fi
exec robin-worker "$@"
EOF
ENTRYPOINT ["robin-worker-gpu"]

# Check that each folder exists and holds a shared library. This does not import
# TensorFlow: an amd64 build on another machine cannot run TensorFlow's AVX code.
RUN python -c "\
import os, pathlib; \
folders = [pathlib.Path(f) for f in os.environ['LD_LIBRARY_PATH'].split(':')]; \
missing = [str(f) for f in folders if not any(f.glob('*.so*'))]; \
assert not missing, f'no CUDA library in: {missing}'; \
print(f'{len(folders)} CUDA library folders')"

# Check that XLA's tools are where XLA_FLAGS and PATH say.
RUN command -v ptxas \
    && test -f "${XLA_FLAGS#--xla_gpu_cuda_data_dir=}/nvvm/libdevice/libdevice.10.bc"
