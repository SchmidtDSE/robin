# The worker image

`worker.Dockerfile` builds the image that AWS Batch runs for each task: `robin-worker`
with every model and head runtime of `robin-models` (OWL, BirdNET, Perch, and ONNX
heads), for `linux/amd64` on CPU. `worker-gpu.Dockerfile` builds the GPU
image on top of it. Neither holds a credential or a profile.

## Build it

Docker must be running.

```sh
pixi run -e test worker-image
```

This builds Robin's own wheels into `.worker-image/dist`, and then builds the image
`robin-worker:dev`. pip installs every other dependency inside the build, for
`linux/amd64` and Python 3.13, and the build ends with `pip check`.

To check the image:

```sh
docker run --rm --platform linux/amd64 robin-worker:dev --help
docker run --rm --platform linux/amd64 robin-worker:dev output-versions
```

## The GPU image

```sh
pixi run -e test worker-image-gpu
```

This builds the CPU image first, and then `robin-worker:dev-gpu`: the CPU image with
the CUDA and cuDNN libraries that TensorFlow needs (`tensorflow[and-cuda]`, at the
same TensorFlow version). `LD_LIBRARY_PATH` names their folders: without it,
TensorFlow runs on the CPU and gives no error. `PATH` and `XLA_FLAGS` name `ptxas` and
`libdevice`, which XLA needs to compile for the GPU. Perch has functions that XLA must
compile. The GPU image is about three
times the size of the CPU image, so CPU jobs use the CPU image.

The GPU image's entrypoint logs the GPU's use with `nvidia-smi` once a minute, as lines
that start with `gpu:`, and then runs `robin-worker`. At the same interval, it logs the
container's memory, now and at its peak so far, as lines that start with `memory:`. The
values come from the container's cgroup, so they include the page cache that the
container uses. Set `ROBIN_GPU_LOG_SECONDS` in the job definition to change the
interval, or to `0` to turn both off.

On a Mac with Apple Silicon, these images run under emulation, which cannot run
TensorFlow. To check that TensorFlow sees the GPU, run a task on a GPU queue (see
[AWS Batch](../docs/aws-batch.md)), and look in the job's log for TensorFlow's
`Created device ... GPU:0` line. A "Cannot dlopen some GPU libraries" line means that
TensorFlow runs on the CPU. A model runs on the GPU only when the setup's resources give
`device: gpu` (see [AWS Batch](../docs/aws-batch.md#gpu)); the `gpu:` lines then show
its use above 0 %. For Perch, also look for an `XLA service ... initialized for platform
CUDA` line. A Batch job cannot override the image's entrypoint, so the
check is a real task, not a separate `python` command.

## Push it to a registry

`worker-push` builds both images from the current commit and pushes them as
`<commit>` and `<commit>-gpu`, so that each job definition revision names one image.
It refuses a working tree with uncommitted changes, because the tag would then not
name the code in the image.

Give the repository once, as a line in `.env` (git-ignored). The task reads only this
line:

```sh
ROBIN_IMAGE_REPOSITORY=123456789012.dkr.ecr.us-west-2.amazonaws.com/robin-worker
```

Then push. For an ECR repository, the task logs in with your AWS login, in the
region that the address names. Set `AWS_PROFILE` if your login is not the default one.
For another registry, run `docker login` first.

```sh
pixi run -e test worker-push
```

An argument, `pixi run -e test worker-push <repository>`, takes precedence, then
`ROBIN_IMAGE_REPOSITORY` from the environment, then the line in `.env`. The task runs
`docker/push-worker-images.sh`, and prints the two tags for the job definitions.

Then register a new revision of the job definition with this image. See
[AWS Batch](../docs/aws-batch.md) for what a new revision does to runs in progress.
