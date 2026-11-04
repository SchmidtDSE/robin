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
TensorFlow runs on the CPU and gives no error. The GPU image is about three times the
size of the CPU image, so CPU jobs use the CPU image.

On a Mac with Apple Silicon, these images run under emulation, which cannot run
TensorFlow. To check that TensorFlow sees the GPU, run a task on a GPU queue (see
[AWS Batch](../docs/aws-batch.md)), and look in the job's log for TensorFlow's
`Created device ... GPU:0` line. A "Cannot dlopen some GPU libraries" line means that
TensorFlow runs on the CPU. A Batch job cannot override the image's entrypoint, so the
check is a real task, not a separate `python` command.

## Push it to ECR

Give the image a tag that names its build, such as the commit, so that each job
definition revision names one image. Use your own account, region, and repository.

```sh
ACCOUNT=123456789012
REGION=us-west-2
REPOSITORY=robin-worker
TAG=$(git rev-parse --short HEAD)
REGISTRY=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com

aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $REGISTRY
docker tag robin-worker:dev $REGISTRY/$REPOSITORY:$TAG
docker push $REGISTRY/$REPOSITORY:$TAG

# The GPU image, under its own tag.
docker tag robin-worker:dev-gpu $REGISTRY/$REPOSITORY:$TAG-gpu
docker push $REGISTRY/$REPOSITORY:$TAG-gpu
```

Then register a new revision of the job definition with this image. See
[AWS Batch](../docs/aws-batch.md) for what a new revision does to runs in progress.
