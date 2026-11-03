# The worker image

`worker.Dockerfile` builds the image that AWS Batch runs for each task: `robin-worker`
with the OWL model, for `linux/amd64` on CPU. It holds no credential and no profile.

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
```

Then register a new revision of the job definition with this image. See
[AWS Batch](../docs/aws-batch.md) for what a new revision does to runs in progress.
