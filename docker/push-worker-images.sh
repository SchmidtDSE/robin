#!/usr/bin/env bash
# Builds both worker images from HEAD and pushes them as <commit> and <commit>-gpu.
#
#   pixi run -e test worker-push [repository]
#
# The repository is the argument, else ROBIN_IMAGE_REPOSITORY from the environment,
# else the ROBIN_IMAGE_REPOSITORY line of .env. No other line of .env is read.
#
# For an ECR repository, the script logs in with your AWS login (set AWS_PROFILE if it
# is not the default one), in the region that the address names. For another registry,
# run `docker login` first. See docker/README.md.
set -euo pipefail

cd "$(dirname "$0")/.."

repository=${1:-${ROBIN_IMAGE_REPOSITORY:-}}
if [[ -z $repository && -f .env ]]; then
  line=$(grep -m1 '^ROBIN_IMAGE_REPOSITORY=' .env || true)
  repository=${line#ROBIN_IMAGE_REPOSITORY=}
fi
if [[ -z $repository ]]; then
  echo "Give a repository, or set ROBIN_IMAGE_REPOSITORY in .env or the environment." >&2
  exit 1
fi

if ! git diff --quiet HEAD; then
  echo "Commit your changes first: the image tag names a commit." >&2
  exit 1
fi
tag=$(git rev-parse --short HEAD)

registry=${repository%%/*}
if [[ $registry == *.dkr.ecr.*.amazonaws.com ]]; then
  region=$(cut -d. -f4 <<<"$registry")
  echo "Logging in to $registry with your AWS login."
  aws ecr get-login-password --region "$region" \
    | docker login --username AWS --password-stdin "$registry"
fi

pixi run -e test worker-image-gpu

docker tag robin-worker:dev "$repository:$tag"
docker push "$repository:$tag"
docker tag robin-worker:dev-gpu "$repository:$tag-gpu"
docker push "$repository:$tag-gpu"

echo
echo "Pushed. For terraform.tfvars:"
echo "cpu_image_tag = \"$tag\""
echo "gpu_image_tag = \"$tag-gpu\""
