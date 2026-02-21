#!/bin/bash
# build_image.sh
set -e

VERSION=${1:-dev}

# Ensure builder exists and is active
if ! docker buildx ls | grep -q "iris-builder"; then
    echo "Creating iris-builder..."
    docker buildx create --name iris-builder --driver docker-container --use
else
    echo "Using existing iris-builder"
    docker buildx use iris-builder
fi

echo "Building IRIS Qwen image: $VERSION"

docker buildx build --platform linux/amd64 \
  --cache-from=type=registry,ref=registry.rcp.epfl.ch/iris-qwen/iris-qwen:buildcache \
  --cache-to=type=registry,ref=registry.rcp.epfl.ch/iris-qwen/iris-qwen:buildcache,mode=max \
  --tag registry.rcp.epfl.ch/iris-qwen/iris-qwen:"${VERSION}" \
  --build-arg LDAP_GROUPNAME=rcp-runai-aiteam_AppGrpU \
  --build-arg LDAP_GID=84800 \
  --build-arg LDAP_USERNAME=mhamelin \
  --build-arg LDAP_UID=258812 \
  --push \
  .

echo "✓ Build complete: registry.rcp.epfl.ch/iris-qwen/iris-qwen:${VERSION}"
