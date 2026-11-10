#!/usr/bin/env bash
# Build the image, create a local kind cluster, deploy the service and test it.
# Requirements: docker, kind, kubectl.  Usage: bash k8s/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=claim-frequency:v1
CLUSTER=claims

docker build -t "$IMAGE" .

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --config k8s/kind-config.yaml
fi
kind load docker-image "$IMAGE" --name "$CLUSTER"

EXISTED=$(kubectl --context "kind-$CLUSTER" get deployment claim-frequency -o name 2>/dev/null || true)
kubectl --context "kind-$CLUSTER" apply -f k8s/deployment.yaml
# Same image tag on a re-deploy: restart the pods so they pick up the new image
[ -n "$EXISTED" ] && kubectl --context "kind-$CLUSTER" rollout restart deployment/claim-frequency
kubectl --context "kind-$CLUSTER" rollout status deployment/claim-frequency --timeout=180s
kubectl --context "kind-$CLUSTER" get deploy,pods,svc -l app=claim-frequency

echo "Testing http://localhost:30096 ..."
uv run python predict_test.py --url http://localhost:30096
