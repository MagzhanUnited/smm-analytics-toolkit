#!/usr/bin/env bash
# Bring up the local CI/CD stack: kind cluster, kubeconfig for Jenkins, Jenkins + registry.
set -euo pipefail
cd "$(dirname "$0")"
root=../..
kind get clusters | grep -qx smm-ci || kind create cluster --config "$root/deploy/kind-config.yaml"
# internal kubeconfig: Jenkins reaches the API server over the shared "kind" Docker network
kind get kubeconfig --name smm-ci --internal > kubeconfig
[ -f .env ] || echo "JENKINS_ADMIN_PASSWORD=$(openssl rand -hex 12)" > .env
docker compose up -d --build
echo "Jenkins: http://localhost:8080  (user admin, password in ci/jenkins/.env)"
