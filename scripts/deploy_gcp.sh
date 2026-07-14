#!/usr/bin/env bash
# GCP deployment script for ED Orchestrator
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-ed-orchestrator-dev}"
REGION="${GCP_REGION:-us-central1}"
TAG="${1:-latest}"

echo "Building and deploying to project ${PROJECT_ID}..."

gcloud config set project "${PROJECT_ID}"

gcloud artifacts repositories create ed-orchestrator \
  --repository-format=docker \
  --location="${REGION}" 2>/dev/null || true

docker build -f infra/docker/Dockerfile.api \
  -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/ed-orchestrator/api:${TAG}" .

docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/ed-orchestrator/api:${TAG}"

gcloud run deploy ed-orchestrator-api \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/ed-orchestrator/api:${TAG}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "USE_GCP=true,GCP_PROJECT=${PROJECT_ID}" \
  --memory 2Gi \
  --cpu 2

docker build -f infra/docker/Dockerfile.dashboard \
  -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/ed-orchestrator/dashboard:${TAG}" .

docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/ed-orchestrator/dashboard:${TAG}"

API_URL=$(gcloud run services describe ed-orchestrator-api --region "${REGION}" --format='value(status.url)')

gcloud run deploy ed-orchestrator-dashboard \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/ed-orchestrator/dashboard:${TAG}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "ED_API_URL=${API_URL}" \
  --memory 1Gi

echo "Deployment complete. API: ${API_URL}"
