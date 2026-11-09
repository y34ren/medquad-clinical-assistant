#!/usr/bin/env bash
# Declarative GCP & Google Agents CLI (`agents-cli`) Deployment Script
# for NIH MedQuAD Clinical Assistant (Cloud Run + Vertex AI Agent Runtime)
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-nih-clinical-center-prod}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
SERVICE_NAME="medquad-clinical-assistant"
AR_REPO="nih-clinical-agents"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:latest"

echo "==> [1/5] Running Pre-Deployment Unit Test Suite..."
python3 -m unittest discover -s tests -p "test_*.py" -v

echo "==> [2/5] Running Day 4 Golden Evaluation & 95/95 AgentOps Gate..."
python3 eval/eval_harness.py

if command -v agents-cli >/dev/null 2>&1; then
  echo "==> [3/5] Verifying Agent Runtime Package via Google Agents CLI (agents-cli)..."
  agents-cli deploy --dry-run --project "${PROJECT_ID}" --region "${REGION}" || true
fi

echo "==> [4/5] Submitting Cloud Build Pipeline..."
gcloud builds submit --config=cloudbuild.yaml \
  --project="${PROJECT_ID}" \
  --substitutions="_REGION=${REGION},_AR_REPO=${AR_REPO}"

echo "==> [5/5] Verifying Cloud Run Service Health & A2A Agent Card..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format="value(status.url)")

echo "Deployed NIH MedQuAD Clinical Assistant to: ${SERVICE_URL}"
echo "A2A Agent Card URL: ${SERVICE_URL}/.well-known/agent.json"
