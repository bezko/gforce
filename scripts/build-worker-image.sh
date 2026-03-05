#!/bin/bash
# Build and push G-Force worker image to Google Artifact Registry

set -e

# Configuration
PROJECT_ID="${GFORCE_GCP_PROJECT:-$(gcloud config get-value project)}"
REGION="${GFORCE_GCP_REGION:-us-central1}"
REPOSITORY="gforce"
IMAGE_NAME="worker"
TAG="${1:-latest}"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: GCP project ID not set. Set GFORCE_GCP_PROJECT or run 'gcloud config set project <project-id>'"
    exit 1
fi

echo "Building G-Force worker image..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Tag: $TAG"

# Ensure Artifact Registry API is enabled
echo "Enabling Artifact Registry API..."
gcloud services enable artifactregistry.googleapis.com --project="$PROJECT_ID"

# Create repository if it doesn't exist
echo "Creating Artifact Registry repository if needed..."
if ! gcloud artifacts repositories describe "$REPOSITORY" \
    --location="$REGION" \
    --project="$PROJECT_ID" 2>/dev/null; then
    gcloud artifacts repositories create "$REPOSITORY" \
        --repository-format=docker \
        --location="$REGION" \
        --description="G-Force ML worker images"
fi

# Configure Docker authentication
echo "Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Build the image
echo "Building Docker image..."
docker build -t "${IMAGE_NAME}:${TAG}" -f Dockerfile .

# Tag for Artifact Registry
echo "Tagging image for Artifact Registry..."
docker tag "${IMAGE_NAME}:${TAG}" \
    "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

# Push to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

echo ""
echo "✓ Image built and pushed successfully!"
echo "Image URI: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"
