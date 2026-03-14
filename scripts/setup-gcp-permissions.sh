#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== GCP Cloud Run Permissions Setup ===${NC}\n"

# Get project ID (use parameter or current gcloud config)
if [ -n "$1" ]; then
    PROJECT_ID=$1
    echo -e "Using provided project ID: ${YELLOW}${PROJECT_ID}${NC}"
else
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: No project ID provided and no default project set${NC}"
        echo "Usage: ./setup-gcp-permissions.sh [PROJECT_ID]"
        exit 1
    fi
    echo -e "Using current gcloud project: ${YELLOW}${PROJECT_ID}${NC}"
fi

# Get project number
echo -e "\n${GREEN}Getting project number...${NC}"
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
echo -e "Project Number: ${YELLOW}${PROJECT_NUMBER}${NC}"

# Enable required APIs
echo -e "\n${GREEN}Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable storage.googleapis.com --project=$PROJECT_ID
echo -e "${GREEN}✓ APIs enabled${NC}"

# Grant permissions to Cloud Build service account
echo -e "\n${GREEN}Granting permissions to Cloud Build service account...${NC}"
CLOUD_BUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo -e "  ${YELLOW}→${NC} Granting Cloud Run Admin role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${CLOUD_BUILD_SA}" \
  --role="roles/run.admin" \
  --quiet > /dev/null

echo -e "  ${YELLOW}→${NC} Granting Service Account User role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${CLOUD_BUILD_SA}" \
  --role="roles/iam.serviceAccountUser" \
  --quiet > /dev/null

echo -e "  ${YELLOW}→${NC} Granting Storage Admin role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${CLOUD_BUILD_SA}" \
  --role="roles/storage.admin" \
  --quiet > /dev/null

# Grant permissions to Compute service account
echo -e "\n${GREEN}Granting permissions to Compute service account...${NC}"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo -e "  ${YELLOW}→${NC} Granting Storage Object Viewer role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/storage.objectViewer" \
  --quiet > /dev/null

echo -e "\n${GREEN}✓ All permissions granted successfully!${NC}"
echo -e "\n${GREEN}You can now deploy your app with:${NC}"
