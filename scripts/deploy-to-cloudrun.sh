#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Deploying Backend to Cloud Run ===${NC}\n"

# Configuration (can be overridden with environment variables)
SERVICE_NAME=${SERVICE_NAME:-"primed-live-hackathon"}
REGION=${REGION:-"us-central1"}
MEMORY=${MEMORY:-"1Gi"}
CPU=${CPU:-"1"}
TIMEOUT=${TIMEOUT:-"3600"}
CONCURRENCY=${CONCURRENCY:-"25"}
MIN_INSTANCES=${MIN_INSTANCES:-"0"}
MAX_INSTANCES=${MAX_INSTANCES:-"10"}
ENV_FILE=${ENV_FILE:-".env.yaml"}

# Check if .env.yaml exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: $ENV_FILE not found${NC}"
    echo "Please create $ENV_FILE with your environment variables"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No default project set${NC}"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Project:      ${YELLOW}${PROJECT_ID}${NC}"
echo -e "  Service:      ${YELLOW}${SERVICE_NAME}${NC}"
echo -e "  Region:       ${YELLOW}${REGION}${NC}"
echo -e "  Memory:       ${YELLOW}${MEMORY}${NC}"
echo -e "  CPU:          ${YELLOW}${CPU}${NC}"
echo -e "  Timeout:      ${YELLOW}${TIMEOUT}s${NC}"
echo -e "  Concurrency:  ${YELLOW}${CONCURRENCY}${NC}"
echo -e "  Min/Max:      ${YELLOW}${MIN_INSTANCES}/${MAX_INSTANCES}${NC}"
echo -e "  Env File:     ${YELLOW}${ENV_FILE}${NC}"

echo -e "\n${YELLOW}Starting deployment...${NC}\n"

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --timeout=$TIMEOUT \
  --concurrency=$CONCURRENCY \
  --min-instances=$MIN_INSTANCES \
  --max-instances=$MAX_INSTANCES \
  --memory=$MEMORY \
  --cpu=$CPU \
  --env-vars-file=$ENV_FILE

# Get the service URL
echo -e "\n${GREEN}Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format='value(status.url)')

echo -e "\n${GREEN}✓ Deployment successful!${NC}\n"
echo -e "${BLUE}Service URL:${NC} ${YELLOW}${SERVICE_URL}${NC}"
echo -e "${BLUE}API Docs:${NC}    ${YELLOW}${SERVICE_URL}/docs${NC}"
echo -e "${BLUE}Health:${NC}      ${YELLOW}${SERVICE_URL}/api/v1/health${NC}"

echo -e "\n${GREEN}Testing API...${NC}"
if curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/api/v1/health" | grep -q "200"; then
    echo -e "${GREEN}✓ API is responding!${NC}"
else
    echo -e "${YELLOW}⚠ API health check failed - may still be starting${NC}"
fi

echo -e "\n${BLUE}Next steps:${NC}"
echo -e "  1. Test your API: ${YELLOW}curl ${SERVICE_URL}/api/v1/health${NC}"
echo -e "  2. View docs: ${YELLOW}open ${SERVICE_URL}/docs${NC}"
