#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔍 Checking Docker environment setup..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found${NC}"
    echo ""
    echo "To fix this:"
    echo "  1. Copy the example: cp .env.example .env"
    echo "  2. Edit .env and set your credentials"
    echo "  3. See DOCKER_SETUP.md for details"
    exit 1
fi

echo -e "${GREEN}✓${NC} .env file exists"

# Load .env file
export $(cat .env | grep -v '^#' | xargs)

# Check required variables
MISSING=0

if [ -z "$DB_USER" ]; then
    echo -e "${RED}❌ DB_USER is not set${NC}"
    MISSING=1
else
    echo -e "${GREEN}✓${NC} DB_USER is set"
fi

if [ -z "$DB_USER_PASS" ]; then
    echo -e "${RED}❌ DB_USER_PASS is not set${NC}"
    MISSING=1
else
    echo -e "${GREEN}✓${NC} DB_USER_PASS is set"
fi

if [ -z "$JWT_SECRET" ]; then
    echo -e "${RED}❌ JWT_SECRET is not set${NC}"
    MISSING=1
else
    if [ ${#JWT_SECRET} -lt 32 ]; then
        echo -e "${YELLOW}⚠${NC} JWT_SECRET is set but too short (${#JWT_SECRET} chars, minimum 32)"
        echo "   Generate a secure one with:"
        echo "   openssl rand -base64 48"
        MISSING=1
    else
        echo -e "${GREEN}✓${NC} JWT_SECRET is set (${#JWT_SECRET} chars)"
    fi
fi

echo ""

if [ $MISSING -eq 1 ]; then
    echo -e "${RED}Configuration incomplete. Please fix the issues above.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ All required environment variables are properly configured!${NC}"
    echo ""
    echo "You can now run:"
    echo "  docker-compose up --build"
    exit 0
fi
