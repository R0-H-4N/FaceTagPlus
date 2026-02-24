# Docker Setup Guide

## Prerequisites
- Docker
- Docker Compose

## Environment Variables Setup

The application requires environment variables for security credentials. These are **NOT** included in the repository.

### Step 1: Create .env file

Copy the example file and configure your credentials:

```bash
cp .env.example .env
```

### Step 2: Edit .env file

Open `.env` and set your values:

```env
# Database Credentials (Required)
DB_USER=your_database_username
DB_USER_PASS=your_secure_database_password

# JWT Secret (Required - MUST be changed!)
# Generate a secure random string at least 32 characters long
JWT_SECRET=your_random_secure_string_min_32_chars

# Optional
NODE_ENV=production
```

**⚠️ IMPORTANT:** 
- Never commit the `.env` file to git (it's already in `.gitignore`)
- Change `JWT_SECRET` to a unique, random string
- Use strong passwords for production

### Step 3: Generate a secure JWT_SECRET

You can generate a secure JWT secret using:

```bash
# Using OpenSSL
openssl rand -base64 48

# Using Node.js
node -e "console.log(require('crypto').randomBytes(48).toString('base64'))"

# Using Python
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Step 4: Validate your configuration

Run the validation script to check your setup:

```bash
./check-env.sh
```

This will verify that all required variables are set and properly configured.

## Running the Application

Once your `.env` is configured:

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

## Services

- **Frontend**: http://localhost (port 80)
- **Backend API**: http://localhost:3000
- **ML Model API**: http://localhost:8000

## Environment Variable Validation

Docker Compose will fail with an error if required variables are missing:

```
ERROR: The DB_USER variable is not set.
ERROR: The DB_USER_PASS variable is not set.
ERROR: The JWT_SECRET variable is not set.
```

If you see these errors, ensure your `.env` file exists and contains all required variables.

## Stopping the Application

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f model
```
