#!/bin/bash

# Deployment script for Titanic ML API
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-production}
DOCKER_IMAGE="titanic-ml-api"
DOCKER_TAG="latest"

echo "🚀 Deploying Titanic ML API to $ENVIRONMENT environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -f Dockerfile.api -t $DOCKER_IMAGE:$DOCKER_TAG .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down || true

# Start the new deployment
echo "▶️ Starting new deployment..."
docker-compose up -d

# Wait for health check
echo "⏳ Waiting for service to be healthy..."
timeout=60
counter=0

while [ $counter -lt $timeout ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        break
    fi
    echo "⏳ Waiting for service to be ready... ($counter/$timeout)"
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -eq $timeout ]; then
    echo "❌ Service failed to become healthy within $timeout seconds"
    docker-compose logs
    exit 1
fi

# Run smoke tests
echo "🧪 Running smoke tests..."
if curl -f http://localhost:8000/docs > /dev/null 2>&1; then
    echo "✅ API documentation is accessible"
else
    echo "❌ API documentation is not accessible"
    exit 1
fi

# Test prediction endpoint
echo "🧪 Testing prediction endpoint..."
PREDICTION_RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "passenger": {
            "passenger_id": 1,
            "pclass": 1,
            "sex": "female",
            "age": 29,
            "sibsp": 0,
            "parch": 0,
            "fare": 211.3375,
            "embarked": "S"
        }
    }')

if echo "$PREDICTION_RESPONSE" | grep -q "prediction"; then
    echo "✅ Prediction endpoint is working"
else
    echo "❌ Prediction endpoint failed"
    echo "Response: $PREDICTION_RESPONSE"
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo "📊 API is available at: http://localhost:8000"
echo "📚 Documentation at: http://localhost:8000/docs"
echo "🏥 Health check at: http://localhost:8000/health"

# Show container status
echo "📋 Container status:"
docker-compose ps 