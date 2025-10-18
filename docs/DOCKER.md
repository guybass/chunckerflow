## Docker Deployment Guide

### Quick Start

**Using Docker Compose (Recommended)**

```bash
# Start the API server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

The API will be available at `http://localhost:8000`

**Using Docker directly**

```bash
# Build image
docker build -t chunkflow:latest .

# Run container
docker run -p 8000:8000 chunkflow:latest
```

### Configuration

**Environment Variables**

Create a `.env` file (see `.env.example`):

```env
# API Keys
CHUNK_FLOW_OPENAI_API_KEY=sk-...
CHUNK_FLOW_GOOGLE_API_KEY=...

# Configuration
CHUNK_FLOW_LOG_LEVEL=INFO
CHUNK_FLOW_LOG_FORMAT=json
```

**Volume Mounts**

```yaml
volumes:
  # Configuration
  - ./config:/app/config:ro

  # Logs
  - ./logs:/app/logs

  # For development: live code reload
  - ./chunk_flow:/app/chunk_flow
```

### Production Deployment

**Build production image**

```bash
# Multi-stage build (optimized size)
docker build -t chunkflow:prod .

# Tag for registry
docker tag chunkflow:prod your-registry.com/chunkflow:latest

# Push to registry
docker push your-registry.com/chunkflow:latest
```

**Run in production**

```bash
docker run -d \
  --name chunkflow-api \
  -p 8000:8000 \
  -e CHUNK_FLOW_LOG_LEVEL=INFO \
  -e CHUNK_FLOW_LOG_FORMAT=json \
  -e CHUNK_FLOW_OPENAI_API_KEY=$OPENAI_API_KEY \
  --restart unless-stopped \
  chunkflow:prod
```

**Health Checks**

The Dockerfile includes a health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"
```

Check container health:

```bash
docker inspect --format='{{.State.Health.Status}}' chunkflow-api
```

### Kubernetes Deployment

**deployment.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chunkflow-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chunkflow
  template:
    metadata:
      labels:
        app: chunkflow
    spec:
      containers:
      - name: api
        image: your-registry.com/chunkflow:latest
        ports:
        - containerPort: 8000
        env:
        - name: CHUNK_FLOW_LOG_LEVEL
          value: "INFO"
        - name: CHUNK_FLOW_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: chunkflow-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

**service.yaml**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: chunkflow-api
spec:
  selector:
    app: chunkflow
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Monitoring

**View logs**

```bash
# Docker Compose
docker-compose logs -f api

# Docker
docker logs -f chunkflow-api

# Last 100 lines
docker logs --tail 100 chunkflow-api
```

**Resource usage**

```bash
docker stats chunkflow-api
```

**Access container**

```bash
docker exec -it chunkflow-api /bin/bash
```

### Security Best Practices

1. **Use secrets management**
   - Never hardcode API keys
   - Use environment variables or secret managers
   - Rotate keys regularly

2. **Run as non-root user**
   - Already configured in Dockerfile (user `chunkflow`)

3. **Limit resources**
   - Set memory and CPU limits
   - Prevent resource exhaustion

4. **Network security**
   - Use HTTPS in production
   - Configure CORS appropriately
   - Use firewall rules

5. **Keep images updated**
   - Regular security patches
   - Scan images for vulnerabilities

### Troubleshooting

**Container won't start**

```bash
# Check logs
docker logs chunkflow-api

# Check config
docker inspect chunkflow-api
```

**Health check failing**

```bash
# Test manually
docker exec chunkflow-api curl http://localhost:8000/health

# Check application logs
docker logs --tail 50 chunkflow-api
```

**High memory usage**

```bash
# Check stats
docker stats chunkflow-api

# Adjust memory limits
docker run -m 2g chunkflow:latest
```

**Performance issues**

- Increase workers: `CMD ["uvicorn", "chunk_flow.api.app:app", "--workers", "4"]`
- Use gunicorn: `gunicorn chunk_flow.api.app:app -w 4 -k uvicorn.workers.UvicornWorker`
- Enable caching (Redis)

### Advanced Configuration

**Multi-worker deployment**

```dockerfile
CMD ["gunicorn", "chunk_flow.api.app:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

**With Redis caching** (coming soon)

```yaml
services:
  api:
    depends_on:
      - redis
    environment:
      - CHUNK_FLOW_REDIS_URL=redis://redis:6379

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
```

### References

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Production Checklist](https://github.com/tiangolo/fastapi/blob/master/deployment.md)
