# ========================
# Dockerfile for ProspectIQ AI Dataset Generator
# ========================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy script
COPY prospectiq_dataset_ai_full.py /app/

# Default command (will be overridden by Makefile or CLI args)
ENTRYPOINT ["python", "prospectiq_dataset_ai_full.py"]

