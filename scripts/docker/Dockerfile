# CairnIQ — Docker Image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# libxml2-dev / libxslt1-dev let lxml>=6 build from source if no wheel matches.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (user_data excluded via .dockerignore)
COPY . .

# Create required directories
RUN mkdir -p logs/{agent,chat_runtime,frontend,server,tools} \
    user_data/{cache,embeddings,profiles,daily_cache} \
    tmp

# Expose the FastAPI server port
EXPOSE 8000

# Run the FastAPI server
CMD ["python", "server.py"]
