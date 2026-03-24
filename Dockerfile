FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for building quant libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter/dashboard ports (optional)
EXPOSE 8888 8000

# Default command
CMD ["python"]