FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command for testing
CMD ["python3", "-c", "from lib.config import Config; c = Config(); print('✅ Config loaded successfully'); print(f'Current preset: {c.MODEM_PRESET}'); print(f'Preset details: {c.current_preset}')"]