FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start on port 8000 (Railway networking is configured for this port)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
