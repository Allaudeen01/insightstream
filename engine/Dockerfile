FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default port (Railway will override via environment)
ENV PORT=8000

# Use shell form with explicit /bin/sh to expand $PORT
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
