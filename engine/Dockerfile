FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose default port (Railway will map the internal port)
EXPOSE 8000

# Run using python directly - main.py logic will handle PORT env var
CMD ["python", "main.py"]
