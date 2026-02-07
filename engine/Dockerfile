FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the application directly with Python
# Run uvicorn using shell form to ensure $PORT expansion works
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --proxy-headers
