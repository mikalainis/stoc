FROM python:3.10-slim

# Force logs to show in Cloud Logging immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy logic
COPY main.py .

# Run application
CMD ["python", "main.py"]