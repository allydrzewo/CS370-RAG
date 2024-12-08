FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the app port
EXPOSE 8000

# Command to run the app
CMD ["python", "main.py"]
