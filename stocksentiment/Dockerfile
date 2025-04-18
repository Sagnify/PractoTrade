# # Use official Python image
# FROM python:3.12.9-slim

# # Set working directory inside the container
# WORKDIR /app

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy project files into the container
# COPY . .

# # Expose port (Django/Gunicorn will run on this)
# EXPOSE 8000

# # Start server
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "stocksentiment.wsgi:application"]


# Use official Python image
FROM python:3.12-slim
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Expose port (Django/Gunicorn will run on this)
EXPOSE 8000

# Start server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "stocksentiment.wsgi:application"]
