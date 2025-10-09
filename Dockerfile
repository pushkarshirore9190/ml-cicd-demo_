# Use official Python image as base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Default command to run when container starts
CMD ["python", "train.py"]



