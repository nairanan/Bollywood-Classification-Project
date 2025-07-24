# Use an official lightweight Python image as base
FROM python:3.12

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy everything from your local folder to /app in the container
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 5000 so Flask can be reached
EXPOSE 5000

# Tell Docker how to start your app
CMD ["python", "Server/server.py"]