# Base image with PyTorch and CUDA (adjust version as needed)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements_torch.txt

# If you have extra dependencies, e.g., for visualization
# RUN apt-get update && apt-get install -y libgl1-mesa-glx

