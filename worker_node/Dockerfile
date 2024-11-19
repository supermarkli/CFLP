# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the default gRPC communication port (if needed for diagnostics)
EXPOSE 50052

# Step 6: Define the command to run the Worker Node
CMD ["python", "worker_node.py"]
