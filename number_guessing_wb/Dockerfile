# Use the official Python image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# We use --no-cache-dir for smaller image size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and the model file
# Ensure your model file name is correct here
COPY . /app

# The port that Cloud Run will use to send requests to the container
ENV PORT=8080

# Expose the port (Flask will need to bind to this)
EXPOSE 8080

# Run the Python application (app.py) when the container starts
# The flask run command is slightly different for production environments
CMD exec gunicorn --bind :$PORT --workers 1 app:app
