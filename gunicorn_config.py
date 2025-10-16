# gunicorn_config.py
# This file configures Gunicorn, the production server for our Flask app.

# The number of worker processes for handling requests
workers = 1

# The socket to bind to.
# We use 0.0.0.0 to accept connections from any IP address,
# which is necessary for a containerized environment like Cloud Run.
# The port is determined by the PORT environment variable set in the Dockerfile.
bind = "0.0.0.0:8080"

# The number of seconds to wait for requests on a worker before timing out and restarting the worker.
timeout = 120

