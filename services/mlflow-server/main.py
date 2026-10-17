import os
import subprocess
import sys

def start_mlflow_server():
    
    """
    Starts the MLflow tracking server.
    In a production environment, this would be handled by a
    dedicated process manager or Docker, but this script
    allows for easy local testing and CI smoke tests.
    """
    
    print("--- Initializing MLflow Tracking Server ---")

    # Configuration for MLflow
    # We use a local directory for the backend store and artifact store for simplicity.
    # In production, these would be a Postgres DB and an S3 bucket.
    backend_store_uri = "sqlite:///mlflow.db"
    default_artifact_root = "artifacts"
    host = "0.0.0.0"
    port = "5000"

    # Command to start the MLflow server
    # --host 0.0.0.0 makes it accessible from outside the container
    # --backend-store-uri defines where the experiment data is saved
    # --default-artifact-root defines where the actual model files go
    cmd = [
        "mlflow", "server",
        "--host", host,
        "--port", port,
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", default_artifact_root,
        "--no-frontend" # Optional: disable UI to save resources in CI
    ]

    print(f"Executing: {' '.join(cmd)}")

    try:
        # Start the server as a subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"MLflow server started successfully on port {port}...")
        return process
    except Exception as e:
        print(f"Failed to start MLflow server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    server_process = start_mlflow_server()
    # Keep the script running so the server stays alive
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
        server_process.terminate()
