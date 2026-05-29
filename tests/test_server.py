import os
import time
import pytest
from fastapi.testclient import TestClient

# Add Somasays root to Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_service.server import app, DB_PATH, OUTPUT_DIR

client = TestClient(app)

def test_create_generation_task():
    response = client.post(
        "/v1/tasks",
        json={"prompt_sequence": "MKA____VLA", "num_steps": 4, "temperature": 0.5}
    )
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] in ["PENDING", "PROCESSING", "COMPLETED"]
    assert data["prompt_sequence"] == "MKA____VLA"
    
    # Retrieve status and verify it successfully processed
    task_id = data["task_id"]
    time.sleep(0.5) # Give the background thread a moment if async
    
    status_response = client.get(f"/v1/tasks/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["task_id"] == task_id
    assert status_data["status"] == "COMPLETED"
    assert status_data["result_sequence"] == "MKAAAAAVLA" # Replaced underscores with A
    assert os.path.exists(status_data["pdb_path"])

def test_task_not_found():
    response = client.get("/v1/tasks/invalid-uuid-string")
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"
