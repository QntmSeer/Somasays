import os
import sys
import pytest
from unittest.mock import MagicMock

# Inject a mock module for generation_engine.optimized_inference
# to prevent the test suite from attempting to load heavy model weights or CUDA
class MockProtein:
    def to_pdb(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 90.00\nEND\n")

class MockESM3Generator:
    def __init__(self, *args, **kwargs):
        self.precision = kwargs.get("precision", "bf16")
        self.enable_sdpa = kwargs.get("enable_sdpa", True)
        
    def generate(self, prompt_sequence: str, num_steps: int, temperature: float) -> dict:
        # Simple rule: replace underscores with Alanine
        result_seq = prompt_sequence.replace("_", "A")
        return {
            "sequence": result_seq,
            "protein": MockProtein()
        }

# Create mock module
mock_inference = MagicMock()
mock_inference.OptimizedESM3Generator = MockESM3Generator
sys.modules["generation_engine.optimized_inference"] = mock_inference

@pytest.fixture(autouse=True)
def clean_api_database():
    """Fixture to ensure the SQLite database and tasks directory are clean before/after tests."""
    db_path = "tasks.db"
    out_dir = "api_outputs"
    
    # Remove existing db if it exists
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
            
    # Recreate the table schema
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            prompt_sequence TEXT,
            num_steps INTEGER,
            temperature REAL,
            status TEXT,
            result_sequence TEXT,
            pdb_path TEXT,
            error_message TEXT
        )
    """)
    conn.commit()
    conn.close()
    
    yield
    
    # Cleanup after test
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
            
    if os.path.exists(out_dir):
        import shutil
        try:
            shutil.rmtree(out_dir)
        except OSError:
            pass
