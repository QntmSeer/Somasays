import os
import sys
import uuid
import sqlite3
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Ensure Somasays root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize FastAPI App
app = FastAPI(
    title="Somasays ESM3 Model Serving Service",
    description="An enterprise-grade, systems-optimized API for generating and folding protein sequences.",
    version="1.0.0"
)

DB_PATH = "tasks.db"
OUTPUT_DIR = "api_outputs"

# Create output folder and database schema
os.makedirs(OUTPUT_DIR, exist_ok=True)
conn = sqlite3.connect(DB_PATH)
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

class TaskRequest(BaseModel):
    prompt_sequence: str = Field(
        default="MKA___________________VLA",
        description="Protein sequence template with '_' representing positions to generate de novo."
    )
    num_steps: int = Field(default=8, ge=1, le=32, description="Inference steps for generation and folding.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    prompt_sequence: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    prompt_sequence: str
    result_sequence: Optional[str] = None
    pdb_path: Optional[str] = None
    error_message: Optional[str] = None

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def run_esm3_worker(task_id: str, prompt_sequence: str, num_steps: int, temperature: float):
    """
    Asynchronous background worker executing Somasays optimized ESM3 generation.
    This simulates an out-of-process SQS task worker handling heavy GPU workloads.
    """
    db = get_db_connection()
    cursor = db.cursor()
    
    # Update status to processing
    cursor.execute("UPDATE tasks SET status = 'PROCESSING' WHERE task_id = ?", (task_id,))
    db.commit()
    
    try:
        # Import the generator dynamically to allow lazy GPU allocation
        from generation_engine.optimized_inference import OptimizedESM3Generator
        
        # Initialize the generator with performance features enabled
        generator = OptimizedESM3Generator(
            precision="bf16",
            enable_sdpa=True,
            force_flash_attn=False
        )
        
        # Execute optimized bfloat16 + FlashAttention forward loop
        result = generator.generate(
            prompt_sequence=prompt_sequence,
            num_steps=num_steps,
            temperature=temperature
        )
        
        generated_seq = result["sequence"]
        protein = result["protein"]
        
        # Export coordinate files
        pdb_filename = os.path.join(OUTPUT_DIR, f"{task_id}.pdb")
        protein.to_pdb(pdb_filename)
        
        cursor.execute(
            "UPDATE tasks SET status = 'COMPLETED', result_sequence = ?, pdb_path = ? WHERE task_id = ?",
            (generated_seq, pdb_filename, task_id)
        )
        db.commit()
        
    except Exception as e:
        error_msg = str(e)
        cursor.execute(
            "UPDATE tasks SET status = 'FAILED', error_message = ? WHERE task_id = ?",
            (error_msg, task_id)
        )
        db.commit()
    finally:
        db.close()

@app.post("/v1/tasks", response_model=TaskResponse, status_code=202)
def create_generation_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Submits a protein generation and folding task to the asynchronous background worker."""
    task_id = str(uuid.uuid4())
    
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO tasks (task_id, prompt_sequence, num_steps, temperature, status) VALUES (?, ?, ?, ?, 'PENDING')",
        (task_id, request.prompt_sequence, request.num_steps, request.temperature)
    )
    db.commit()
    db.close()
    
    # Enqueue task to FastAPI background executor (simulates SQS decoupling)
    background_tasks.add_task(
        run_esm3_worker,
        task_id=task_id,
        prompt_sequence=request.prompt_sequence,
        num_steps=request.num_steps,
        temperature=request.temperature
    )
    
    return {
        "task_id": task_id,
        "status": "PENDING",
        "prompt_sequence": request.prompt_sequence
    }

@app.get("/v1/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Retrieves the current execution status and outputs of a generation task."""
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute(
        "SELECT task_id, status, prompt_sequence, result_sequence, pdb_path, error_message FROM tasks WHERE task_id = ?",
        (task_id,)
    )
    row = cursor.fetchone()
    db.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return {
        "task_id": row[0],
        "status": row[1],
        "prompt_sequence": row[2],
        "result_sequence": row[3],
        "pdb_path": row[4],
        "error_message": row[5]
    }
