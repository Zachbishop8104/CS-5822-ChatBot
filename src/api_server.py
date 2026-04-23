from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import subprocess
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import File, UploadFile, Form
from upload_file import download_note

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    model_file_name: str = "Model_finetuned_best.pth"
    temperature: float = 0.01
    top_k: int = 50
    top_p: float = 0.92
    repetition_penalty: float = 1.0
    max_new_tokens: int = 80
    user_id: str = "default" # Required for retrieval
    selected_note_files: list[str] = Field(default_factory=list)

@app.post("/generate")
async def generate(req: GenerateRequest):
    args = [
        "python", "generate.py",
        "--prompt", req.prompt,
        "--username", req.user_id,
        "--checkpoint", req.model_file_name,
        "--temperature", str(req.temperature),
        "--top_k", str(req.top_k),
        "--top_p", str(req.top_p),
        "--max_new_tokens", str(req.max_new_tokens),
        "--repetition_penalty", str(req.repetition_penalty),
        "--debug"
    ]

    # Run the generate.py script and capture output
    proc = subprocess.run(
        args, 
        capture_output=True, 
        text=True, 
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # stdout contains the generated text
    response = proc.stdout.strip()
    
    # stderr contains the tokenizer info and all [DEBUG] lines
    debug = proc.stderr.strip()
    
    return JSONResponse({
        "response": response, 
        "debug": debug, 
        "returncode": proc.returncode
    })


@app.get("/notes/{user_id}")
async def list_notes(user_id: str):
    user_dir = BASE_DIR / "users" / user_id
    if not user_dir.exists():
        return JSONResponse({"notes": []})
    notes = sorted(f.name for f in user_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt")
    return JSONResponse({"notes": notes})

@app.post("/uploadfile/")
async def uploadfile(file: UploadFile, user_id: str = Form("default")):
    response = await download_note(file, user_id=user_id)
    if isinstance(response, dict) and "error" in response:
        return JSONResponse(status_code=400, content={"detail": response["error"]})
    return JSONResponse({"message": response})

@app.get("/")
async def root():
    return {"message": "FastAPI server for generate.py. POST to /generate."}

app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")