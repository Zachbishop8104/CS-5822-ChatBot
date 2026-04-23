from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import subprocess
import tempfile
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
    context: str = ""
    instruction: str = ""
    temperature: float = 0.1
    top_k: int = 10
    repetition_penalty: float = 1.3
    max_new_tokens: int = 80
    prepend_bos: bool = False
    auto_grounding: bool = True
    polish_grounded_answers: bool = True
    rewrite_temperature: float = 0.2
    rewrite_top_k: int = 20
    rewrite_max_new_tokens: int = 70
    grounding_sentences: int = 3
    debug_grounding: bool = False
    user_id: str = "default"
    selected_note_files: list[str] = Field(default_factory=list)

@app.post("/generate")
async def generate(req: GenerateRequest):
    args = [
        "python", "generate.py",
        "--prompt", req.prompt,
        "--checkpoint", req.model_file_name,
        "--temperature", str(req.temperature),
        "--top_k", str(req.top_k),
        "--max_new_tokens", str(req.max_new_tokens),
        "--repetition_penalty", str(req.repetition_penalty),
        "--debug"
    ]

    # Run the generate.py script and capture output
    proc = subprocess.run(args, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    response = proc.stdout.strip()
    debug = proc.stderr.strip()
    return JSONResponse({"response": response, "debug": debug, "returncode": proc.returncode})


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