from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import File, UploadFile, Form
from upload_file import download_note

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
    max_new_tokens: int = 80
    prepend_bos: bool = False
    auto_grounding: bool = False
    grounding_sentences: int = 2
    debug_grounding: bool = False,
    user_id: str = "default"

@app.post("/generate")
async def generate(req: GenerateRequest):
    args = [
        "python", "generate.py",
        "--model_file_name", req.model_file_name,
        "--prompt", req.prompt,
        "--temperature", str(req.temperature),
        "--top_k", str(req.top_k),
        "--max_new_tokens", str(req.max_new_tokens),
        "--user_id", req.user_id,
    ]
    if req.context:
        args += ["--context", req.context]
    if req.instruction:
        args += ["--instruction", req.instruction]
    if req.prepend_bos:
        args.append("--prepend_bos")
    if req.auto_grounding:
        args.append("--auto_grounding")
    if req.grounding_sentences:
        args += ["--grounding_sentences", str(req.grounding_sentences)]
    if req.debug_grounding:
        args.append("--debug_grounding")

    # Run the generate.py script and capture output
    proc = subprocess.run(args, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    output = proc.stdout.strip()
    return JSONResponse({"output": output, "stderr": proc.stderr.strip(), "returncode": proc.returncode})

@app.post("/uploadfile/")
async def uploadfile(file: UploadFile, user_id: str = Form("default")):
    response = await download_note(file, user_id=user_id)
    if isinstance(response, dict) and "error" in response:
        return JSONResponse(status_code=400, content={"detail": response["error"]})
    return JSONResponse({"message": response})

@app.get("/")
async def root():
    return {"message": "FastAPI server for generate.py. POST to /generate."}

app.mount("/", StaticFiles(directory="..", html=True), name="static")