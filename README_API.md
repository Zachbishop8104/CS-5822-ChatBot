# API Server for generate.py

This FastAPI

## Usage

1. **Install FastAPI and Uvicorn**

```bash
pip install fastapi uvicorn
```

2. **Start the server**

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

3. **POST to /generate**

Send a JSON payload to `http://<server-ip>:8000/generate` with the following fields:

- `prompt` (str, required)
- `model_file_name` (str, default: Model_finetuned_best.pth)
- `context` (str, optional)
- `instruction` (str, optional)
- `temperature` (float, default: 0.1)
- `top_k` (int, default: 10)
- `max_new_tokens` (int, default: 80)
- `prepend_bos` (bool, default: false)
- `auto_grounding` (bool, default: false)
- `grounding_sentences` (int, default: 2)
- `debug_grounding` (bool, default: false)

Example request (using fetch in JS):

```js
fetch('http://<server-ip>:8000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "How hot is the sun?",
    context: "The Sun's photosphere is about 5,500 C, while its core reaches around 15 million C.",
    instruction: "Answer in one sentence and include both photosphere and core temperatures with units.",
    auto_grounding: true,
    debug_grounding: true
  })
})
  .then(r => r.json())
  .then(console.log)
```

## Notes
- The server simply shells out to generate.py and returns its output.
- The server must be started from the project root or paths may need adjustment.
