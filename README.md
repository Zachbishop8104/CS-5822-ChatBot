## CS-5822 RAG

FastAPI server for the web UI. Wraps `generate.py` as a subprocess and handles note management.

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

---

## Endpoints

### `POST /generate`

Runs `generate.py` and returns the output.

| Field | Type | Default |
|---|---|---|
| `prompt` | str | **required** |
| `user_id` | str | **required** |
| `model_file_name` | str | `Model_finetuned_best.pth` |
| `temperature` | float | `0.01` |
| `top_k` | int | `50` |
| `top_p` | float | `0.92` |
| `repetition_penalty` | float | `1.0` |
| `max_new_tokens` | int | `80` |
| `selected_note_files` | list[str] | `[]` |

Response: `{ "response": "...", "debug": "...", "returncode": 0 }`

`response` is stdout (generated text). `debug` is stderr (tokenizer info + `[DEBUG]` lines).

---

### `GET /notes/{user_id}`

Lists `.txt` files in `users/{user_id}/`.

Response: `{ "notes": ["file1.txt", "file2.txt"] }`

---

### `POST /uploadfile/`

Uploads a note file. Multipart form data.

| Field | Type |
|---|---|
| `file` | file |
| `user_id` | str (default: `"default"`) |