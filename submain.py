"""
Persona Chatbot (Flask + local GPT4All GGUF model)
Drop your GGUF file at models/mistral-7b-instruct-q4.gguf (or change MODEL_FILE_NAME).
"""

import os
import shutil
import time
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ---------- CONFIG ----------
MODELS_DIR = Path("models")
MODEL_FILE_NAME = "mistral-7b-instruct-q4.gguf"   # <-- ensure this matches your filename
MODEL_PATH = (MODELS_DIR / MODEL_FILE_NAME).as_posix()
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))

# ---------- GPT4All import ----------
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
    gpt4all_import_error = None
except Exception as e:
    GPT4ALL_AVAILABLE = False
    GPT4All = None
    gpt4all_import_error = str(e)

# ---------- In-memory state ----------
CHAT_HISTORY = []  # list of dicts: {"role":"user"/"assistant", "text": "..."}
PERSONA = "You are a helpful assistant."
_model = None  # lazy model

def _ensure_model_container(model_file: Path):
    """
    If model_file is a file (gguf/bin), create a container directory next to it and
    place a link or copy named `custom.gguf` (or custom.bin) so gpt4all constructors
    that expect a directory will find it.
    Return the container path (Path) or None if not created.
    """
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    if model_file.is_file():
        container = model_file.parent / (model_file.name + "_container")
        container.mkdir(parents=True, exist_ok=True)
        # choose name
        if model_file.suffix.lower().endswith(".gguf"):
            custom_name = "custom.gguf"
        else:
            custom_name = "custom.bin"
        dest = container / custom_name
        if dest.exists():
            return container
        # try hard link then fallback to copy
        try:
            os.link(model_file, dest)
        except Exception:
            shutil.copyfile(model_file, dest)
        return container
    return None

def init_model():
    """Initialize the global _model, trying several constructor styles."""
    global _model
    if _model is not None:
        return _model

    if not GPT4ALL_AVAILABLE:
        raise RuntimeError(f"gpt4all not installed: {gpt4all_import_error}")

    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at: {model_file}")

    model_container = None
    try:
        model_container = _ensure_model_container(model_file)
    except Exception as e:
        # If we couldn't prepare container, keep going but remember the error
        app.logger.warning("Could not prepare container dir: %s", e)

    last_exc = None
    attempts = []

    # If container created, try passing container dir first
    if model_container is not None:
        attempts.append(("model_path_dir", lambda: GPT4All(model_path=str(model_container))))
        attempts.append(("model_name_custom_dir", lambda: GPT4All(model_name="custom", model_path=str(model_container), allow_download=False)))

    # Try positional file, keyword file, and model_name/custom_file
    attempts.extend([
        ("positional_file", lambda: GPT4All(str(model_file))),
        ("model_path_kw_file", lambda: GPT4All(model_path=str(model_file))),
        ("model_name_custom_file", lambda: GPT4All(model_name="custom", model_path=str(model_file), allow_download=False)),
    ])

    for name, fn in attempts:
        try:
            app.logger.info("Trying GPT4All constructor: %s", name)
            _model = fn()
            app.logger.info("GPT4All initialized using: %s", name)
            return _model
        except Exception as e:
            last_exc = e
            app.logger.debug("Constructor %s failed: %s", name, e, exc_info=True)

    # All attempts failed
    raise RuntimeError(f"Failed to initialize GPT4All. Last error: {last_exc}")

def generate_reply(prompt: str, max_tokens:int = MAX_TOKENS, temperature:float = TEMPERATURE) -> str:
    model = init_model()
    # Try a couple generate signatures
    try:
        out = model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    except TypeError:
        try:
            out = model.generate(prompt, max_tokens)
        except Exception:
            out = model.generate(prompt)
    if isinstance(out, (list, tuple)):
        return " ".join(str(x) for x in out).strip()
    return str(out).strip()

# ---------- Flask routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    model_file = Path(MODEL_PATH)
    file_exists = model_file.exists()
    file_size = f"{model_file.stat().st_size / (1024**3):.2f} GB" if file_exists else "N/A"
    return jsonify({
        "model_path": MODEL_PATH,
        "file_exists": file_exists,
        "file_size": file_size,
        "gpt4all_available": GPT4ALL_AVAILABLE,
        "gpt4all_import_error": gpt4all_import_error
    })

@app.route("/api/persona", methods=["POST"])
def api_persona():
    global PERSONA, CHAT_HISTORY
    data = request.get_json(force=True) or {}
    persona = (data.get("persona") or "").strip()
    if not persona:
        return jsonify({"error":"Empty persona"}), 400
    PERSONA = persona
    CHAT_HISTORY = []
    return jsonify({"status":"ok","persona":PERSONA})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error":"Empty message"}), 400

    # build prompt
    recent = CHAT_HISTORY[-6:]  # keep limited history
    history_text = ""
    for u, b in recent:
        history_text += f"User: {u}\nAssistant: {b}\n"
    if history_text:
        history_text = "Conversation so far:\n" + history_text + "\n"
    prompt = f"{PERSONA}\n\n{history_text}User: {user_msg}\nAssistant:"

    try:
        start = time.time()
        reply = generate_reply(prompt)
        elapsed = time.time() - start
    except FileNotFoundError as e:
        app.logger.error("Model file missing: %s", e, exc_info=True)
        return jsonify({"error":f"Model file missing: {e}"}), 500
    except Exception as e:
        app.logger.error("Model generation failed: %s", e, exc_info=True)
        tb = traceback.format_exc()
        return jsonify({"error":f"Model generation failed: {e}", "trace": tb}), 500

    CHAT_HISTORY.append((user_msg, reply))
    return jsonify({"reply": reply, "elapsed": round(elapsed,2), "history": CHAT_HISTORY})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    global CHAT_HISTORY
    CHAT_HISTORY = []
    return jsonify({"status":"cleared"})

if __name__ == "__main__":
    print("="*60)
    print("Persona Chatbot (local GPT4All)")
    print("="*60)
    print(f"MODEL_PATH = {MODEL_PATH}")
    model_file = Path(MODEL_PATH)
    if model_file.exists():
        size_gb = model_file.stat().st_size / (1024**3)
        print(f"Model exists: {model_file.name} ({size_gb:.2f} GB)")
    else:
        print("Model file NOT FOUND. Put the GGUF at:", model_file)
    print("Starting Flask on http://127.0.0.1:5000")
    app.run(debug=True)

