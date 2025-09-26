# app.py
"""
Persona Chatbot (Flask + local GPT4All GGUF model)

1) Put your GGUF model file in ./models/, for example:
   models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf

2) Optionally set environment variable if model is in a different location:
   (PowerShell)
   $env:GPT4ALL_MODEL_PATH = "D:\path\to\models\Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"

3) Create & activate venv and install deps:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install flask gpt4all requests

4) Run:
   python app.py
"""

import os
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------- CONFIG ----------
# Default relative model path (change the filename to match the model you downloaded)
DEFAULT_MODEL_REL_PATH = "models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"

# Allow overriding via environment
MODEL_PATH = os.environ.get("GPT4ALL_MODEL_PATH", DEFAULT_MODEL_REL_PATH)
MAX_TOKENS = int(os.environ.get("GPT4ALL_MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("GPT4ALL_TEMPERATURE", "0.2"))

# ---------- CHAT STATE (demo only; in-memory) ----------
CHAT_HISTORY = []  # list of (user, bot)
PERSONA = "You are a helpful assistant."

# ---------- GPT4All availability ----------
try:
    from gpt4all import GPT4All  # type: ignore
    GPT4ALL_AVAILABLE = True
    _gpt4all_import_error = None
except Exception as e:
    GPT4ALL_AVAILABLE = False
    GPT4All = None  # type: ignore
    _gpt4all_import_error = str(e)

_model = None  # lazy model instance


import os
import shutil
from pathlib import Path

def init_model():
    """
    Initialize GPT4All trying multiple constructor styles.
    If MODEL_PATH points to a .gguf/.bin file, create a container dir and put a link named
    'custom.gguf' inside it so gpt4all versions that expect a directory will find it.
    """
    global _model
    if _model is not None:
        return _model

    if not GPT4ALL_AVAILABLE:
        raise RuntimeError(
            "gpt4all package not installed. Install it in your venv: pip install gpt4all\n"
            f"Import error: {_gpt4all_import_error}"
        )

    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_file}\n"
            "Download a GGUF model and place it there, or set GPT4ALL_MODEL_PATH to the file path."
        )

    # If the user pointed to a file (e.g. .gguf or .bin), create a container directory and
    # place a *link* (or copy fallback) inside called custom.gguf so gpt4all finds it.
    # We'll only create this container if it's needed.
    model_container = None
    if model_file.is_file():
        # choose a container directory name next to the file
        # e.g. models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf_container
        model_container = model_file.parent / (model_file.name + "_container")
        custom_name = "custom.gguf" if model_file.suffix.lower().endswith("gguf") else "custom.bin"

        if not model_container.exists():
            model_container.mkdir(parents=True, exist_ok=True)

        dest = model_container / custom_name
        # If dest already exists, assume it's OK
        if not dest.exists():
            try:
                # try hard link first (fast, no copy) — works on same filesystem
                os.link(model_file, dest)
            except Exception:
                # fallback — copy the file (may take time for multi-GB models)
                shutil.copyfile(model_file, dest)

    last_exc = None

    # Try constructors — but if we created a container, try the directory-based constructor first
    try_order = []

    if model_container is not None:
        # prefer passing the container dir as model_path (matches the library's expectation)
        try_order.append(("model_path_dir", lambda: GPT4All(model_path=str(model_container))))
        try_order.append(("model_name_custom_dir", lambda: GPT4All(model_name="custom", model_path=str(model_container), allow_download=False)))
    # also still try passing the file path as positional arg (works with many versions)
    try_order.append(("positional_file", lambda: GPT4All(str(model_file))))
    try_order.append(("model_path_kw_file", lambda: GPT4All(model_path=str(model_file))))
    try_order.append(("model_name_custom_file", lambda: GPT4All(model_name="custom", model_path=str(model_file), allow_download=False)))

    for desc, fn in try_order:
        try:
            _model = fn()
            app.logger.info("Initialized GPT4All using strategy: %s", desc)
            return _model
        except Exception as e:
            last_exc = e
            app.logger.debug("Constructor %s failed: %s", desc, e, exc_info=True)

    raise RuntimeError(
        "Failed to initialize GPT4All model. Tried multiple constructors but none worked.\n"
        f"Model file: {model_file}\nLast error: {last_exc}"
    )



def generate_reply(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> str:
    """
    Generate a reply from the local model. Tries a few generate() call patterns.
    Returns generated text or raises an exception.
    """
    model = init_model()

    # Try common generate signatures and shapes.
    try:
        # Preferred modern shape
        out = model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    except TypeError:
        try:
            out = model.generate(prompt, max_tokens)
        except Exception:
            out = model.generate(prompt)
    except Exception as e:
        # rethrow so caller will log meaningful message
        raise

    # Normalize output
    if isinstance(out, (list, tuple)):
        return " ".join(str(x) for x in out).strip()
    return str(out).strip()


# ---------- Flask routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/set_persona", methods=["POST"])
def set_persona():
    global PERSONA, CHAT_HISTORY
    try:
        data = request.get_json(force=True, silent=True) or {}
        persona = (data.get("persona") or "").strip()
        if not persona:
            return jsonify({"error": "No persona provided"}), 400
        PERSONA = persona
        CHAT_HISTORY = []
        return jsonify({"status": "ok", "persona": PERSONA})
    except Exception as e:
        app.logger.error("Error in /set_persona: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    global CHAT_HISTORY, PERSONA
    try:
        if not GPT4ALL_AVAILABLE:
            return (
                jsonify(
                    {
                        "error": "gpt4all package not available. Install with: pip install gpt4all",
                        "detail": _gpt4all_import_error,
                    }
                ),
                500,
            )

        data = request.get_json(force=True, silent=True) or {}
        user_msg = (data.get("message") or "").strip()
        if not user_msg:
            return jsonify({"error": "Empty message"}), 400

        # Build prompt with persona and recent history
        history_lines = []
        for u, b in CHAT_HISTORY[-6:]:
            history_lines.append(f"User: {u}\nBot: {b}")
        history_text = "\n".join(history_lines).strip()

        parts = [PERSONA, ""]
        if history_text:
            parts += ["Conversation so far:", history_text, ""]
        parts += [f"User: {user_msg}", "Bot:"]
        prompt = "\n".join(parts)

        # Generate safely
        try:
            reply = generate_reply(prompt)
            if not reply:
                reply = "Sorry — I couldn't produce an answer."
        except FileNotFoundError as e:
            app.logger.error("Model file missing: %s", e, exc_info=True)
            return jsonify({"error": f"Model file not found: {e}"}), 500
        except Exception as e:
            app.logger.error("Model generation failed: %s", e, exc_info=True)
            return jsonify({"error": f"Model generation failed: {e}"}), 500

        CHAT_HISTORY.append((user_msg, reply))
        return jsonify({"reply": reply, "history": CHAT_HISTORY})

    except Exception as e:
        app.logger.error("Unexpected /chat error: %s", e, exc_info=True)
        return jsonify({"error": "Unexpected server error", "detail": str(e)}), 500


# ---------- startup ----------
if __name__ == "__main__":
    print("Starting Persona Chatbot (local GPT4All).")
    print(f"Model path: {MODEL_PATH}")
    if not GPT4ALL_AVAILABLE:
        print("WARNING: gpt4all package not installed. Install with: pip install gpt4all")
    else:
        if not Path(MODEL_PATH).exists():
            print(f"WARNING: Model file not found at {MODEL_PATH}. Set GPT4ALL_MODEL_PATH or place model there.")
    app.run(debug=True)

