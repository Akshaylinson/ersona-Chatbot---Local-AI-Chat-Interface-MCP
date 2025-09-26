# app.py
"""
Persona Chatbot (Flask + local GPT4All GGUF)
- Place a GGUF model file in ./models/ (e.g. models/my-model.gguf)
- The loader will:
    * accept MODEL_PATH pointing to a file or directory
    * create a container dir + hard link named custom.gguf if the library expects a directory
    * try multiple GPT4All constructor signatures for compatibility
- Exposes APIs used by the frontend:
    GET  /api/status
    GET  /api/model_info
    POST /api/chat
    POST /api/persona
    GET  /api/history
    POST /api/clear
"""

import os
import sys
import time
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, render_template, session

# ---------------- CONFIG ----------------
DEFAULT_MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"
MODEL_PATH = os.environ.get("GPT4ALL_MODEL_PATH", DEFAULT_MODEL_PATH)

MAX_TOKENS = int(os.environ.get("GPT4ALL_MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("GPT4ALL_TEMPERATURE", "0.2"))
TOP_K = int(os.environ.get("GPT4ALL_TOP_K", "40"))
TOP_P = float(os.environ.get("GPT4ALL_TOP_P", "0.9"))

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("persona-chat")

# ---------------- FLASK ----------------
app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key-2024-chatbot")

# ---------------- GPT4All IMPORT ----------------
GPT4ALL_AVAILABLE = False
_g4a_import_error = None
try:
    from gpt4all import GPT4All  # type: ignore
    GPT4ALL_AVAILABLE = True
    try:
        import pkg_resources
        GPT4ALL_VERSION = pkg_resources.get_distribution("gpt4all").version
    except Exception:
        GPT4ALL_VERSION = "unknown"
    logger.info(f"gpt4all imported (version={GPT4ALL_VERSION})")
except Exception as e:
    GPT4ALL_AVAILABLE = False
    _g4a_import_error = str(e)
    logger.warning(f"gpt4all import failed: {_g4a_import_error}")

# ---------------- GLOBAL STATE ----------------
_model = None
_init_error: Optional[str] = None

# ---------------- UTIL: container for file ----------------
def ensure_container_for_file(model_file: Path) -> Path:
    """
    If model_file is a file (gguf or bin), create a container directory next to it
    and place a hard link (or copy fallback) named custom.gguf or custom.bin inside it.
    Return the container directory Path.
    """
    if model_file.is_dir():
        return model_file

    container = model_file.parent / (model_file.name + "_container")
    container.mkdir(parents=True, exist_ok=True)

    suffix = model_file.suffix.lower()
    custom_name = "custom.gguf" if suffix.endswith("gguf") else "custom.bin"
    dest = container / custom_name

    if dest.exists():
        logger.debug("Container destination already exists: %s", dest)
        return container

    try:
        os.link(model_file, dest)
        logger.info("Created hard link: %s -> %s", dest, model_file)
    except Exception as e:
        logger.info("Hard link failed (%s). Copying file into container (may take time)...", e)
        shutil.copyfile(model_file, dest)
        logger.info("Copied model to container: %s", dest)

    return container

# ---------------- INIT MODEL ----------------
def init_model():
    """
    Initialize GPT4All model robustly. Try container + multiple constructor signatures.
    Raises exceptions on fatal errors.
    """
    global _model, _init_error
    if _model is not None:
        return _model

    if not GPT4ALL_AVAILABLE:
        _init_error = f"gpt4all package not installed: {_g4a_import_error}"
        logger.error(_init_error)
        raise RuntimeError(_init_error)

    model_path_obj = Path(MODEL_PATH)
    if not model_path_obj.exists():
        _init_error = f"Model file not found at: {MODEL_PATH}"
        logger.error(_init_error)
        raise FileNotFoundError(_init_error)

    # If passed a file, build a container dir expected by some gpt4all versions
    model_container = None
    if model_path_obj.is_file():
        model_container = ensure_container_for_file(model_path_obj)

    last_exc = None
    attempts = []
    if model_container:
        attempts.append(("model_path_dir", lambda: GPT4All(model_path=str(model_container))))
        attempts.append(("model_name_custom_dir", lambda: GPT4All(model_name="custom", model_path=str(model_container), allow_download=False)))
    attempts.append(("positional_file", lambda: GPT4All(str(model_path_obj))))
    attempts.append(("model_path_kw_file", lambda: GPT4All(model_path=str(model_path_obj))))
    attempts.append(("model_name_custom_file", lambda: GPT4All(model_name="custom", model_path=str(model_path_obj), allow_download=False)))

    for desc, fn in attempts:
        try:
            logger.info("Trying GPT4All constructor: %s", desc)
            _model = fn()
            logger.info("GPT4All initialized using strategy: %s", desc)
            return _model
        except Exception as e:
            last_exc = e
            logger.debug("Constructor %s failed: %s", desc, e, exc_info=True)

    _init_error = f"Failed to initialize GPT4All. Last error: {last_exc}"
    logger.error(_init_error)
    raise RuntimeError(_init_error)

# ---------------- NORMALIZE OUTPUT ----------------
def normalize_generate_output(out) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, (list, tuple)):
        return " ".join(str(x) for x in out).strip()
    return str(out).strip()

# ---------------- GENERATION ----------------
def generate_reply(user_prompt: str, persona: str, history: list) -> str:
    """
    Build the prompt using persona + short history, call model.generate and return text.
    """
    global _model
    if _model is None:
        init_model()

    history_lines = []
    for msg in history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_lines.append(f"{role}: {msg.get('content')}")
    history_text = "\n".join(history_lines).strip()

    if history_text:
        prompt = f"{persona}\n\nConversation so far:\n{history_text}\n\nUser: {user_prompt}\nAssistant:"
    else:
        prompt = f"{persona}\n\nUser: {user_prompt}\nAssistant:"

    try:
        # Try typical signatures; fallback to positional if kw args not supported
        try:
            out = _model.generate(prompt,
                                  max_tokens=MAX_TOKENS,
                                  temp=TEMPERATURE,
                                  top_k=TOP_K,
                                  top_p=TOP_P,
                                  streaming=False)
        except TypeError:
            out = _model.generate(prompt, MAX_TOKENS)
        return normalize_generate_output(out)
    except Exception as e:
        logger.exception("Model generation error")
        raise

# ---------------- SIMPLE IN-MEM CHAT ----------------
class SimpleChatManager:
    def __init__(self):
        self.sessions = {}

    def get_session(self, sid):
        if sid not in self.sessions:
            self.sessions[sid] = {
                "persona": "You are a helpful, friendly AI assistant that acts as a professional career tutor. Provide detailed and thoughtful responses.",
                "history": [],
                "created_at": time.time()
            }
        return self.sessions[sid]

    def add_message(self, sid, role, content):
        sess = self.get_session(sid)
        msg = {"role": role, "content": content, "ts": time.time()}
        sess["history"].append(msg)
        if len(sess["history"]) > 40:
            sess["history"] = sess["history"][-40:]
        return msg

    def clear_history(self, sid):
        if sid in self.sessions:
            self.sessions[sid]["history"] = []
            return True
        return False

chat_mgr = SimpleChatManager()

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    # Best-effort init
    try:
        if _model is None:
            init_model()
    except Exception:
        logger.debug("Background init attempt failed; UI will show status")
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    model_file = Path(MODEL_PATH)
    file_exists = model_file.exists()
    file_size = f"{model_file.stat().st_size / (1024**3):.2f} GB" if file_exists and model_file.is_file() else "N/A"
    return jsonify({
        "gpt4all_available": GPT4ALL_AVAILABLE,
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "model_name": Path(MODEL_PATH).name if MODEL_PATH else "Unknown",
        "file_exists": file_exists,
        "file_size": file_size,
        "error": _init_error
    })

@app.route("/api/model_info")
def api_model_info():
    return api_status()

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        payload = request.get_json(force=True) or {}
        message = (payload.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400

        sid = session.get("sid", str(uuid.uuid4()))
        session["sid"] = sid
        chat_mgr.add_message(sid, "user", message)
        sess = chat_mgr.get_session(sid)

        start = time.time()
        try:
            reply = generate_reply(message, sess["persona"], sess["history"])
        except Exception as e:
            logger.error("Generation error: %s", e)
            return jsonify({"error": f"Generation error: {e}"}), 500
        elapsed = time.time() - start

        chat_mgr.add_message(sid, "assistant", reply)
        return jsonify({"reply": reply, "response_time": round(elapsed, 2), "message_count": len(sess["history"])})
    except Exception as e:
        logger.exception("api_chat exception")
        return jsonify({"error": str(e)}), 500

@app.route("/api/persona", methods=["POST"])
def api_persona():
    try:
        data = request.get_json(force=True) or {}
        persona = (data.get("persona") or "").strip()
        if not persona:
            return jsonify({"error": "Empty persona"}), 400
        sid = session.get("sid", str(uuid.uuid4()))
        session["sid"] = sid
        sess = chat_mgr.get_session(sid)
        sess["persona"] = persona
        chat_mgr.clear_history(sid)
        return jsonify({"status": "ok", "persona": persona})
    except Exception as e:
        logger.exception("api_persona")
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        sid = session.get("sid", str(uuid.uuid4()))
        sess = chat_mgr.get_session(sid)
        return jsonify({"history": sess["history"], "persona": sess["persona"], "message_count": len(sess["history"])})
    except Exception as e:
        logger.exception("api_history")
        return jsonify({"error": str(e)}), 500

@app.route("/api/clear", methods=["POST"])
def api_clear():
    try:
        sid = session.get("sid", str(uuid.uuid4()))
        chat_mgr.clear_history(sid)
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.exception("api_clear")
        return jsonify({"error": str(e)}), 500

# ---------------- START ----------------
if __name__ == "__main__":
    print("=" * 60)
    print("Persona Chatbot starting")
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Model path (env/GPT4ALL_MODEL_PATH or default): {MODEL_PATH}")
    model_file = Path(MODEL_PATH)
    if model_file.exists():
        size_gb = model_file.stat().st_size / (1024**3) if model_file.is_file() else 0
        print(f"Model found: {model_file.name} ({size_gb:.2f} GB)")
    else:
        print("Model file not found at the configured path.")
    print("=" * 60)
    try:
        init_model()
    except Exception as e:
        print("Model init failed (server will still start). Error:", e)

    app.run(debug=True, host="0.0.0.0", port=5000)

