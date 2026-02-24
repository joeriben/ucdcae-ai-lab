"""
LLM Inference Routes - REST endpoints for production LLM inference

Endpoints:
- POST /api/llm/chat      - Messages-based chat (mirrors Ollama /api/chat)
- POST /api/llm/generate   - Raw prompt generation (mirrors Ollama /api/generate)
- GET  /api/llm/available  - Health check
- GET  /api/llm/models     - List loaded models
- POST /api/llm/unload     - Force unload a model
"""

import logging
from flask import Blueprint, request, jsonify
import asyncio

logger = logging.getLogger(__name__)

llm_bp = Blueprint('llm', __name__, url_prefix='/api/llm')


def _run_async(coro):
    """Run async coroutine in sync Flask context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@llm_bp.route('/chat', methods=['POST'])
def chat():
    """Messages-based chat inference.

    Request:
        {
            "model": "qwen3:1.7b",
            "messages": [{"role": "user", "content": "Hello"}],
            "images": ["base64..."],         // optional, for vision models
            "temperature": 0.7,              // optional
            "max_new_tokens": 500,           // optional
            "keep_alive": "10m"              // accepted for compat, ignored
        }

    Response:
        {
            "success": true,
            "content": "...",
            "thinking": "..." or null
        }
    """
    from config import LLM_INFERENCE_ENABLED
    if not LLM_INFERENCE_ENABLED:
        return jsonify({"error": "LLM inference backend disabled"}), 503

    data = request.get_json() or {}
    model = data.get("model")
    messages = data.get("messages")

    if not model:
        return jsonify({"error": "model required"}), 400
    if not messages:
        return jsonify({"error": "messages required"}), 400

    images = data.get("images")
    temperature = data.get("temperature", 0.7)
    max_new_tokens = data.get("max_new_tokens", data.get("num_predict", 500))
    repetition_penalty = data.get("repetition_penalty")
    enable_thinking = data.get("enable_thinking", True)

    # Ollama compat: options.temperature / options.num_predict / options.repeat_penalty
    options = data.get("options", {})
    if "temperature" in options:
        temperature = options["temperature"]
    if "num_predict" in options:
        max_new_tokens = options["num_predict"]
    if "repeat_penalty" in options:
        repetition_penalty = options["repeat_penalty"]

    from services.llm_inference_backend import get_llm_inference_backend
    backend = get_llm_inference_backend()

    result = _run_async(backend.chat(
        model_name=model,
        messages=messages,
        images=images,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        enable_thinking=enable_thinking,
    ))

    if result is None:
        return jsonify({"error": f"Inference failed for model {model}"}), 500

    return jsonify({
        "success": True,
        "content": result.get("content", ""),
        "thinking": result.get("thinking"),
        # Ollama compat: also include message format
        "message": {
            "role": "assistant",
            "content": result.get("content", ""),
            "thinking": result.get("thinking"),
        }
    })


@llm_bp.route('/generate', methods=['POST'])
def generate():
    """Raw prompt generation.

    Request:
        {
            "model": "qwen3:1.7b",
            "prompt": "Translate this text...",
            "temperature": 0.7,              // optional
            "max_new_tokens": 500,           // optional
            "keep_alive": "10m"              // accepted for compat, ignored
        }

    Response:
        {
            "success": true,
            "response": "...",
            "thinking": "..." or null
        }
    """
    from config import LLM_INFERENCE_ENABLED
    if not LLM_INFERENCE_ENABLED:
        return jsonify({"error": "LLM inference backend disabled"}), 503

    data = request.get_json() or {}
    model = data.get("model")
    prompt = data.get("prompt")

    if not model:
        return jsonify({"error": "model required"}), 400
    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    temperature = data.get("temperature", 0.7)
    max_new_tokens = data.get("max_new_tokens", data.get("num_predict", 500))
    repetition_penalty = data.get("repetition_penalty")
    enable_thinking = data.get("enable_thinking", True)

    from services.llm_inference_backend import get_llm_inference_backend
    backend = get_llm_inference_backend()

    result = _run_async(backend.generate(
        model_name=model,
        prompt=prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        enable_thinking=enable_thinking,
    ))

    if result is None:
        return jsonify({"error": f"Inference failed for model {model}"}), 500

    return jsonify({
        "success": True,
        "response": result.get("response", ""),
        "thinking": result.get("thinking"),
    })


@llm_bp.route('/available', methods=['GET'])
def available():
    """Health check for LLM inference backend."""
    from config import LLM_INFERENCE_ENABLED
    return jsonify({
        "available": LLM_INFERENCE_ENABLED,
        "backend": "llm_inference",
    })


@llm_bp.route('/models', methods=['GET'])
def list_models():
    """List currently loaded LLM models."""
    from config import LLM_INFERENCE_ENABLED
    if not LLM_INFERENCE_ENABLED:
        return jsonify({"models": []})

    from services.llm_inference_backend import get_llm_inference_backend
    backend = get_llm_inference_backend()

    return jsonify({"models": backend.get_loaded_models()})


@llm_bp.route('/unload', methods=['POST'])
def unload():
    """Force unload a model.

    Request:
        {"model_id": "Qwen/Qwen3-1.7B"}
    """
    from config import LLM_INFERENCE_ENABLED
    if not LLM_INFERENCE_ENABLED:
        return jsonify({"error": "LLM inference backend disabled"}), 503

    data = request.get_json() or {}
    model_id = data.get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400

    from services.llm_inference_backend import get_llm_inference_backend
    backend = get_llm_inference_backend()

    success = _run_async(backend.unload_model(model_id))

    return jsonify({"success": success, "model_id": model_id})
