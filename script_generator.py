# File: script_generator.py
# SmolLM3 script generator with cross-platform support
# - Mac (Apple Silicon): Uses MLX for fast inference
# - Windows/Linux: Uses Transformers + PyTorch (CUDA or CPU)

import logging
import platform
from typing import Optional

logger = logging.getLogger(__name__)

# Global model instances
_model = None
_tokenizer = None
_pipeline = None  # For transformers backend
MODEL_LOADED = False
BACKEND = None  # "mlx" or "transformers"

# Paralinguistic tags supported by Chatterbox Turbo
PARALINGUISTIC_TAGS = [
    "[laugh]", "[chuckle]", "[sigh]", "[gasp]",
    "[cough]", "[clear throat]", "[sniff]", "[groan]", "[shush]"
]

SYSTEM_PROMPT = """You are a live Twitch streamer reading and responding to chat messages. Keep responses SHORT and natural.

You can use these tags sparingly: [laugh], [chuckle], [sigh], [gasp], [cough], [clear throat], [sniff], [groan]

Rules:
1. ONE sentence maximum
2. Be energetic, engaging, and authentic like a real streamer
3. React naturally to chat - thank them for subs, respond to jokes, answer questions
4. No asterisks or stage directions
5. Output ONLY your spoken words/no_think"""


def _detect_best_backend() -> str:
    """Detect the best available backend for the current platform."""
    # Try MLX first (Mac only)
    if platform.system() == "Darwin":
        try:
            import mlx_lm
            logger.info("MLX backend available (Apple Silicon)")
            return "mlx"
        except ImportError:
            logger.info("MLX not available, trying transformers...")

    # Fall back to transformers
    try:
        import transformers
        import torch
        logger.info(f"Transformers backend available (PyTorch device: {'cuda' if torch.cuda.is_available() else 'cpu'})")
        return "transformers"
    except ImportError:
        logger.error("No LLM backend available. Install mlx-lm (Mac) or transformers (Windows/Linux)")
        return None


def load_model(model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct") -> bool:
    """
    Load the LLM model using the best available backend.

    Args:
        model_name: HuggingFace model name. Defaults to SmolLM2-1.7B-Instruct
                   which is smaller and faster than SmolLM3-3B.
    """
    global _model, _tokenizer, _pipeline, MODEL_LOADED, BACKEND

    if MODEL_LOADED:
        logger.info(f"LLM model already loaded (backend: {BACKEND})")
        return True

    BACKEND = _detect_best_backend()
    if BACKEND is None:
        return False

    try:
        if BACKEND == "mlx":
            return _load_mlx(model_name)
        else:
            return _load_transformers(model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def _load_mlx(model_name: str) -> bool:
    """Load model using MLX backend (Mac)."""
    global _model, _tokenizer, MODEL_LOADED

    # Use SmolLM3 on Mac since MLX handles it well
    if "SmolLM2" in model_name:
        model_name = "HuggingFaceTB/SmolLM3-3B"

    logger.info(f"Loading model with MLX: {model_name}")
    from mlx_lm import load

    _model, _tokenizer = load(model_name)
    MODEL_LOADED = True
    logger.info("MLX model loaded successfully.")
    return True


def _load_transformers(model_name: str) -> bool:
    """Load model using Transformers backend (Windows/Linux)."""
    global _model, _tokenizer, _pipeline, MODEL_LOADED

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    logger.info(f"Loading model with Transformers on {device}: {model_name}")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device == "cuda" else None,
    )

    if device == "cpu":
        _model = _model.to(device)

    # Create pipeline for easier generation
    _pipeline = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        device=0 if device == "cuda" else -1,
    )

    MODEL_LOADED = True
    logger.info(f"Transformers model loaded successfully on {device}.")
    return True


def unload_model():
    """Unload the model to free memory."""
    global _model, _tokenizer, _pipeline, MODEL_LOADED, BACKEND
    _model = None
    _tokenizer = None
    _pipeline = None
    MODEL_LOADED = False
    BACKEND = None
    logger.info("LLM model unloaded.")


def generate_script(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Optional[str]:
    """
    Generate a speech script from a prompt.

    Args:
        prompt: The topic or scenario for the script
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Top-p sampling parameter

    Returns:
        Generated script text or None if failed
    """
    global _model, _tokenizer, _pipeline, MODEL_LOADED, BACKEND

    if not MODEL_LOADED:
        if not load_model():
            return None

    try:
        import time
        import re

        total_start = time.perf_counter()

        # Build the full prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        if BACKEND == "mlx":
            response = _generate_mlx(messages, max_tokens, temperature, top_p)
        else:
            response = _generate_transformers(messages, max_tokens, temperature, top_p)

        if response is None:
            return None

        # Clean up the response - remove thinking tags
        script = response.strip()

        # Remove <think>...</think> blocks (SmolLM reasoning)
        script = re.sub(r'<think>.*?</think>', '', script, flags=re.DOTALL)
        script = script.strip()

        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(f"[BENCH] Script generation total: {total_time:.1f}ms ({len(script)} chars)")

        return script

    except Exception as e:
        logger.error(f"Script generation failed: {e}")
        return None


def _generate_mlx(messages: list, max_tokens: int, temperature: float, top_p: float) -> Optional[str]:
    """Generate using MLX backend."""
    import time
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    # Apply chat template
    template_start = time.perf_counter()
    formatted_prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    template_time = (time.perf_counter() - template_start) * 1000
    logger.info(f"[BENCH] Template applied in {template_time:.1f}ms")

    # Create sampler with temperature and top_p
    sampler = make_sampler(temp=temperature, top_p=top_p)

    # Generate response
    generate_start = time.perf_counter()
    response = generate(
        _model,
        _tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    generate_time = (time.perf_counter() - generate_start) * 1000
    logger.info(f"[BENCH] MLX generation took {generate_time:.1f}ms")

    return response


def _generate_transformers(messages: list, max_tokens: int, temperature: float, top_p: float) -> Optional[str]:
    """Generate using Transformers backend."""
    import time

    # Apply chat template
    template_start = time.perf_counter()
    formatted_prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    template_time = (time.perf_counter() - template_start) * 1000
    logger.info(f"[BENCH] Template applied in {template_time:.1f}ms")

    # Generate response
    generate_start = time.perf_counter()
    outputs = _pipeline(
        formatted_prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id,
        return_full_text=False,
    )
    generate_time = (time.perf_counter() - generate_start) * 1000
    logger.info(f"[BENCH] Transformers generation took {generate_time:.1f}ms")

    if outputs and len(outputs) > 0:
        return outputs[0]["generated_text"]
    return None


def generate_response(
    username: str,
    message: str,
    style: str = "friendly",
    include_tags: bool = True,
) -> Optional[str]:
    """
    Generate a conversational response to a user's message.

    Args:
        username: The name of the user sending the message
        message: The user's message to respond to
        style: Response style (friendly, excited, calm, sarcastic, professional)
        include_tags: Whether to include paralinguistic tags

    Returns:
        Generated response text or None if failed
    """
    style_instructions = {
        "friendly": "Be warm, friendly, and personable.",
        "excited": "Be enthusiastic and energetic!",
        "calm": "Be relaxed and soothing in tone.",
        "sarcastic": "Be playfully sarcastic but still helpful.",
        "professional": "Be polite and professional.",
    }

    tag_instruction = ""
    if include_tags:
        tag_instruction = "Include 1-2 paralinguistic tags if they feel natural."
    else:
        tag_instruction = "Do not include any paralinguistic tags."

    prompt = f"""Chat message from {username}: "{message}"

{style_instructions.get(style, style_instructions['friendly'])}
{tag_instruction}

Reply to {username} like a streamer reading chat. ONE short sentence. Use their name naturally."""

    response = generate_script(prompt, max_tokens=100)

    # Clean up any accidental label format like "username:" at the start
    if response:
        import re
        # Remove patterns like "Username:" or "username:" at the start
        response = re.sub(rf'^{re.escape(username)}\s*:\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()

        # Ensure response addresses the user
        if not re.search(rf'\b{re.escape(username)}\b', response, flags=re.IGNORECASE):
            response = f"Hey {username}, {response}"

    return response


def generate_script_for_topic(
    topic: str,
    style: str = "conversational",
    length: str = "medium",
    include_tags: bool = True,
) -> Optional[str]:
    """
    Generate a speech script for a specific topic with style options.

    Args:
        topic: The topic to speak about
        style: Speaking style (conversational, formal, excited, calm, nervous)
        length: Script length (short ~100 chars, medium ~200 chars, long ~400 chars)
        include_tags: Whether to include paralinguistic tags

    Returns:
        Generated script text or None if failed
    """
    length_guide = {
        "short": "Keep it very brief, around 50-100 characters.",
        "medium": "Keep it moderate length, around 150-250 characters.",
        "long": "Make it detailed, around 300-500 characters.",
    }

    tag_instruction = ""
    if include_tags:
        tag_instruction = "Include 2-4 paralinguistic tags naturally throughout the speech."
    else:
        tag_instruction = "Do not include any paralinguistic tags."

    prompt = f"""Write a {style} speech script about: {topic}

{length_guide.get(length, length_guide['medium'])}
{tag_instruction}

Remember: Output ONLY the speech itself, no descriptions or stage directions."""

    max_tokens = {"short": 150, "medium": 300, "long": 600}.get(length, 300)

    return generate_script(prompt, max_tokens=max_tokens)


# Preset prompts for common use cases
SCRIPT_PRESETS = {
    "podcast_intro": "Introduce yourself as a podcast host starting today's episode about technology news",
    "wedding_toast": "Give a heartfelt but slightly nervous wedding toast for your best friend",
    "product_demo": "Enthusiastically introduce a new smartphone product",
    "storytelling": "Tell a short spooky story around a campfire",
    "news_anchor": "Report breaking news about a local community event",
    "teacher": "Explain a simple concept to students in an encouraging way",
    "customer_service": "Politely help a customer with their order issue",
    "meditation": "Guide someone through a brief calming meditation",
}


def generate_from_preset(preset_name: str, **kwargs) -> Optional[str]:
    """Generate a script from a preset prompt."""
    if preset_name not in SCRIPT_PRESETS:
        logger.error(f"Unknown preset: {preset_name}")
        return None

    return generate_script_for_topic(
        topic=SCRIPT_PRESETS[preset_name],
        **kwargs
    )
