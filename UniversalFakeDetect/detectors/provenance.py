"""
Layer 1: Provenance checks (C2PA content credentials + file metadata).

These checks do not guess from pixels. They read records stored inside the file.
- A C2PA manifest that declares AI generation is strong evidence of FAKE.
- Missing C2PA / missing EXIF is NOT evidence of REAL or FAKE: messaging apps,
  screenshots and re-saving strip this data. Callers must treat "no signal"
  as "unknown", never as "real".
"""
import io
import json
from typing import Optional

from PIL import Image, ExifTags

# IPTC digital source types that mean "made or altered by generative AI".
# https://cv.iptc.org/newscodes/digitalsourcetype/
AI_SOURCE_TYPES = (
    "trainedalgorithmicmedia",               # fully AI-generated
    "compositewithtrainedalgorithmicmedia",  # real photo edited with AI
    "algorithmicmedia",
)

# Known generator names that appear in C2PA claim_generator / software agents
# or in EXIF/PNG metadata.
AI_TOOL_NAMES = (
    "openai", "chatgpt", "dall-e", "dall·e", "gpt-4o", "sora",
    "adobe firefly", "firefly", "midjourney", "stable diffusion",
    "stablediffusion", "comfyui", "automatic1111", "novelai", "leonardo",
    "imagen", "gemini", "google ai", "bing image creator", "microsoft designer",
    "flux", "ideogram", "runway", "pika", "kling", "veo",
)


def _find_ai_markers(text: str) -> list:
    t = text.lower()
    hits = [s for s in AI_SOURCE_TYPES if s in t]
    hits += [n for n in AI_TOOL_NAMES if n in t]
    # Keep the most specific source-type match only once
    return sorted(set(hits))


def check_c2pa(data: bytes, mime: Optional[str] = None) -> dict:
    """Read an embedded C2PA manifest, if present."""
    result = {"present": False, "ai_declared": False, "markers": [],
              "claim_generator": None, "validation": None, "error": None}
    try:
        import c2pa
    except ImportError:
        result["error"] = "c2pa-python not installed"
        return result

    reader = None
    try:
        stream = io.BytesIO(data)
        reader = c2pa.Reader(mime, stream) if mime else c2pa.Reader(stream)
        manifest_json = reader.json()
    except Exception as e:  # no manifest is the normal case
        msg = str(e)
        if "ManifestNotFound" not in msg and "JumbfNotFound" not in msg:
            result["error"] = msg[:200]
        return result
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass

    result["present"] = True
    try:
        store = json.loads(manifest_json)
        active = store.get("manifests", {}).get(store.get("active_manifest", ""), {})
        cg = active.get("claim_generator") or ""
        if not cg and active.get("claim_generator_info"):
            cg = ", ".join(i.get("name", "") for i in active["claim_generator_info"])
        result["claim_generator"] = cg or None
        result["validation"] = store.get("validation_state")
    except Exception:
        pass

    result["markers"] = _find_ai_markers(manifest_json)
    result["ai_declared"] = any(m in result["markers"] for m in AI_SOURCE_TYPES) or \
        any(n in result["markers"] for n in AI_TOOL_NAMES)
    return result


def check_metadata(data: bytes) -> dict:
    """Inspect EXIF and PNG text chunks."""
    result = {"camera": None, "software": None, "ai_markers": [], "has_exif": False}
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        return result

    texts = []
    try:
        exif = img.getexif()
        if exif:
            result["has_exif"] = True
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            make = str(tags.get("Make", "")).strip()
            model = str(tags.get("Model", "")).strip()
            if make or model:
                result["camera"] = f"{make} {model}".strip()
            if tags.get("Software"):
                result["software"] = str(tags["Software"]).strip()
            texts += [str(v) for v in tags.values() if isinstance(v, (str, bytes))]
    except Exception:
        pass

    # PNG tEXt/iTXt chunks: Stable Diffusion UIs write "parameters"/"prompt"/"workflow"
    for k, v in (img.info or {}).items():
        if isinstance(v, (str, bytes)):
            sv = v.decode("utf-8", "ignore") if isinstance(v, bytes) else v
            texts.append(f"{k}={sv[:2000]}")
            if str(k).lower() in ("parameters", "prompt", "workflow", "sd-metadata", "dream"):
                result["ai_markers"].append(f"png:{k}")

    result["ai_markers"] = sorted(set(result["ai_markers"] + _find_ai_markers(" ".join(texts))))
    return result


def check_provenance(data: bytes, mime: Optional[str] = None) -> dict:
    """
    Combined provenance verdict.
    verdict: "ai_declared" (strong fake evidence) | "camera_metadata" (weak real hint)
             | "none" (no usable signal)
    """
    c2 = check_c2pa(data, mime)
    md = check_metadata(data)

    if c2["ai_declared"] or md["ai_markers"]:
        verdict = "ai_declared"
    elif md["camera"]:
        verdict = "camera_metadata"
    else:
        verdict = "none"

    return {"verdict": verdict, "c2pa": c2, "metadata": md}
