---
sidebar_position: 5
title: Multimodal Frontier
description: Vision, audio, video, and unified models — where multimodal AI actually is vs where people think it is.
---

# Multimodal Frontier

Models that process and generate across more than one modality — image + text, audio + text, or all three together. The frontier question is whether unified models that handle everything in one architecture outperform specialized pipelines.

## The problem it solves

Most real-world information isn't text. A customer service photo, an audio complaint, a scanned invoice, a screenshot of a bug — all require non-text understanding before any reasoning can happen. The traditional pipeline (OCR → text → LLM, or audio → ASR → text → LLM) loses information at every handoff and breaks for content that doesn't reduce cleanly to text (spatial relationships in images, tone in audio, motion in video).

## How it works under the hood

**Vision.** Image inputs are encoded as patch embeddings. A ViT (Vision Transformer) splits the image into fixed-size patches (e.g., 14×14 px), projects each patch into the model's token dimension, and processes them through the transformer alongside text tokens. The model learns to align image regions with text tokens during pretraining.

Some architectures use a separate vision encoder (CLIP, SigLIP) whose output is projected into the LLM's embedding space via a learned adapter. Others (GPT-4o, Gemini) are natively multimodal — images and text share the same token vocabulary from pretraining.

**Audio.** Speech is typically encoded as a log-mel spectrogram, then processed by a specialized encoder (Whisper-style) that produces token-rate representations the LLM can attend to. End-to-end audio models like GPT-4o's voice mode bypass the spectrogram step and encode audio directly into the shared token space, enabling the model to hear tone, pacing, and non-speech sounds that pure transcription loses.

**Video.** Frame sampling (1–8 fps) + ViT encoding per frame is the current standard. True temporal attention across frames is expensive; most production systems treat video as a sequence of images with a summarization step. Dense video understanding (tracking objects, understanding causation across time) remains an open research problem.

**Unified models.** Gemini 1.5 Pro, GPT-4o, and Claude 3.5+ accept interleaved text and image inputs in a single context; GPT-4o also accepts audio natively. Claude's input modalities are text and images only — audio requires a separate transcription step. Output modalities are more restricted — text generation is standard; image generation requires a separate model (DALL-E, Imagen) or native image output (Gemini Imagen integration, GPT-4o image generation).

## Concrete example

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()


def analyze_image(image_path: str, question: str) -> str:
    """Send an image to Claude and ask a question about it."""
    image_data = Path(image_path).read_bytes()
    b64_image = base64.standard_b64encode(image_data).decode()

    # Detect media type from extension
    suffix = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/jpeg")

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_image,
                    },
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }],
    )
    return response.content[0].text


def analyze_image_from_url(image_url: str, question: str) -> str:
    """Send an image URL to Claude (no base64 encoding needed)."""
    # Security: if image_url comes from user input, validate it before sending —
    # arbitrary URLs can be used for SSRF (Server-Side Request Forgery) to probe
    # internal network resources. Allowlist expected domains or validate the URL scheme.
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }],
    )
    return response.content[0].text


def extract_structured_data_from_image(image_path: str, schema: dict) -> dict:
    """
    Extract structured data from an image using vision + structured output.
    Useful for invoice parsing, form extraction, receipt OCR.
    """
    import json

    image_data = Path(image_path).read_bytes()
    b64_image = base64.standard_b64encode(image_data).decode()
    suffix = Path(image_path).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64_image},
                },
                {
                    "type": "text",
                    "text": (
                        f"Extract the following fields from this image and return as JSON:\n"
                        f"{json.dumps(schema, indent=2)}\n\n"
                        f"If a field is not visible or legible, use null. "
                        f"Return only valid JSON, no explanation."
                    ),
                },
            ],
        }],
    )

    # Validate the returned JSON against your schema before trusting field values —
    # the model may hallucinate fields or return unexpected types for numeric fields.
    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        # Strip markdown code fences if present
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())


def process_multiple_images(image_paths: list[str], task: str) -> str:
    """Process multiple images in a single context — useful for comparison tasks."""
    content = []
    for i, path in enumerate(image_paths):
        image_data = base64.standard_b64encode(Path(path).read_bytes()).decode()
        suffix = Path(path).suffix.lower()
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        content.append({
            "type": "text",
            "text": f"Image {i + 1}:",
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_data},
        })

    content.append({"type": "text", "text": task})

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


# Example usage patterns
if __name__ == "__main__":
    # 1. Basic visual QA
    # answer = analyze_image("screenshot.png", "What error is shown in this screenshot?")

    # 2. Invoice data extraction
    schema = {
        "vendor_name": "string",
        "invoice_number": "string",
        "total_amount": "number",
        "due_date": "string (YYYY-MM-DD)",
        "line_items": [{"description": "string", "amount": "number"}],
    }
    # extracted = extract_structured_data_from_image("invoice.jpg", schema)

    # 3. Multi-image comparison
    # comparison = process_multiple_images(
    #     ["before.png", "after.png"],
    #     "What changed between these two screenshots?"
    # )
    pass
```

## When to use it / when not to

**Vision is production-ready for:**
- Document understanding (invoices, receipts, forms, screenshots)
- Visual QA ("what does this chart show?", "what error is on screen?")
- Image description and alt-text generation
- Code screenshot analysis ("what does this code do?")
- Multi-image comparison

**Vision is less reliable for:**
- Precise spatial reasoning ("is the red box to the left or right of the blue box?") — models still fail on non-obvious spatial tasks
- Exact text extraction from dense, handwritten, or low-resolution documents — use a dedicated OCR tool first, then pass text to the LLM (see [[Document Processing]] for chunking and extraction pipelines)
- Real-time video analysis at meaningful frame rates — cost and latency are prohibitive
- Counting objects accurately when count > 10

**Audio:** End-to-end audio LLMs (GPT-4o voice) are impressive but still early. For production transcription, Whisper + LLM is more reliable and cheaper. Real-time voice with low latency requires specialized infrastructure (WebRTC, streaming ASR).

**Video:** Treat as image sequences. Temporal understanding is weak. Don't expect the model to track objects reliably across frames or understand cause-and-effect in video without explicit prompting.

## Main tools and libraries

| Tool | Role |
|------|------|
| Claude Vision API | Image + text understanding — strong on documents and screenshots |
| GPT-4o | Native multimodal — image, audio, video in one context |
| Whisper | Open-source ASR — production-grade speech-to-text |
| Deepgram / AssemblyAI | Managed real-time ASR with speaker diarization |
| ElevenLabs / OpenAI TTS | Text-to-speech output |
| Tesseract / AWS Textract | Dedicated OCR — better than LLM vision for dense text extraction |
| `pdf2image` | Convert PDF pages to images for vision model processing |

For document processing at scale, the winning stack is usually: dedicated OCR → structured text → LLM reasoning. Pure vision models for every document page is expensive and slower.

## Common failure modes and gotchas

**Hallucinated text in images.** Models sometimes "read" text that isn't there, or misread similar-looking characters. Always verify extracted text from images against the source when accuracy matters.

**OCR vs. vision confusion.** LLM vision is not OCR. For dense, small, or low-contrast text, use Tesseract or AWS Textract first and pass the extracted text to the LLM. Mixing modes (vision for layout + OCR for text) often outperforms pure vision.

**Token cost for images.** A 1024×1024 image costs roughly 1,590 tokens with Claude. Processing 100 invoices = significant cost. Resize images before sending; most documents don't need full resolution.

**Spatial reasoning failures.** "Is X to the left of Y?" is surprisingly unreliable. For layout-critical tasks (table extraction, form field mapping), supplement vision with structured HTML or accessibility tree data when available.

**Video as image frames loses temporal information.** Two consecutive frames showing a ball before and after a bounce look like two static images to a frame-by-frame model. Explicit temporal prompting ("Image 1 was taken 500ms before image 2") helps but doesn't fully solve the problem.

## Project ideas

- **Invoice processing pipeline**: image → vision model → structured JSON → validation → database insert. Measure extraction accuracy on 100 real invoices and track error rate by field type.
- **Screenshot bug reporter**: users screenshot a bug, the model describes it, extracts any visible error messages, and files a structured GitHub issue — bridging the gap between "I see something wrong" and a useful bug report.
- **Multi-document comparison**: given two versions of the same contract as scanned PDFs, identify what changed — using vision to handle the PDF layout and LLM reasoning to explain the significance of differences.

## Going deeper

- [Anthropic Vision docs](https://docs.anthropic.com/en/docs/build-with-claude/vision) — image input reference
- [Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)](https://arxiv.org/abs/2103.00020) — the architecture underlying most vision-language models
- [OpenAI Whisper paper](https://arxiv.org/abs/2212.04356) — robust speech recognition via large-scale weak supervision
- [Gemini Technical Report](https://arxiv.org/abs/2312.11805) — details on natively multimodal training across modalities
- [[Document Processing]] — practical chunking and extraction pipelines for document-heavy workflows
