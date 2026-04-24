---
sidebar_position: 7
title: Novel Interaction Paradigms
description: Ambient agents, voice-first interfaces, spatial computing — what comes after the chat box.
---

# Novel Interaction Paradigms

The chat text box is a constraint imposed by early infrastructure, not a fundamental property of AI interaction. What comes next is AI that fits into activities — driving, cooking, working — rather than requiring you to stop and type.

## The problem it solves

Text chat requires your full attention and both hands. It demands you to articulate a complete, context-free query every time. It produces output you have to read. These constraints make chat AI useless during physical tasks, and awkward for ongoing, contextual, or ambient assistance.

Novel interaction paradigms address different constraints:
- **Voice** removes the typing requirement and enables hands-free use
- **Ambient agents** remove the need to initiate — the agent notices context and acts or prompts proactively
- **Spatial computing** anchors AI to physical objects and places rather than a screen
- **Proactive agents** reverse the direction — the AI surfaces information rather than waiting to be asked

## How it works under the hood

**Voice pipeline (current).** Wake word detection → Voice Activity Detection (VAD) → Automatic Speech Recognition (ASR) → LLM → Text-to-Speech (TTS). Each hop adds latency: typical pipeline end-to-end is 1.5–3 seconds. The weak link is usually ASR (accuracy, latency) and TTS naturalness. OpenAI's Realtime API and similar end-to-end voice models collapse the ASR + LLM + TTS into a single model, reducing latency to under 600ms.

**End-to-end voice models.** Instead of transcribing speech to text and then processing text, end-to-end models accept audio input tokens directly and output audio tokens. Benefits: they can hear tone, pacing, and non-speech cues (laughter, hesitation) that transcription loses. Current limitation: harder to control, debug, and integrate with existing text-based pipelines.

**Ambient agents.** A persistent background process monitors a stream of context — calendar events, open documents, recent communications, location, time-of-day — and proactively surfaces relevant information or actions. The core engineering challenge: deciding when to interrupt vs. stay silent. Interrupting too often creates notification fatigue; too rarely and the agent is invisible.

**Spatial computing.** AR glasses or mixed-reality headsets (Apple Vision Pro, Meta Quest) provide a new rendering surface: digital overlays anchored to physical objects. An AI agent in this context can: annotate a physical machine with repair instructions when you look at it, overlay meeting notes when you enter a conference room, show navigation cues overlaid on the physical environment.

## Concrete example

```python
# Voice pipeline using Whisper (ASR) + Claude (LLM) + pyttsx3 (TTS)
# pip install openai-whisper pyaudio pyttsx3 anthropic

import anthropic
import tempfile
import wave
from pathlib import Path

client = anthropic.Anthropic()


def record_audio(duration_seconds: int = 5, sample_rate: int = 16000) -> bytes:
    """Record from the default microphone. Returns raw PCM bytes."""
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024,
    )
    frames = []
    for _ in range(int(sample_rate / 1024 * duration_seconds)):
        frames.append(stream.read(1024))
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b"".join(frames)


def save_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Save raw PCM to a temp WAV file for Whisper."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return tmp.name


_whisper_model = None  # cached after first load — loading inside the call adds 2-5s per turn


def transcribe(wav_path: str) -> str:
    """Transcribe audio with Whisper."""
    import whisper
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")  # base = fast; use "small" or "medium" for better accuracy
    result = _whisper_model.transcribe(wav_path)
    return result["text"].strip()


def speak(text: str) -> None:
    """Text-to-speech output."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)  # words per minute
    engine.say(text)
    engine.runAndWait()


def voice_conversation_loop(
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise — under 3 sentences.",
    max_turns: int = 10,
) -> None:
    """Basic voice conversation loop."""
    messages = []
    # Consent requirement: continuous microphone capture requires explicit user consent
    # and a visible recording indicator. Many jurisdictions require all-party consent
    # for audio recording. Never start recording silently.
    print("Voice assistant ready. Recording active. Press Ctrl+C to stop.")

    for _ in range(max_turns):
        print("Listening...")
        pcm = record_audio(duration_seconds=5)
        wav_path = save_wav(pcm)

        user_text = transcribe(wav_path)
        Path(wav_path).unlink(missing_ok=True)

        if not user_text or len(user_text) < 3:
            continue

        print(f"You: {user_text}")
        messages.append({"role": "user", "content": user_text})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # fast model for low latency
            max_tokens=200,
            system=system_prompt,
            messages=messages,
        )
        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        print(f"Assistant: {assistant_text}")
        speak(assistant_text)


# Ambient agent skeleton — proactive context monitoring
import time
from dataclasses import dataclass


@dataclass
class ContextSignal:
    # Consent: each source type (calendar, location, communications) requires
    # explicit user opt-in before monitoring. Do not collect signals the user
    # hasn't explicitly enabled; surface a permission model at setup time.
    source: str     # "calendar", "document", "location", "time"
    content: str
    relevance_hint: str  # what kind of assistance this might trigger


def should_interrupt(signal: ContextSignal, recent_interruptions: list[float]) -> bool:
    """
    Decide whether to surface a proactive suggestion.
    Respects a minimum gap between interruptions to avoid notification fatigue.
    """
    min_gap_seconds = 300  # at least 5 minutes between interruptions
    now = time.time()
    if recent_interruptions and (now - recent_interruptions[-1]) < min_gap_seconds:
        return False
    return True  # in practice: also score signal relevance before deciding


def generate_proactive_suggestion(signal: ContextSignal) -> str | None:
    """Generate a proactive suggestion only if genuinely useful."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system=(
            "You are an ambient assistant. Only respond if there is a clearly useful, "
            "specific, and timely action or insight to surface. "
            "If the context doesn't warrant interruption, reply with exactly: NO_INTERRUPT"
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Context signal: {signal.source}\n"
                f"Content: {signal.content}\n"
                f"Relevance hint: {signal.relevance_hint}\n\n"
                f"Should I surface a proactive suggestion? If yes, what?"
            ),
        }],
    )
    text = response.content[0].text.strip()
    return None if text == "NO_INTERRUPT" else text
```

The voice loop runs Whisper locally (base model, ~5 seconds to load on first run) and calls Claude Haiku for low-latency responses. Total pipeline latency is 1.5–4 seconds depending on hardware and response length. The ambient agent skeleton shows the core decision loop: receive a context signal, decide whether to interrupt, generate a suggestion only if warranted.

## When to use it / when not to

**Voice is right when:**
- Hands are occupied (cooking, driving, physical work)
- Users are mobile or can't look at a screen
- The interaction is short and conversational (quick answers, brief commands)
- Accessibility: users with visual impairments or motor limitations

**Voice is wrong when:**
- Precision matters — voice ASR error rates are still 5–15% in noisy environments
- Long or complex output — reading a 500-word response aloud is worse than text
- The environment is public and quiet — voice interaction in an open office disrupts others
- Security context — voice doesn't authenticate the speaker

**Ambient agents are right when:**
- Context is rich and continuously available (calendar, location, document state)
- The value is in surfacing things the user would forget to ask
- Interruption cost is low (the user isn't in a focused work state)

**Ambient agents are wrong when:**
- The interruption cost is high (deep work, meetings, driving in traffic)
- Context signals are sparse or unreliable
- Privacy is critical — ambient agents require persistent monitoring of user activity

**Current state.** Voice assistants are production-ready for constrained domains (smart speakers, car infotainment, accessibility tools). Open-domain voice conversation with frontier LLMs is usable but latency is still noticeable. Ambient agents are mostly in research/prototype stage outside of tightly scoped notification systems.

## Main tools and libraries

| Tool | Role |
|------|------|
| OpenAI Whisper | Open-source ASR — local, accurate, supports 99 languages |
| Deepgram / AssemblyAI | Managed real-time streaming ASR with low latency |
| OpenAI Realtime API | End-to-end voice (speech in, speech out) — lowest latency current option |
| ElevenLabs | High-quality neural TTS with voice cloning |
| LiveKit / WebRTC | Real-time audio/video transport for web and mobile |
| `pyaudio` / `sounddevice` | Microphone capture in Python |
| `pyttsx3` | Cross-platform offline TTS (lower quality but no API needed) |
| Apple Speech framework | Native on-device ASR on iOS/macOS — high accuracy, private |

## Common failure modes and gotchas

**Latency kills conversational feel.** The expected turn-taking cadence in human conversation is under 300ms. A 2-second pipeline latency feels like the assistant is ignoring you. Optimize the slowest step (usually ASR or TTS) before adding features. Use streaming TTS (start speaking while tokens are still generating) to reduce perceived latency.

**ASR error propagation.** A transcription error like "set a timer for 15 minutes" → "set a timer for 50 minutes" can't be caught by the LLM — it sees correct text. Always echo back what the model heard before acting on high-stakes commands.

**Wake word false positives.** Always-on wake word detection (like "Hey Siri") triggers on similar sounds in conversation or media. In a shared environment, this causes privacy leakage and unwanted activations. Rate-limit activations and require confirmation for anything with side effects.

**Ambient agent notification fatigue.** An agent that interrupts more than once every 10 minutes will be disabled within a week. Tune the interrupt threshold aggressively toward silence; err on the side of not interrupting.

**Voice in noisy environments.** Whisper base/small fails significantly in background noise. Use a Voice Activity Detector (VAD) like Silero-VAD to only send audio segments that contain speech, and deploy the larger Whisper model in noisy environments.

## Project ideas

- **Cooking assistant**: voice-only interface that reads recipe steps, answers ingredient questions, and sets timers — triggered by wake word, no screen needed.
- **Meeting summarizer**: ambient agent that transcribes a meeting in the background (local Whisper, privacy-preserving), generates a summary and action items when the meeting ends, and posts to Slack.
- **Spatial annotation tool**: on Apple Vision Pro or Quest, build an app that lets users attach voice notes to physical objects ("look at this machine part, say 'note: replace every 3 months'") and retrieves them when looking at the same object later.

## Going deeper

- [OpenAI Realtime API documentation](https://platform.openai.com/docs/guides/realtime) — end-to-end voice with function calling
- [Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)](https://arxiv.org/abs/2212.04356)
- [LiveKit agents framework](https://github.com/livekit/agents) — open-source real-time voice agent infrastructure
- [Proactive Information Delivery — Google Research survey](https://research.google/pubs/) — literature on when and how ambient agents should interrupt
- [[Agentic Computer Use]] — physical-world action layer that spatial agents need
