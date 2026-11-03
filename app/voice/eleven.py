"""Thin wrapper over the ElevenLabs REST API: text to speech, dubbing projects, and agents.

Endpoints follow https://elevenlabs.io/docs/api-reference (checked 2026-10-08).
"""
import os

import httpx
from provider_safety import require_paid_services

API = "https://api.elevenlabs.io"
TTS_MODEL = "eleven_multilingual_v2"  # handles English and Spanish from the text itself
_http = httpx.Client(base_url=API, timeout=120)


class ElevenError(Exception):
    pass


def _key():
    require_paid_services()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise ElevenError("ELEVENLABS_API_KEY is not set in voice/.env.")
    return key


def _call(method, path, **kw):
    r = _http.request(method, path, headers={"xi-api-key": _key()}, **kw)
    if r.status_code >= 400:
        raise ElevenError(f"ElevenLabs {method} {path} failed ({r.status_code}): {r.text[:500]}")
    return r


# ---------- Text to speech ----------

def speak(text, voice_id):
    """Return MP3 bytes. Slightly slower than default speed: calmer for an anxious listener."""
    return _call("POST", f"/v1/text-to-speech/{voice_id}", params={"output_format": "mp3_44100_128"},
                 json={"text": text, "model_id": TTS_MODEL,
                       "voice_settings": {"stability": 0.6, "similarity_boost": 0.75, "speed": 0.95}}).content


# ---------- Dubbing (doctor's own voice, translated) ----------

def dub_create(audio, filename, content_type, target_language, source_language="en"):
    """Create a dubbing project and queue the first language. Charges one language up front."""
    return _call("POST", "/v1/dubbing/project",
                 files={"file": (filename, audio, content_type)},
                 data={"source_language": source_language, "target_language": target_language,
                       "reference": "right: doctor message"}).json()


def dub_project(project_id):
    return _call("GET", f"/v1/dubbing/project/{project_id}").json()


def dub_language(project_id, language_id):
    return _call("GET", f"/v1/dubbing/project/{project_id}/language/{language_id}").json()


def dub_transcript(project_id, language_id):
    return _call("GET", f"/v1/dubbing/project/{project_id}/language/{language_id}/transcript").json()


def download(signed_url):
    require_paid_services()
    r = httpx.get(signed_url, timeout=120, follow_redirects=True)  # signed URL: no API key
    if r.status_code >= 400:
        raise ElevenError(f"Downloading the dubbed audio failed ({r.status_code}).")
    return r.content


# ---------- Agents ----------

def create_agent(config):
    return _call("POST", "/v1/convai/agents/create", json=config).json()["agent_id"]


def update_agent(agent_id, config):
    _call("PATCH", f"/v1/convai/agents/{agent_id}", json=config)


def conversation_token(agent_id):
    """WebRTC token for one browser session. Keeps the API key on the server."""
    return _call("GET", "/v1/convai/conversation/token", params={"agent_id": agent_id}).json()


def conversation(conversation_id):
    return _call("GET", f"/v1/convai/conversations/{conversation_id}").json()
