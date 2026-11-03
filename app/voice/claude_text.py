"""Claude calls: the patient-facing trial script (English + Spanish) and teach-back grading."""
import os
from typing import Literal

import anthropic
from provider_safety import require_paid_services
from pydantic import BaseModel, Field, ValidationError

MODEL = "claude-opus-5-5"
_client = None


def client():
    require_paid_services()
    global _client
    if _client is None:
        # Reads ANTHROPIC_API_KEY from .env. A key that isn't scoped to a workspace also needs the workspace ID.
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        _client = anthropic.Anthropic(**({"default_headers": {"anthropic-workspace-id": ws}} if ws else {}))
    return _client


class KeyPoint(BaseModel):
    id: str = Field(description="Short ID: K1, K2, ...")
    en: str = Field(description="The fact as one plain English sentence")
    es: str = Field(description="The same fact as one plain Spanish sentence")


class PatientScript(BaseModel):
    plain_title: str = Field(description="What the study tests, under 10 words, plain English")
    plain_title_es: str = Field(description="The same title in plain Spanish")
    en: str = Field(description="The spoken script in English, 110 to 160 words")
    es: str = Field(description="The spoken script in Spanish, same content")
    key_points: list[KeyPoint] = Field(description="4 to 6 facts the patient should be able to repeat back")


class PointResult(BaseModel):
    id: str = Field(description="The key point ID, e.g. K1")
    result: Literal["understood", "partly", "missed"]
    evidence: str = Field(description="Short quote of what the patient said about it, in their language; empty if nothing")


class TeachbackGrade(BaseModel):
    points: list[PointResult]
    clinician_note: str = Field(description="One or two sentences in English telling the doctor what to go over again")


SCRIPT_SYSTEM = """You write the script a doctor plays aloud to a patient in a mental health clinic about one clinical \
trial the doctor thinks the patient may qualify for. The patient hears it once through a speaker, so write for the ear.

Write:
- plain_title and plain_title_es: what the study is testing, under 10 words, no jargon.
- en: 110 to 160 words at about a 6th-grade reading level, short sentences, speaking to the patient as "you". Cover what \
the study is testing and why; what taking part involves (treatment, visits, length, chance of placebo) as far as the \
record says; that joining is voluntary and saying no will not change their regular care; that their doctor thinks they \
may qualify but the study team decides after screening; and that they can ask questions now.
- es: the same script in natural, neutral Latin American Spanish, written as a native speaker would say it rather than \
translated word for word. Address the patient as "usted".
- key_points: 4 to 6 facts the patient should be able to explain back, in both languages. Always include that joining is \
voluntary, that the study team makes the final decision, and what taking part involves. Include the chance of placebo \
only if the record says there is one.

Use only facts in the trial record. If the record does not say something (payment, side effects, number of visits, \
location), leave it out rather than guess. You do not know who the patient is, so never describe their health. Never \
say the patient is eligible, qualifies, or will be accepted; say they may qualify. Keep a calm, neutral tone with no \
selling, urgency, or promised benefit. Write numbers and abbreviations the way they should be spoken, for example \
"eight weeks" rather than "8 wks"."""

GRADE_SYSTEM = """You grade a teach-back conversation from a clinic. A voice assistant asked a patient to explain a \
clinical trial in their own words, then asked follow-up questions. Grade each key point from the patient's lines only \
(role "user"); the assistant's lines are context.

- understood: the patient stated it correctly in their own words without being told it first.
- partly: partly correct, vague, or correct only after the assistant hinted at or explained it.
- missed: not mentioned, or stated wrongly.

The patient may speak Spanish. Quote their words in the language they used. Then write a short note for the doctor, in \
English, naming what to go over again. If the patient said anything suggesting distress or risk of harm, put that \
first in the note."""


def _ask(system, prompt, schema, effort):
    try:
        resp = _parse(system, prompt, schema, effort)
    except ValidationError as e:
        raise RuntimeError("Claude's reply was cut off or malformed. Try again.") from e
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"Claude returned no result (stop reason: {resp.stop_reason}).")
    return resp.parsed_output


def _parse(system, prompt, schema, effort):
    return client().beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        # Server-side fallback: if a safety classifier declines, the API retries on a recommended model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": effort},
        output_format=schema,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )


def write_script(trial_record: str) -> PatientScript:
    return _ask(SCRIPT_SYSTEM, f"Trial record:\n\n{trial_record}", PatientScript, effort="high")


def grade_teachback(key_points: list[dict], transcript: list[dict]) -> TeachbackGrade:
    points = "\n".join(f"{k['id']}: {k['en']} / {k['es']}" for k in key_points)
    lines = "\n".join(f"{t['role']}: {t['message']}" for t in transcript if t.get("message"))
    return _ask(GRADE_SYSTEM, f"Key points:\n{points}\n\nConversation:\n{lines}", TeachbackGrade, effort="medium")
