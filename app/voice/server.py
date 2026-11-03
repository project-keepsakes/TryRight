"""Triright voice service: Claude trial script + ElevenLabs speech, dubbing, Q&A agent, and teach-back.

Local:  from app/voice, .venv/bin/uvicorn server:app --port 8001   then open http://localhost:8001
Vercel: app/api/voice.py serves this app under /voice on the same site as the trial match.

No patient record goes to Claude or ElevenLabs: scripts and agents are built from trial information only. During the
Q&A and teach-back calls, the patient's own voice goes to ElevenLabs, and the teach-back transcript goes to Claude.
"""
import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.responses import JSONResponse
from provider_safety import PaidServicesDisabled, require_paid_services
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).parent
for env_file in (HERE / ".env", HERE.parent / ".env", HERE.parent.parent / ".env"):  # voice/, app/, or repo level
    load_dotenv(env_file)
ON_VERCEL = bool(os.environ.get("VERCEL"))

import agents  # noqa: E402  (after .env is loaded)
import claude_text  # noqa: E402
import eleven  # noqa: E402

LANGS = {"en": "English", "es": "Spanish"}
STORE_FILE = Path("/tmp/voice-cache.json") if ON_VERCEL else HERE / ".cache.json"  # Vercel's disk is read-only
# One JSON file: trials, draft and approved scripts. Survives restarts, so Claude isn't re-billed for a demo re-run.
STORE = json.loads(STORE_FILE.read_text()) if STORE_FILE.exists() else {"trials": {}, "draft": {}, "approved": {}}
AUDIO = {}  # (trial_id, language, text hash) -> mp3 bytes; replays don't re-bill ElevenLabs

app = FastAPI(title="Triright voice")


@app.middleware("http")
async def paid_provider_gate(request, call_next):
    path = request.scope.get("path", request.url.path)
    root_path = request.scope.get("root_path", "").rstrip("/")
    if root_path and path.startswith(root_path + "/"):
        path = path[len(root_path):]
    public_read = request.method in ("GET", "HEAD") and (
        path == "/api/config" or path.startswith("/api/trials/")
    )
    if path.startswith("/api/") and not public_read:
        try:
            require_paid_services()
        except PaidServicesDisabled:
            return JSONResponse({"error": "Paid services are disabled"}, status_code=503)
    return await call_next(request)
# The trial-match app in ../app runs on port 8000 and can call this service directly.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


_save_lock = threading.Lock()


def save():
    """Write to a temp file, then swap it in, so a crash mid-write can't corrupt the cache."""
    with _save_lock:
        data = json.dumps(STORE, indent=1, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=STORE_FILE.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, STORE_FILE)


def fail(e, code=502):
    raise HTTPException(code, str(e)) from e


def need(var):
    if not os.environ.get(var):
        where = "the Vercel project's environment variables" if ON_VERCEL else "the .env file"
        raise HTTPException(503, f"{var} is not set in {where}.")


def doctor_voice():
    return os.environ.get("ELEVENLABS_DOCTOR_VOICE_ID") or agents.DOCTOR_VOICE


# ---------- Trials ----------

FIELDS = ("NCTId,BriefTitle,BriefSummary,DetailedDescription,Phase,Condition,InterventionName,LeadSponsorName,"
          "EnrollmentCount,MinimumAge,MaximumAge,Sex,EligibilityCriteria,LocationFacility,LocationCity,LocationState")


def fetch_trial(nct_id):
    """Same trial shape as app/app.py, so a trial from the match app can be posted here unchanged."""
    url = f"https://clinicaltrials.gov/api/v2/studies/{urllib.parse.quote(nct_id)}"
    r = httpx.get(url, params={"fields": FIELDS}, timeout=20)
    if r.status_code == 404:
        raise HTTPException(404, f"{nct_id} was not found on ClinicalTrials.gov.")
    r.raise_for_status()
    p = r.json()["protocolSection"]
    e = p.get("eligibilityModule", {})
    locs = p.get("contactsLocationsModule", {}).get("locations", [])
    ny = [l for l in locs if l.get("state") == "New York"] or locs
    return {
        "nctId": p["identificationModule"]["nctId"], "title": p["identificationModule"]["briefTitle"],
        "site": f"{ny[0].get('facility', 'Site')}, {ny[0].get('city', '')}" if ny else None, "nSites": len(locs),
        "brief": p.get("descriptionModule", {}).get("briefSummary", ""),
        "detail": p.get("descriptionModule", {}).get("detailedDescription", "")[:3000],
        "sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
        "phase": ", ".join(p.get("designModule", {}).get("phases", [])) or "N/A",
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "interventions": [i["name"] for i in p.get("armsInterventionsModule", {}).get("interventions", [])],
        "enrollment": p.get("designModule", {}).get("enrollmentInfo", {}).get("count"),
        "minAge": e.get("minimumAge", ""), "maxAge": e.get("maximumAge", ""), "sex": e.get("sex", ""),
        "criteria": e.get("eligibilityCriteria", ""),
    }


def facts(t):
    """The trial record as plain text, for Claude and for the Q&A agent."""
    c = t.get("criteria_summary") or {}
    rows = [
        ("Official title", t.get("title")), ("ClinicalTrials.gov ID", t.get("nctId")), ("Sponsor", t.get("sponsor")),
        ("Phase", t.get("phase")), ("Conditions", ", ".join(t.get("conditions") or [])),
        ("Treatments studied", ", ".join(t.get("interventions") or [])),
        ("Ages", f"{t.get('minAge') or 'any'} to {t.get('maxAge') or 'no upper limit'}"), ("Sex", t.get("sex")),
        ("Planned enrollment", t.get("enrollment")), ("Nearest site", t.get("site")),
        ("Overview", c.get("overview")),
        ("Who can join", "; ".join(x.get("criterion", "") for x in c.get("inclusion") or [])),
        ("Who cannot join", "; ".join(x.get("criterion", "") for x in c.get("exclusion") or [])),
        ("What taking part involves", c.get("involves")), ("Payment", c.get("compensation")),
        ("Benefits", c.get("benefits")),
        ("Summary", t.get("brief")), ("Description", (t.get("detail") or "")[:2500]),
        ("Eligibility text", (t.get("criteria") or "")[:3000]),
    ]
    return "\n".join(f"{k}: {v}" for k, v in rows if v)


def get_trial(trial_id):
    if trial_id not in STORE["trials"]:
        STORE["trials"][trial_id] = fetch_trial(trial_id)
        save()
    return STORE["trials"][trial_id]


class Context(BaseModel):
    """What the page already holds. On Vercel each request may reach a fresh instance with an empty cache, so the
    panel sends the trial and the approved script along, and the instance picks up from there."""
    trial: dict | None = None
    criteria_summary: dict | None = None
    script: dict | None = None  # the script the doctor approved, as shown in the panel


def register(trial, criteria_summary=None):
    t = {k: v for k, v in trial.items() if v not in (None, "", [])}
    if not t.get("nctId"):
        raise HTTPException(422, "The trial needs an nctId.")
    if t["nctId"].startswith("NCT"):  # match results omit the full eligibility text; fill it in from the registry
        try:
            t = {**fetch_trial(t["nctId"]), **t}
        except (httpx.HTTPError, HTTPException):
            pass
    t["criteria_summary"] = criteria_summary
    STORE["trials"][t["nctId"]] = t
    return t


def remember(trial_id, ctx):
    if ctx is None:
        return
    if ctx.trial and trial_id not in STORE["trials"]:
        register(ctx.trial, ctx.criteria_summary)
    if ctx.script and trial_id not in STORE["approved"]:
        script = claude_text.PatientScript(**ctx.script).model_dump()
        STORE["approved"][trial_id] = dict(script, approved_at=ctx.script.get("approved_at", ""))


@app.get("/api/config")
def config():
    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "elevenlabs": bool(os.environ.get("ELEVENLABS_API_KEY")),
        "qa_agent": bool(agents.agent_id("qa")),
        "teachback_agent": bool(agents.agent_id("teachback")),
        "languages": LANGS,
    }


@app.get("/api/trials/{trial_id}")
def trial(trial_id: str):
    try:
        t = get_trial(trial_id)
    except httpx.HTTPError as e:
        fail(e)
    return {"trial": t, "draft": STORE["draft"].get(trial_id), "approved": STORE["approved"].get(trial_id)}


class TrialIn(BaseModel):
    trial: dict
    criteria_summary: dict | None = None  # optional: the match app's criteria summary (involves, compensation, ...)


@app.post("/api/trials")
def put_trial(body: TrialIn):
    """Register a trial from the match app, including the team reference trial that isn't on ClinicalTrials.gov.
    Returns any script already drafted or approved for it."""
    t = register(body.trial, body.criteria_summary)
    save()
    tid = t["nctId"]
    return {"trial_id": tid, "draft": STORE["draft"].get(tid), "approved": STORE["approved"].get(tid)}


@app.post("/api/trials/{trial_id}/script")
def draft_script(trial_id: str, refresh: bool = False, ctx: Context | None = None):
    """Claude drafts the patient script. The doctor must approve it before anything is played."""
    remember(trial_id, ctx)
    if refresh or trial_id not in STORE["draft"]:
        need("ANTHROPIC_API_KEY")
        try:
            STORE["draft"][trial_id] = claude_text.write_script(facts(get_trial(trial_id))).model_dump()
        except Exception as e:
            fail(e)
        save()
    return STORE["draft"][trial_id]


@app.post("/api/trials/{trial_id}/approve")
def approve(trial_id: str, script: claude_text.PatientScript):
    """Store the doctor-edited script. Speech and both agents only ever use the approved version."""
    STORE["approved"][trial_id] = dict(script.model_dump(), approved_at=time.strftime("%Y-%m-%d %H:%M"))
    save()
    return STORE["approved"][trial_id]


def approved(trial_id):
    s = STORE["approved"].get(trial_id)
    if not s:
        raise HTTPException(409, "The doctor hasn't approved a script for this trial yet.")
    return s


class SpeechIn(Context):
    language: str = "en"


@app.post("/api/trials/{trial_id}/speech")
def speech(trial_id: str, body: SpeechIn):
    if body.language not in LANGS:
        raise HTTPException(400, f"Language must be one of {list(LANGS)}.")
    remember(trial_id, body)
    text = approved(trial_id)[body.language]
    need("ELEVENLABS_API_KEY")
    key = (trial_id, body.language, hashlib.sha256(text.encode()).hexdigest())
    if key not in AUDIO:
        try:
            AUDIO[key] = eleven.speak(text, doctor_voice())
        except eleven.ElevenError as e:
            fail(e)
    return Response(AUDIO[key], media_type="audio/mpeg")


# ---------- A: doctor's message, dubbed in the doctor's own voice ----------

SAMPLE_MESSAGE = ("Hi, I'm your doctor. I'd like to tell you about a research study that might be a good fit for you. "
                  "Taking part is completely your choice, and your care here stays the same either way. "
                  "Please listen to the short summary, and ask me anything you'd like.")


@app.post("/api/sample-doctor-message")
def sample_doctor_message():
    """Until we have a real recording, the stand-in doctor voice says a short English message to dub."""
    need("ELEVENLABS_API_KEY")
    key = ("sample", "en", hashlib.sha256(SAMPLE_MESSAGE.encode()).hexdigest())
    if key not in AUDIO:
        try:
            AUDIO[key] = eleven.speak(SAMPLE_MESSAGE, doctor_voice())
        except eleven.ElevenError as e:
            fail(e)
    return Response(AUDIO[key], media_type="audio/mpeg")


@app.post("/api/dub")
async def dub(file: UploadFile = File(...), target_language: str = Form("es")):
    need("ELEVENLABS_API_KEY")
    try:
        p = eleven.dub_create(await file.read(), file.filename or "message.webm",
                              file.content_type or "audio/webm", target_language)
    except eleven.ElevenError as e:
        fail(e)
    return {"project_id": p["project_id"]}


@app.get("/api/dub/{project_id}")
def dub_status(project_id: str):
    try:
        p = eleven.dub_project(project_id)
        out = {"project_status": p["status"], "status": p["status"], "segments": None, "error": p.get("error")}
        if p["status"] == "failed" or not p.get("language_ids"):
            return out
        lang = eleven.dub_language(project_id, p["language_ids"][0])
        out.update(status=lang["status"], error=lang.get("error") or out["error"])
        if lang["status"] == "completed":
            tr = eleven.dub_transcript(project_id, p["language_ids"][0])
            out["segments"] = [{"source": s["source_text"], "translation": s.get("translation")}
                               for s in tr["segments"]]
        return out
    except eleven.ElevenError as e:
        fail(e)


@app.get("/api/dub/{project_id}/audio")
def dub_audio(project_id: str):
    """Proxy the dubbed FLAC: ElevenLabs' signed URL expires after an hour, so fetch a fresh one each time."""
    try:
        p = eleven.dub_project(project_id)
        if not p.get("language_ids"):
            raise HTTPException(409, f"The dub isn't ready yet (status: {p['status']}).")
        lang = eleven.dub_language(project_id, p["language_ids"][0])
        url = (lang.get("outputs") or {}).get("lossless_audio")
        if not url:
            raise HTTPException(409, f"The dub isn't ready yet (status: {lang['status']}).")
        return Response(eleven.download(url), media_type="audio/flac")
    except eleven.ElevenError as e:
        fail(e)
    except httpx.HTTPError:
        raise HTTPException(502, "Couldn't download the dubbed audio from ElevenLabs. Try again.")


# ---------- B and C: voice agents ----------

def dynamic_variables(kind, trial_id, language):
    s, t = approved(trial_id), get_trial(trial_id)
    title = s["plain_title_es"] if language == "es" else s["plain_title"]
    if kind == "qa":
        return {"trial_title": title, "trial_id": trial_id, "trial_summary": s[language], "trial_facts": facts(t)}
    return {"trial_title": title, "trial_summary": s[language],
            "key_points": "\n".join(f"- {k[language]}" for k in s["key_points"])}


class SessionIn(Context):
    trial_id: str
    language: str = "en"


@app.post("/api/agents/{kind}/session")
def agent_session_post(kind: str, body: SessionIn):
    remember(body.trial_id, body)
    return agent_session(kind, body.trial_id, body.language)


@app.get("/api/agents/{kind}/session")
def agent_session(kind: str, trial_id: str, language: str = "en"):
    """A one-time WebRTC token plus the trial details the agent needs. The API key never reaches the browser."""
    if kind not in agents.AGENTS:
        raise HTTPException(404, "Unknown agent.")
    agent_id = agents.agent_id(kind)
    if not agent_id:
        raise HTTPException(503, f"{agents.AGENTS[kind]['env']} is not set. Run `python agents.py` first.")
    if language not in LANGS:
        raise HTTPException(400, f"Language must be one of {list(LANGS)}.")
    variables = dynamic_variables(kind, trial_id, language)
    need("ELEVENLABS_API_KEY")
    try:
        token = eleven.conversation_token(agent_id)
    except eleven.ElevenError as e:
        fail(e)
    return {"token": token["token"], "dynamic_variables": variables, "language": language}


class GradeIn(Context):
    conversation_id: str
    trial_id: str


@app.post("/api/teachback/grade")
def grade(body: GradeIn):
    """Fetch the finished teach-back transcript, then have Claude score each key point."""
    remember(body.trial_id, body)
    key_points = approved(body.trial_id)["key_points"]
    need("ANTHROPIC_API_KEY")
    try:
        for _ in range(15):  # the transcript can take a few seconds to settle after the call ends
            conv = eleven.conversation(body.conversation_id)
            if conv["status"] in ("done", "failed"):
                break
            time.sleep(2)
        transcript = [{"role": t["role"], "message": t.get("message")} for t in conv.get("transcript", [])]
        if not any(t["role"] == "user" and t["message"] for t in transcript):
            raise HTTPException(422, "The patient didn't say anything in this conversation, so there is nothing to grade.")
        g = claude_text.grade_teachback(key_points, transcript).model_dump()
    except eleven.ElevenError as e:
        fail(e)
    except HTTPException:
        raise
    except Exception as e:
        fail(e)
    text = {k["id"]: k["en"] for k in key_points}
    for p in g["points"]:
        p["point"] = text.get(p["id"], p["id"])
    g.update(understood=sum(p["result"] == "understood" for p in g["points"]),
             partly=sum(p["result"] == "partly" for p in g["points"]), total=len(key_points),
             transcript=transcript)
    return g


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    return FileResponse(HERE / "static" / "index.html")
