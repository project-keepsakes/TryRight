"""Trial pre-screening demo: ClinicalTrials.gov trial -> Claude criteria summary -> per-patient screen.

Run:  uv run --with anthropic app.py   then open http://localhost:8000
"""
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic
from voice.provider_safety import PaidServicesDisabled, require_paid_services

HERE = Path(__file__).parent
MODEL = "claude-opus-5-5"
DATA_FILE = HERE / "synthetic-patient-data.md"
CACHE_FILE = HERE / ".cache.json"

for env_file in (HERE / ".env", HERE.parent / ".env"):  # app/.env, or the repo-level .env shared with voice/
    for line in (env_file.read_text().splitlines() if env_file.exists() else []):
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
ON_VERCEL = bool(os.environ.get("VERCEL"))
SNAPSHOT = HERE / "snapshot"  # frozen trials + parsed criteria for the Vercel deploy (read-only disk)
if ON_VERCEL:
    CACHE_FILE = Path("/tmp/cache.json")
# An organization key that isn't scoped to a workspace must name one on every request.
WORKSPACE = os.environ.get("ANTHROPIC_WORKSPACE_ID")
client = None


def get_client():
    require_paid_services()
    global client
    client = client or anthropic.Anthropic(**({"default_headers": {"anthropic-workspace-id": WORKSPACE}} if WORKSPACE else {}))
    return client


# ponytail: one JSON file cache, so a demo re-run is instant and survives a network drop
_seed = CACHE_FILE if CACHE_FILE.exists() else SNAPSHOT / "cache.json"
CACHE = json.loads(_seed.read_text()) if _seed.exists() else {}
CACHE_LOCK = threading.Lock()


def clean(s):
    return re.sub(r"\\(.)", r"\1", s.replace("**", "")).strip()


def load_patients():
    md = DATA_FILE.read_text()
    section1 = md.split("## **Section 2")[0]
    labels = r"(Profile|Situation|Symptoms|GAD-7 / PHQ-9|Duration|Treatment|Meds|Flags):"
    patients = []
    for block in section1.split("### **")[1:]:
        head, _, body = block.partition("\n")
        pid, name = clean(head).split(":", 1)
        parts = re.split(labels, clean(body).replace("|", "\n"))
        fields = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        m = re.match(r"(\d+)\s*([MF])", fields.get("Profile", ""))
        g = re.match(r"(\d+)\s*/\s*(\d+)", fields.get("GAD-7 / PHQ-9", ""))
        patients.append({
            "id": pid.strip(), "name": name.strip(),
            "age": int(m.group(1)) if m else None, "sex": m.group(2) if m else "",
            "gad7": int(g.group(1)) if g else None, "phq9": int(g.group(2)) if g else None,
            "fields": fields,
            "record": f"{pid.strip()}: {name.strip()}\n" + "\n".join(f"{k}: {v}" for k, v in fields.items()),
        })
    return patients


def reference_trial():
    md = DATA_FILE.read_text()
    crit = md.split("### **2.2")[1].split("## **Section 3")[0]
    return {
        "nctId": "REF-GAD-01", "title": "Reference trial: Moderate GAD (team protocol from data file)",
        "sponsor": "Internal", "phase": "N/A", "conditions": ["Generalized Anxiety Disorder"],
        "enrollment": None, "minAge": "18 Years", "maxAge": "65 Years", "sex": "ALL",
        "criteria": clean("### 2.2" + crit), "interventions": [],
        "brief": "Team reference protocol for adults with moderate generalized anxiety disorder.",
        "detail": "", "site": "Team clinic (internal protocol)", "nSites": 1,
    }


FIELDS = ("NCTId,BriefTitle,BriefSummary,DetailedDescription,Phase,Condition,InterventionName,LeadSponsorName,"
          "EnrollmentCount,MinimumAge,MaximumAge,Sex,EligibilityCriteria,ContactsLocationsModule")
CITY = ("New York City", 40.7580, -73.9855, 15)  # Midtown Manhattan, radius in miles


def miles(lat1, lon1, lat2, lon2):
    import math
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def nearest_site(p):
    """PRD step 1: overall status is not enough; need a RECRUITING site inside the radius."""
    _, lat, lon, radius = CITY
    sites = []
    for loc in p.get("contactsLocationsModule", {}).get("locations", []):
        g = loc.get("geoPoint")
        if loc.get("status") == "RECRUITING" and g:
            d = miles(lat, lon, g["lat"], g["lon"])
            if d <= radius:
                sites.append((d, f"{loc.get('facility', 'Site')}, {loc.get('city', '')}"))
    sites.sort()
    return (f"{sites[0][1]} ({sites[0][0]:.1f} mi)", len(sites)) if sites else (None, 0)


def fetch_trials(params):
    url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode({**params, "fields": FIELDS})
    with urllib.request.urlopen(url, timeout=20) as r:
        studies = json.load(r).get("studies", [])
    out = []
    for s in studies:
        p = s["protocolSection"]
        e = p.get("eligibilityModule", {})
        site, n_sites = nearest_site(p)
        out.append({
            "nctId": p["identificationModule"]["nctId"], "title": p["identificationModule"]["briefTitle"],
            "site": site, "nSites": n_sites,
            "brief": p.get("descriptionModule", {}).get("briefSummary", ""),
            "detail": p.get("descriptionModule", {}).get("detailedDescription", "")[:3000],
            "sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
            "phase": ", ".join(p.get("designModule", {}).get("phases", [])) or "N/A",
            "conditions": p.get("conditionsModule", {}).get("conditions", []),
            "interventions": [i["name"] for i in p.get("armsInterventionsModule", {}).get("interventions", [])],
            "enrollment": p.get("designModule", {}).get("enrollmentInfo", {}).get("count"),
            "minAge": e.get("minimumAge", ""), "maxAge": e.get("maximumAge", ""), "sex": e.get("sex", ""),
            "criteria": e.get("eligibilityCriteria", ""),
        })
    return out


def search_trials(term):
    q = {"query.cond": "anxiety", "filter.overallStatus": "RECRUITING", "pageSize": "40"}
    if term:
        q["query.term"] = term
    return fetch_trials(q)


def demo_trials():
    name, lat, lon, radius = CITY
    live = fetch_trials({"query.cond": "anxiety", "filter.overallStatus": "RECRUITING",
                         "filter.geo": f"distance({lat},{lon},{radius}mi)", "pageSize": "100"})
    return [reference_trial()] + [t for t in live if t["site"]]


CRITERION = {"type": "object", "properties": {
    "id": {"type": "string", "description": "Short ID such as I-1 or E-3"},
    "criterion": {"type": "string", "description": "The criterion, condensed to one line"},
    "plain": {"type": "string", "description": "What the physician must check in the chart, one short line"},
}, "required": ["id", "criterion", "plain"]}

SUMMARY_TOOL = {"name": "criteria_summary", "description": "Structured eligibility summary for a physician.",
                "input_schema": {"type": "object", "properties": {
                    "overview": {"type": "string", "description": "2-3 sentences: who the trial wants and why"},
                    "inclusion": {"type": "array", "items": CRITERION},
                    "exclusion": {"type": "array", "items": CRITERION},
                    "screening_notes": {"type": "array", "items": {"type": "string"},
                                        "description": "Criteria that need a manual screen, interview, or lab"},
                    "compensation": {"type": "string", "description":
                                     "Payment, reimbursement, or benefits to participants (free treatment, devices, "
                                     "travel costs) as stated in the record. If none is stated, say exactly: "
                                     "'Not stated in the registry record; ask the study team.' Never guess an amount."},
                    "benefits": {"type": "string", "description":
                                 "Participant benefits the record states, e.g. study drug or therapy at no cost, free "
                                 "app access, free assessments, close clinical monitoring. One line. Only what the "
                                 "record supports; if nothing, say 'None stated in the registry record.'"},
                    "involves": {"type": "string", "description": "One line: intervention, visits, duration, placebo"},
                }, "required": ["overview", "inclusion", "exclusion", "screening_notes", "compensation", "benefits",
                                "involves"]}}

SCREEN_TOOL = {"name": "screen_result", "description": "Eligibility decision for one patient.",
               "input_schema": {"type": "object", "properties": {
                   "status": {"type": "string", "enum": ["ELIGIBLE", "VERIFY", "EXCLUDED"],
                              "description": "EXCLUDED if any inclusion criterion is clearly NOT_MET or any "
                                             "exclusion criterion clearly applies. VERIFY if no hard failure but a "
                                             "criterion cannot be confirmed from the record. Else ELIGIBLE."},
                   "match_score": {"type": "integer", "minimum": 0, "maximum": 100},
                   "summary": {"type": "string", "description": "1-2 sentences for the physician"},
                   "criteria": {"type": "array", "items": {"type": "object", "properties": {
                       "id": {"type": "string"},
                       "result": {"type": "string", "enum": ["MET", "NOT_MET", "UNKNOWN"],
                                  "description": "For exclusions: NOT_MET means the patient is clear of it"},
                       "evidence": {"type": "string", "description": "Fact from the record, short"},
                   }, "required": ["id", "result", "evidence"]}},
                   "open_checks": {"type": "array", "items": {"type": "string"}},
               }, "required": ["status", "match_score", "summary", "criteria", "open_checks"]}}


def call_tool(tool, system, prompt):
    key = hashlib.sha256(json.dumps([tool["name"], system, prompt]).encode()).hexdigest()
    if key in CACHE:
        return CACHE[key]
    msg = get_client().messages.create(
        # Opus 5.5 rejects a forced tool_choice, so ask for the tool in the system prompt instead.
        model=MODEL, max_tokens=4000, system=system + f" Respond only by calling the {tool['name']} tool.",
        tools=[tool], tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": prompt}],
    )
    out = next(b.input for b in msg.content if b.type == "tool_use")
    with CACHE_LOCK:
        CACHE[key] = out
        CACHE_FILE.write_text(json.dumps(CACHE))
    return out


def summarize(trial):
    return call_tool(
        SUMMARY_TOOL,
        "You are a clinical research coordinator in psychiatry. Summarize trial eligibility for a busy "
        "physician. Keep each criterion to one line. Keep every numeric threshold exactly as written. "
        "Do not invent criteria that are not in the text; demographic limits given separately count as criteria. "
        "Report compensation and participant benefits only as the record states them.",
        f"Trial {trial['nctId']}: {trial['title']}\nSex: {trial['sex']}  Age: {trial['minAge']} to "
        f"{trial['maxAge'] or 'no max'}\n\nSummary:\n{trial.get('brief', '')}\n\nDescription:\n"
        f"{trial.get('detail', '')}\n\nEligibility text:\n{trial['criteria']}",
    )


def screen(summary, patient):
    crit = "\n".join(f"{c['id']}: {c['criterion']}" for c in summary["inclusion"] + summary["exclusion"])
    return call_tool(
        SCREEN_TOOL,
        "You pre-screen synthetic patients for a clinical trial. Judge each criterion only from the record. "
        "If the record does not state a fact, use UNKNOWN; never assume. Exception: the record is the patient's "
        "full problem summary, so a required condition or diagnosis that the record does not show (for example "
        "cancer, a scheduled surgery, pregnancy) is NOT_MET, not UNKNOWN. Be conservative: a pending check "
        "means VERIFY, not ELIGIBLE. Return one entry per criterion ID.",
        f"Criteria:\n{crit}\n\nPatient record:\n{patient['record']}",
    )


def years(age_text):
    m = re.match(r"(\d+)\s*(Year|Month|Week)", age_text or "")
    return None if not m else int(m.group(1)) / (1 if m.group(2) == "Year" else 12 if m.group(2) == "Month" else 52)


def hard_gate(t, patient):
    """PRD step 2: drop on structured age/sex with no LLM call."""
    lo, hi, age = years(t["minAge"]), years(t["maxAge"]), patient["age"]
    if lo and age < lo:
        return f"Age {age} is below the minimum age of {t['minAge']}."
    if hi and age > hi:
        return f"Age {age} is above the maximum age of {t['maxAge']}."
    if t["sex"] in ("MALE", "FEMALE") and patient["sex"] != t["sex"][0]:
        return f"Trial enrolls {t['sex'].lower()} participants only."
    return None


def match(patient):
    """All NYC trials for one patient, best match first; excluded trials last."""
    def one(t):
        summary = summarize(t)
        gate = hard_gate(t, patient)
        result = screen(summary, patient) if not gate else {
            "status": "EXCLUDED", "match_score": 0, "summary": gate, "open_checks": [],
            "criteria": [{"id": "Age/Sex", "result": "NOT_MET", "evidence": gate}]}
        return {"trial": {k: v for k, v in t.items() if k not in ("criteria", "detail")}, "criteria": summary,
                "result": result}
    with ThreadPoolExecutor(len(TRIALS)) as ex:  # all trials at once: ~20s instead of ~70s
        rows = list(ex.map(one, TRIALS))
    rank = {"ELIGIBLE": 0, "VERIFY": 1, "EXCLUDED": 2}
    return sorted(rows, key=lambda r: (rank[r["result"]["status"]], -r["result"]["match_score"]))


def prewarm():
    # Parse each trial's criteria once at startup (PRD step 3). Patient matching runs only on "Run match".
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(summarize, TRIALS))
    print("prewarm done", flush=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE / "static"), **kw)

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path == "/api/patients":
                return self.send_json([{k: v for k, v in p.items() if k != "record"} for p in PATIENTS])
            if u.path == "/api/demo-trials":
                return self.send_json(TRIALS)
            if u.path == "/api/trials":
                term = urllib.parse.parse_qs(u.query).get("q", [""])[0]
                return self.send_json([reference_trial()] + search_trials(term))
        except Exception as e:  # surface API failures to the UI instead of a dead socket
            return self.send_json({"error": str(e)}, 502)
        return super().do_GET()

    def do_POST(self):
        if self.path not in ("/api/summarize", "/api/match", "/api/screen"):
            return self.send_json({"error": "not found"}, 404)
        try:
            require_paid_services()
        except PaidServicesDisabled:
            return self.send_json({"error": "Paid services are disabled"}, 503)
        lengths = self.headers.get_all("Content-Length", [])
        if (len(lengths) != 1 or len(lengths[0]) > 8 or
                not re.fullmatch(r"[0-9]+", lengths[0]) or self.headers.get("Transfer-Encoding")):
            return self.send_json({"error": "Invalid request framing"}, 400)
        length = int(lengths[0])
        if length > 16384:
            return self.send_json({"error": "Request body is too large"}, 413)
        try:
            self.connection.settimeout(5)
            raw = self.rfile.read(length)
            if len(raw) != length:
                return self.send_json({"error": "Incomplete request body"}, 400)
            body = json.loads(raw)
            if not isinstance(body, dict):
                return self.send_json({"error": "A JSON object is required"}, 400)
        except (ValueError, UnicodeError, RecursionError, TimeoutError, OSError):
            return self.send_json({"error": "Invalid request body"}, 400)
        try:
            if self.path == "/api/summarize":
                return self.send_json(summarize(body["trial"]))
            if self.path == "/api/match":
                return self.send_json(match(next(p for p in PATIENTS if p["id"] == body["patientId"])))
            if self.path == "/api/screen":
                p = next(p for p in PATIENTS if p["id"] == body["patientId"])
                return self.send_json(screen(body["summary"], p))
        except (KeyError, StopIteration, TypeError):
            return self.send_json({"error": "Invalid request fields"}, 400)
        except Exception:
            return self.send_json({"error": "Trial screening is unavailable"}, 502)
        self.send_json({"error": "not found"}, 404)


PATIENTS = load_patients()
TRIALS = json.loads((SNAPSHOT / "trials.json").read_text()) if ON_VERCEL else demo_trials()

if __name__ == "__main__":
    assert len(PATIENTS) == 24 and PATIENTS[0]["gad7"] == 8 and PATIENTS[3]["fields"].get("Meds"), PATIENTS[0]
    assert len(TRIALS) > 6 and hard_gate({"minAge": "18 Years", "maxAge": "65 Years", "sex": "ALL"},
                                          {"age": 71, "sex": "F"}), len(TRIALS)
    if os.environ.get("TRIRIGHT_ALLOW_PAID_SERVICES") == "true" and not ON_VERCEL:
        threading.Thread(target=prewarm, daemon=True).start()
    print(f"{len(PATIENTS)} patients, {len(TRIALS)} trials loaded. Open http://localhost:8000", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
