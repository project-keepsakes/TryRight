# Triright: trial match demo

The clinician selects a synthetic patient in the "Triright · Trial Match" tab of the ClinicView EHR mockup and clicks "Run trial match". The app screens that patient against every recruiting anxiety trial with a recruiting site within 15 miles of Midtown Manhattan (live from ClinicalTrials.gov API v2), plus the team reference trial. Each trial gets a match score and a status. Expand a row to see the trial summary and the criterion-by-criterion reasoning.

## How it works

1. At startup, the server gets the trials from ClinicalTrials.gov and keeps only trials with a RECRUITING site inside the radius. Claude Opus 5.5 splits each trial's eligibility text into criteria once.
2. On "Run trial match", age and sex are checked in code first (no model call). Claude then checks the patient against each remaining trial in parallel. A missing fact becomes "confirm with patient", not a pass.

## Run

Paid providers are disabled by default, even when API keys exist. Local
reproduction requires an explicit `TRIRIGHT_ALLOW_PAID_SERVICES=true` opt-in;
this can incur provider charges. Hosted Vercel execution always rejects paid
operations. The archived repository does not manage or retire older deployments.

1. Put the key in `app/.env` as `ANTHROPIC_API_KEY=...`. This file is in `.gitignore`; do not commit it.
2. Start the server from the `app` folder:

```bash
uv run --with anthropic app.py
```

3. Wait for `prewarm done` in the output (about 40 seconds), then open http://localhost:8000.

Results are cached in `.cache.json`, so a second run for the same patient is instant. A first run takes about 30 seconds.

## Files

- `app.py`: server, patient parser, ClinicalTrials.gov fetch, Claude calls.
- `static/index.html`: the team EHR mockup with the "Triright · Trial Match" tab.
- `static/screen.html`: earlier version (one trial against many patients), kept as a fallback.
- `synthetic-patient-data.md`: 24 synthetic personas and the reference trial. The answer key in section 3 is not sent to the model.

Screening aid only. Final eligibility is determined by the study team. Synthetic data only.

## Vercel

The `app` folder deploys to Vercel as is (root directory `app`). `api/index.py` reuses the handler in `app.py`, and `vercel.json` routes `/api/*` to it and `/` to `static/index.html`. On Vercel, the trial list and parsed criteria come from `snapshot/`, a frozen copy taken on 2026-10-08, because the function disk is read-only. Set `ANTHROPIC_API_KEY` as a Vercel environment variable, then redeploy.

The voice service in `voice/` deploys too: `api/voice.py` serves it under `/voice`, which the "Explain to patient" panel calls. Add `ELEVENLABS_API_KEY` to the Vercel environment variables, plus `ANTHROPIC_WORKSPACE_ID` if the Anthropic key is an organization key that isn't scoped to a workspace. Check https://<site>/voice/api/config: every value should be `true`.
