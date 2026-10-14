# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT) and get an **estimated ATS score**, a category breakdown, prioritized improvements with example rewrites, and missing keywords. Optionally paste a job description to check keyword match for a specific role.

Built with **Streamlit** (UI), **Google Gemini Flash** (AI analysis), and **pypdf / python-docx** (parsing).

## How the score works

| Category | Weight | Source |
|---|---|---|
| Keywords | 30% | Gemini (against your job description if provided) |
| Format & Structure | 30% | 60% rule-based checks + 40% Gemini |
| Content & Impact | 25% | Gemini |
| Readability | 15% | Gemini |

The rule-based checks (no AI) look for: machine-readable text, email, phone, LinkedIn/GitHub link, standard section headings, sensible length, quantified achievements, and action verbs.

> Scores are estimates. Every real ATS parses resumes differently, so treat the result as guidance.

## Run locally

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install and run:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-key"  # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

You can also skip the environment variable and paste the key into the sidebar.

**Or** create `.streamlit/secrets.toml` (never commit this file):

```toml
GEMINI_API_KEY = "your-key"
```

## Choosing the Gemini model

The default is `gemini-3.6-flash`. Google releases new Flash versions often, so you can change the model without editing code:

- type a different model ID in the sidebar, or
- set the `GEMINI_MODEL` environment variable / Streamlit secret.

If you see a "model not found" error, the ID was retired or renamed — check https://ai.google.dev/gemini-api/docs/models for the current list.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Pick your repository, branch `main`, and main file path `app.py`.
5. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
6. Click **Deploy**.

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Limitations

- Scanned / image-only PDFs have no extractable text (and ATS software can't read them either). The app tells you when this happens.
- Files are limited to 5 MB; very long resumes are truncated before analysis.
- Resume text is sent to Google's Gemini API. Don't upload documents you aren't comfortable sharing with that service.
- The free Gemini tier has rate limits; if you hit them, wait a minute and retry.
