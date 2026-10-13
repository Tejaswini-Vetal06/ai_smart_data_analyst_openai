# AI Smart Data Analyst — OpenAI Multi-Agent Edition

## What this project does
- Upload CSV datasets or start with generated student data.
- Route questions using a Router Agent powered by an OpenAI model.
- Use a Data Analyst Agent to choose a safe, structured analysis operation.
- Compute numeric results with pandas/Python rather than letting the LLM invent results.
- Use a Response Agent to explain the computed result.
- Train/evaluate regression or classification models with scikit-learn.
- Demonstrate RAG: retrieve relevant notes using TF-IDF and ask an LLM to answer only from retrieved context.
- Browse a glossary of 20 AI terms.

This is a small multi-agent learning prototype. The specialists are separate model calls with explicit instructions; it is not a fully autonomous agent framework. The ML workflow is user-triggered in the ML Lab.

## Setup on Windows / VS Code

1. Install Python 3.10+ and extract this ZIP.
2. Open the `ai_smart_data_analyst_openai` folder in VS Code.
3. Open **Terminal → New Terminal** and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or skip activation and use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

4. Create a file named `.env` in the project folder by copying `.env.example`. Put your own key in it:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

5. Start the app:

```powershell
python -m streamlit run app.py
```

6. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Try it
Upload `sample_students.csv`, then in **Ask AI Agents** ask:
- `What is the average final_marks?`
- `How many rows are in this dataset?`
- `Show the correlation`
- `How many missing values are there?`

The Router Agent should route these to analytics, the Data Analyst Agent proposes a structured operation, Python computes it, and the Response Agent explains it.

## API key and cost
- You need an OpenAI API key with API access and available billing/credits. ChatGPT subscriptions and API billing are separate.
- API usage may incur costs.
- Never share your key in chat or commit `.env` to GitHub. The `.gitignore` excludes `.env`.
- You can paste the key into the sidebar instead of creating `.env`; it is used only for the current app session.

## Troubleshooting
- `OpenAI API key not found`: check `.env` is in the same folder as `app.py`, and restart Streamlit after changing it.
- `401` / authentication error: verify the key is correct and active.
- `429` / quota or rate-limit error: check API billing/usage and limits.
- `model not found`: enter a model available to your API account in the sidebar or change `OPENAI_MODEL`.
- `No module named openai`: activate `.venv`, then run `python -m pip install -r requirements.txt`.
- Wrong column: use the exact column name from the uploaded CSV, or ask `List the columns`.

## Project structure
```text
ai_smart_data_analyst_openai/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── sample_students.csv
```

## Suggested GitHub description
An OpenAI-powered multi-agent data analyst built with Python and Streamlit. It routes natural-language questions to specialized agents, performs verified pandas calculations, trains scikit-learn models, and demonstrates RAG over user-provided notes.
