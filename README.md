---
title: Email Triage
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Team Members
=======
## Team Members
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
- Riya Patel
- Gayatri Kirange
- Rohan Nalawade

# Email Triage OpenEnv

[![openenv](https://img.shields.io/badge/openenv-compliant-blue)](https://github.com/openenv/openenv)

A real-world [OpenEnv](https://github.com/openenv/openenv) environment for training and evaluating AI agents on business email triage. The agent receives a realistic inbox and must classify each email by priority, assign category labels, flag reply requirements, and (hard task) write executive summaries.

<<<<<<< HEAD
... (rest of your README content goes here)
[![openenv](https://img.shields.io/badge/openenv-compliant-blue)](https://github.com/openenv/openenv)

A real-world [OpenEnv](https://github.com/openenv/openenv) environment for training and evaluating AI agents on business email triage. The agent receives a realistic inbox and must classify each email by priority, assign category labels, flag reply requirements, and (hard task) write executive summaries.

=======
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
---

## Project Structure

```
<<<<<<< HEAD

=======
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
email-triage-env/
├── Dockerfile            # builds & runs on port 7860
├── openenv.yaml          # OpenEnv spec metadata
├── requirements.txt
├── inference.py          # mandatory baseline script
├── README.md
├── env/
<<<<<<< HEAD
│   ├── **init**.py
=======
│   ├── __init__.py
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
│   ├── email_data.py     # 25 synthetic emails with ground truth
│   ├── models.py         # Pydantic Observation / Action / Reward
│   ├── tasks.py          # 3 tasks: easy / medium / hard
│   ├── graders.py        # deterministic scoring
│   └── environment.py    # reset() / step() / state()
└── server/
<<<<<<< HEAD
├── **init**.py
└── main.py           # FastAPI HTTP server

````
=======
    ├── __init__.py
    └── main.py           # FastAPI HTTP server
```
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3

---

## Quick Start

### Docker
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
<<<<<<< HEAD
````

### Local

=======
```

### Local
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
```bash
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

### Run inference baseline
<<<<<<< HEAD

=======
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
```bash
export HF_TOKEN=your_token
export ENV_URL=http://localhost:7860
python inference.py
```

---

## API Endpoints

<<<<<<< HEAD
| Method | Path    | Description                  |
| ------ | ------- | ---------------------------- |
| GET    | /health | Health check — returns 200   |
| GET    | /tasks  | List all tasks with metadata |
| POST   | /reset  | Start new episode            |
| POST   | /step   | Submit one triage action     |
| GET    | /state  | Current env state (debug)    |
| GET    | /docs   | Swagger UI                   |

### POST /reset

=======
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check — returns 200 |
| GET | /tasks | List all tasks with metadata |
| POST | /reset | Start new episode |
| POST | /step | Submit one triage action |
| GET | /state | Current env state (debug) |
| GET | /docs | Swagger UI |

### POST /reset
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
```json
{"task": "basic_triage", "seed": 42}
```

### POST /step
<<<<<<< HEAD

=======
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
```json
{
  "email_id": "e001",
  "priority": "urgent",
  "label": "infrastructure",
  "reply_needed": true,
  "summary": "Production database has been down 12 minutes causing 100% error rate."
}
```

---

## Tasks

### Easy — `basic_triage`
<<<<<<< HEAD

5 emails. Assign correct priority only.

* Score: 1.0 exact, 0.5 adjacent, 0.0 wrong
* Expected score (GPT-4 class): ~0.75

### Medium — `full_triage`

8 emails. Priority (50%) + label (30%) + reply flag (20%).

* Expected score (GPT-4 class): ~0.60

### Hard — `triage_and_summarize`

10 emails. Priority (40%) + label (20%) + reply (20%) + 1-sentence summary (20%).

* Expected score (GPT-4 class): ~0.50
=======
5 emails. Assign correct priority only.
- Score: 1.0 exact, 0.5 adjacent, 0.0 wrong
- Expected score (GPT-4 class): ~0.75

### Medium — `full_triage`
8 emails. Priority (50%) + label (30%) + reply flag (20%).
- Expected score (GPT-4 class): ~0.60

### Hard — `triage_and_summarize`
10 emails. Priority (40%) + label (20%) + reply (20%) + 1-sentence summary (20%).
- Expected score (GPT-4 class): ~0.50
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3

---

## Observation Space

<<<<<<< HEAD
| Field            | Type        | Description                                       |
| ---------------- | ----------- | ------------------------------------------------- |
| emails           | List[Email] | Full inbox (id, subject, sender, body, timestamp) |
| current_step     | int         | Steps completed (0 at reset)                      |
| total_emails     | int         | Total emails this episode                         |
| task_name        | str         | Active task                                       |
| task_description | str         | Natural language goal                             |
| triaged_ids      | List[str]   | Already triaged                                   |
| remaining_ids    | List[str]   | Still to triage                                   |
| inbox_metadata   | dict        | Task config details                               |

## Action Space

| Field        | Required      | Values                  |         |                |       |      |       |
| ------------ | ------------- | ----------------------- | ------- | -------------- | ----- | ---- | ----- |
| email_id     | Always        | Must match inbox        |         |                |       |      |       |
| priority     | Always        | urgent                  | high    | normal         | low   | spam |       |
| label        | Medium + Hard | billing                 | support | infrastructure | sales | hr   | other |
| reply_needed | Medium + Hard | true / false            |         |                |       |      |       |
| summary      | Hard only     | 1 sentence, 10-40 words |         |                |       |      |       |
=======
| Field | Type | Description |
|-------|------|-------------|
| emails | List[Email] | Full inbox (id, subject, sender, body, timestamp) |
| current_step | int | Steps completed (0 at reset) |
| total_emails | int | Total emails this episode |
| task_name | str | Active task |
| task_description | str | Natural language goal |
| triaged_ids | List[str] | Already triaged |
| remaining_ids | List[str] | Still to triage |
| inbox_metadata | dict | Task config details |

## Action Space

| Field | Required | Values |
|-------|----------|--------|
| email_id | Always | Must match inbox |
| priority | Always | urgent | high | normal | low | spam |
| label | Medium + Hard | billing | support | infrastructure | sales | hr | other |
| reply_needed | Medium + Hard | true / false |
| summary | Hard only | 1 sentence, 10-40 words |
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3

---

## Reward Function

Dense reward per step in [0.0, 1.0].

**Priority** — graduated partial credit:
<<<<<<< HEAD

* Adjacent priority (e.g. urgent→high) = 0.5
* Two levels off = 0.0
=======
- Adjacent priority (e.g. urgent→high) = 0.5
- Two levels off = 0.0
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3

**Label** — exact=1.0 | valid-but-wrong=0.25 | missing=0.0

**Reply** — binary: correct=1.0, wrong=0.0

**Summary** — heuristic: length (30%) + keyword overlap (50%) + domain terms (20%)

**Episode score** = mean of all step rewards

---

## Baseline Scores (seed=42, Qwen2.5-72B-Instruct)

<<<<<<< HEAD
| Task                 | Score     |
| -------------------- | --------- |
| basic_triage         | ~0.75     |
| full_triage          | ~0.60     |
| triage_and_summarize | ~0.50     |
| **Mean**             | **~0.62** |
=======
| Task | Score |
|------|-------|
| basic_triage | ~0.75 |
| full_triage | ~0.60 |
| triage_and_summarize | ~0.50 |
| **Mean** | **~0.62** |
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3

---

## Environment Variables

<<<<<<< HEAD
| Var          | Default                                                              | Description  |
| ------------ | -------------------------------------------------------------------- | ------------ |
| API_BASE_URL | [https://router.huggingface.co/v1](https://router.huggingface.co/v1) | LLM endpoint |
| MODEL_NAME   | Qwen/Qwen2.5-72B-Instruct                                            | Model id     |
| HF_TOKEN     | —                                                                    | API key      |
| ENV_URL      | [http://localhost:7860](http://localhost:7860)                       | Env base URL |
=======
| Var | Default | Description |
|-----|---------|-------------|
| API_BASE_URL | https://router.huggingface.co/v1 | LLM endpoint |
| MODEL_NAME | Qwen/Qwen2.5-72B-Instruct | Model id |
| HF_TOKEN | — | API key |
| ENV_URL | http://localhost:7860 | Env base URL |
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3

---

## Deploy to Hugging Face Spaces

1. New Space → select **Docker** SDK
<<<<<<< HEAD
2. Add tag `openenv` in Space settings
3. Push:

=======
2. Add tag `openenv` in Space settings  
3. Push:
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
git add .
git commit -m "Initial submission"
git push origin main
```

---

## Validate Before Submitting

```bash
# 1. Build and run
docker build -t email-triage-env .
docker run -d -p 7860:7860 --name test-env email-triage-env
sleep 10

# 2. Smoke test reset
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"basic_triage","seed":42}' | python3 -m json.tool

# 3. Run validator
./validate-submission.sh https://YOUR_SPACE.hf.space .

# Cleanup
docker stop test-env && docker rm test-env
```
<<<<<<< HEAD

```

---

✅ **What’s fixed / improved:**

- Added **Hugging Face front matter** at the top (`title`, `emoji`, `colorFrom`, `colorTo`, `sdk`, etc.)  
- Added **team members at the very top**  
- Preserved **all original content**  
- The SDK is now set to `docker` (matches your Space setup)  

---

If you want, I can also **convert it for Streamlit SDK** instead of Docker, so your Space will **render directly on Hugging Face without Docker**—makes it even easier to launch.  

Do you want me to do that?
```
=======
>>>>>>> f5a84bd29ef220ccce6bb7e70c78f0be3c4aa4d3
