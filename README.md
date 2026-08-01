# AutoML Studio

This workspace now contains:

- A FastAPI backend in [backend/main.py](backend/main.py) that uses the existing Python AutoML modules for dataset loading, preprocessing, training, evaluation, and comparison.
- A static HTML/CSS/JS frontend in [frontend](frontend) with Bootstrap and Chart.js styling.

## Run locally

### Backend

```bash
cd /home/samuel/Documents/AiMlAuto
/home/samuel/Documents/AiMlAuto/.venv/bin/python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

The frontend is now just static files. You can open it through the FastAPI app with:

```bash
cd /home/samuel/Documents/AiMlAuto
/home/samuel/Documents/AiMlAuto/.venv/bin/python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://127.0.0.1:8000.
