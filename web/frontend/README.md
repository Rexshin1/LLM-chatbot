# Antigravity LLM V4 - Web Chat Interface

This is the React + Vite frontend and FastAPI backend for the Antigravity LLM V4.

## How to Run

### Backend

To start the FastAPI backend, run the following command from the project root:

```bash
PYTHONPATH=. .venv/bin/uvicorn web.app:app --host 127.0.0.1 --port 8000
```

### Frontend Development

To run the frontend in development mode with hot-reloading:

```bash
cd web/frontend
npm install
npm run dev
```

The Vite dev server will run at `http://localhost:5173/` and proxy API calls to the backend on `http://127.0.0.1:8000/`.

### Frontend Production Build

To build the React frontend for production (outputting files to `web/static` to be served directly by the FastAPI backend):

```bash
cd web/frontend
npm run build
```
