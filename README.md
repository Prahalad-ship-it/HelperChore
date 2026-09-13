# HelperChore

HelperChore connects older adults with nearby volunteers using skill, certification, distance, review quality, and dispute history. Neo4j stores the relationships. NumPy ranks candidates. NVIDIA NIM turns chore photos into clear steps and gives volunteers safety-aware help.

The AI model is [meta/muse-glimmer-30b](https://build.nvidia.com/meta), accessed through NVIDIA's OpenAI-compatible endpoint at `https://nvidia.com`.

## Layout

`app/main.py` boots FastAPI, CORS, Neo4j, and static files. `app/api/endpoints.py` contains the routes. `app/core/config.py` loads settings, `app/core/database.py` manages Neo4j, `app/core/algorithms.py` contains the NumPy engine, `app/models/schemas.py` contains Pydantic v2 models, and `app/static/` contains the five user interfaces.

## Run in Codespaces

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
docker run -d --name helperchore-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/helperchore-dev neo4j:5
export NEO4J_URI="bolt://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="helperchore-dev"
export NVIDIA_API_KEY="your-nvidia-api-key"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/`, API docs at `http://localhost:8000/docs`, and Neo4j Browser at `http://localhost:7474`. Forward port `8000` in the Codespaces Ports panel when needed.

On Windows PowerShell, use `py -m venv .venv`, `.venv\\Scripts\\Activate.ps1`, `py -m pip install -r requirements.txt`, set `$env:NEO4J_URI`, `$env:NEO4J_USER`, `$env:NEO4J_PASSWORD`, and `$env:NVIDIA_API_KEY`, then run `py -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`.

Startup verifies Neo4j and creates unique ID constraints plus the `resident_location` POINT index. Task images go to `./chore_media/`; misconduct evidence goes to `./dispute_proofs/`.

## Routes

The API provides auth, matching, AI drafting, senior refinement, volunteer copilot help, arrival approval, misconduct reporting, completion rewards, coupon redemption, and `GET /api/admin/disputes`.