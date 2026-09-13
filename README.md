# HelperChore Backend

FastAPI and Neo4j backend for the HelperChore volunteer-matching platform.

## Windows setup

1. Install Python 3.12 or newer and Neo4j 5.x.
2. Create a virtual environment: `py -m venv .venv`.
3. Activate it: `.venv\Scripts\Activate.ps1`.
4. Install dependencies: `py -m pip install -r requirements.txt`.
5. Copy `.env.example` to `.env` and set the Neo4j password. PowerShell loads variables for the current session with `$env:NEO4J_PASSWORD = "..."`.
6. Start the API from the repository root: `py -m uvicorn backend.main:app --reload`.

The first application startup verifies Neo4j connectivity and creates the uniqueness constraints and Resident spatial point index. Set `PROOF_DIRECTORY` to a writable Windows directory when the default `C:/app_data/dispute_proofs` is not appropriate.

## Data conventions

Residents participating as volunteers must have `role: 'VOLUNTEER'`, numeric `latitude` and `longitude`, and a numeric `points` property. Chores use `state` and `complexity_points`; assigned chores connect the volunteer with `(:Resident)-[:ASSIGNED_TO]->(:Task)`. A senior posts a chore with `(:Resident)-[:POSTED]->(:Task)`. Required skills and volunteer skills use the `Skill` node and `HAS_SKILL.certified` relationship property described by the API.

The interactive API contract is available at `/docs` after starting Uvicorn.