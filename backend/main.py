from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from neo4j import Driver, GraphDatabase, ManagedTransaction
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
MODEL = "meta/muse-glimmer-30b"
client = OpenAI(base_url="https://nvidia.com", api_key=NVIDIA_API_KEY or "not-configured")
PROOF_DIRECTORY = Path("./dispute_proofs")
MEDIA_DIRECTORY = Path("./chore_media")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

STATE_TRANSITIONS = {
    "POSTED": "VOLUNTEER_ASSIGNED",
    "VOLUNTEER_ASSIGNED": "ARRIVED_PENDING_APPROVAL",
    "ARRIVED_PENDING_APPROVAL": "IN_PROGRESS",
    "IN_PROGRESS": "SUBMITTED_FOR_REVIEW",
    "SUBMITTED_FOR_REVIEW": "COMPLETED",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserAction(StrictModel):
    user_id: str = Field(min_length=1, max_length=100)
    chore_id: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=80)


class ChoreRefinement(StrictModel):
    senior_id: str = Field(min_length=1, max_length=100)
    refinement_text: str = Field(min_length=1, max_length=5000)

    @field_validator("refinement_text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("refinement_text cannot be blank")
        return value.strip()


class ArrivalCheck(StrictModel):
    senior_id: str = Field(min_length=1, max_length=100)
    approved: bool


class CouponRedemption(StrictModel):
    user_id: str = Field(min_length=1, max_length=100)
    reward_name: str = Field(min_length=1, max_length=120)
    points_cost: int = Field(gt=0, le=1_000_000)


class MatchResult(StrictModel):
    volunteer_id: str
    name: str | None = None
    score: float
    skill_score: float
    distance_score: float
    rating_score: float
    trust_score: float
    distance_miles: float
    mean_rating: float
    dispute_count: int


class CopilotQuestion(StrictModel):
    volunteer_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=3000)


class NIMDraft(BaseModel):
    tools: list[str] = Field(default_factory=list)
    safety_risks: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    summary: str = ""


class NIMAnswer(BaseModel):
    answer: str
    danger_flag: bool = False
    warning: str | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_driver(request: Request) -> Driver:
    driver = getattr(request.app.state, "driver", None)
    if driver is None:
        raise HTTPException(status_code=503, detail="Neo4j is not initialized")
    return driver


def run_read(driver: Driver, callback: Any, **params: Any) -> Any:
    try:
        with driver.session() as session:
            return session.execute_read(callback, **params)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Database read failed") from exc


def run_write(driver: Driver, callback: Any, **params: Any) -> Any:
    try:
        with driver.session() as session:
            return session.execute_write(callback, **params)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Database transaction failed") from exc


def initialize_schema(driver: Driver) -> None:
    statements = [
        "CREATE CONSTRAINT resident_id_unique IF NOT EXISTS FOR (n:Resident) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (n:Task) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (n:Skill) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT report_id_unique IF NOT EXISTS FOR (n:Report) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT coupon_id_unique IF NOT EXISTS FOR (n:Coupon) REQUIRE n.id IS UNIQUE",
        "CREATE POINT INDEX resident_location IF NOT EXISTS FOR (n:Resident) ON (n.location)",
    ]
    with driver.session() as session:
        for statement in statements:
            session.run(statement).consume()


@asynccontextmanager
async def lifespan(app: FastAPI):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), max_connection_pool_size=50)
    try:
        driver.verify_connectivity()
        initialize_schema(driver)
        app.state.driver = driver
        yield
    finally:
        driver.close()


app = FastAPI(title="HelperChore", version="2.0.0", lifespan=lifespan)


def haversine_miles(target_latitude: float, target_longitude: float, coordinates: np.ndarray) -> np.ndarray:
    earth_radius_miles = 3958.7613
    target = np.radians(np.asarray([target_latitude, target_longitude], dtype=np.float64))
    candidates = np.radians(np.asarray(coordinates, dtype=np.float64))
    delta = candidates - target
    a = np.sin(delta[:, 0] / 2) ** 2 + np.cos(target[0]) * np.cos(candidates[:, 0]) * np.sin(delta[:, 1] / 2) ** 2
    return 2 * earth_radius_miles * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def vectorized_match(rows: list[dict[str, Any]], latitude: float, longitude: float, top_k: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    matrix = np.asarray(
        [[row["latitude"], row["longitude"], row["certified"], row["mean_rating"] or 4.0, row["dispute_count"] or 0] for row in rows],
        dtype=np.float64,
    )
    distances = haversine_miles(latitude, longitude, matrix[:, :2])
    skill = 20.0 + 15.0 * matrix[:, 2]
    distance = np.select([distances <= 1, distances <= 5, distances <= 10], [25.0, 18.0, 10.0], default=2.0)
    rating = np.clip(matrix[:, 3], 0, 5) / 5.0 * 25.0
    trust = np.maximum(0.0, 15.0 - matrix[:, 4] * 5.0)
    scores = skill * 0.35 + distance * 0.25 + rating * 0.25 + trust * 0.15
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [
        MatchResult(
            volunteer_id=str(rows[int(i)]["volunteer_id"]),
            name=rows[int(i)].get("name"),
            score=round(float(scores[i]), 3),
            skill_score=round(float(skill[i]), 3),
            distance_score=round(float(distance[i]), 3),
            rating_score=round(float(rating[i]), 3),
            trust_score=round(float(trust[i]), 3),
            distance_miles=round(float(distances[i]), 3),
            mean_rating=round(float(matrix[i, 3]), 3),
            dispute_count=int(matrix[i, 4]),
        ).model_dump()
        for i in order
    ]


def read_match_data(tx: ManagedTransaction, chore_id: str) -> dict[str, Any] | None:
    query = """
    MATCH (senior:Resident)-[:POSTED]->(chore:Task {id: $chore_id})
    OPTIONAL MATCH (chore)-[:REQUIRES]->(required:Skill)
    WITH senior, chore, collect(DISTINCT required.name) AS required_skills
    MATCH (volunteer:Resident {role: 'VOLUNTEER'})-[hs:HAS_SKILL]->(skill:Skill)
    WHERE skill.name IN required_skills AND volunteer.latitude IS NOT NULL AND volunteer.longitude IS NOT NULL
    WITH senior, chore, volunteer, max(CASE WHEN hs.certified = true THEN 1 ELSE 0 END) AS certified
    CALL {
        WITH volunteer
        OPTIONAL MATCH (volunteer)-[review:REVIEWS]->(:Task)
        RETURN coalesce(avg(review.rating), 4.0) AS mean_rating
    }
    CALL {
        WITH volunteer
        OPTIONAL MATCH (volunteer)<-[:ACCUSES]-(report:Report)
        WHERE coalesce(report.active, true) = true
        RETURN count(report) AS dispute_count
    }
    RETURN senior.latitude AS latitude, senior.longitude AS longitude,
           collect({volunteer_id: volunteer.id, name: volunteer.name,
           latitude: volunteer.latitude, longitude: volunteer.longitude,
           certified: certified, mean_rating: mean_rating, dispute_count: dispute_count}) AS candidates
    """
    record = tx.run(query, chore_id=chore_id).single()
    return None if record is None else record.data()


@app.post("/chores/{chore_id}/match", response_model=list[MatchResult])
def match_chore(
    chore_id: str,
    top_k: Annotated[int, Query(ge=1, le=100)] = 20,
    driver: Driver = Depends(get_driver),
) -> list[MatchResult]:
    data = run_read(driver, read_match_data, chore_id=chore_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Chore not found")
    if data["latitude"] is None or data["longitude"] is None:
        raise HTTPException(status_code=422, detail="Senior coordinates are required")
    return [MatchResult(**item) for item in vectorized_match(data["candidates"], data["latitude"], data["longitude"], top_k)]


def nim_text(messages: list[dict[str, Any]], response_format: dict[str, str] | None = None) -> str:
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY is not configured")
    try:
        kwargs: dict[str, Any] = {"model": MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 1200}
        if response_format:
            kwargs["response_format"] = response_format
        result = client.chat.completions.create(**kwargs)
        content = result.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="NIM returned an empty response")
        return content
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="NVIDIA NIM request failed") from exc


async def nim_text_async(messages: list[dict[str, Any]], response_format: dict[str, str] | None = None) -> str:
    return await asyncio.to_thread(nim_text, messages, response_format)


async def save_upload(upload: UploadFile, directory: Path, identifier: str) -> tuple[Path, int]:
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "upload.bin").suffix.lower()
    suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bin"} else ".bin"
    path = directory / f"{identifier}{suffix}"
    size = 0
    try:
        with path.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds the maximum size")
                output.write(chunk)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not save upload") from exc
    finally:
        await upload.close()
    return path, size


def parse_json_response(content: str, schema: type[BaseModel]) -> BaseModel:
    cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return schema.model_validate_json(cleaned)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="NIM returned invalid structured data") from exc


def save_ai_draft(tx: ManagedTransaction, chore_id: str, photo_path: str, draft: NIMDraft) -> dict[str, Any]:
    record = tx.run(
        """
        MATCH (chore:Task {id: $chore_id})
        SET chore.original_photo_path = $photo_path,
            chore.ai_steps = $ai_steps, chore.ai_tools = $tools,
            chore.ai_safety_risks = $risks, chore.ai_summary = $summary,
            chore.ai_generated_at = $created_at
        RETURN chore.id AS chore_id, chore.ai_steps AS ai_steps
        """,
        chore_id=chore_id, photo_path=photo_path, ai_steps=json.dumps(draft.steps),
        tools=draft.tools, risks=draft.safety_risks, summary=draft.summary, created_at=now(),
    ).single()
    if record is None:
        raise HTTPException(status_code=404, detail="Chore not found")
    return record.data()


@app.post("/chores/draft-ai-steps")
async def draft_ai_steps(
    chore_id: Annotated[str, Form(min_length=1, max_length=100)],
    image: Annotated[UploadFile, File()],
    driver: Annotated[Driver, Depends(get_driver)],
) -> dict[str, Any]:
    path, _ = await save_upload(image, MEDIA_DIRECTORY, str(uuid.uuid4()))
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Analyze this household chore image. Return JSON with tools (array), safety_risks (array), steps (array), and summary (string). Make the steps practical for a volunteer assisting a senior."},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
        ]}]
        draft = parse_json_response(await nim_text_async(messages, {"type": "json_object"}), NIMDraft)
        return run_write(driver, save_ai_draft, chore_id=chore_id, photo_path=str(path), draft=draft)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def append_refinement(tx: ManagedTransaction, chore_id: str, senior_id: str, refinement: str) -> dict[str, Any]:
    record = tx.run(
        """
        MATCH (senior:Resident {id: $senior_id})-[:POSTED]->(chore:Task {id: $chore_id})
        SET chore.senior_refinements = coalesce(chore.senior_refinements, []) + $refinement,
            chore.last_refined_at = $created_at
        RETURN chore.id AS chore_id, chore.senior_refinements AS refinements
        """,
        senior_id=senior_id, chore_id=chore_id, refinement=refinement, created_at=now(),
    ).single()
    if record is None:
        raise HTTPException(status_code=404, detail="Senior or chore relationship not found")
    return record.data()


@app.post("/chores/{chore_id}/senior-refinement")
def senior_refinement(
    chore_id: str, refinement: ChoreRefinement, driver: Annotated[Driver, Depends(get_driver)]
) -> dict[str, Any]:
    return run_write(driver, append_refinement, chore_id=chore_id, senior_id=refinement.senior_id, refinement=refinement.refinement_text)


def read_copilot_context(tx: ManagedTransaction, chore_id: str, volunteer_id: str) -> dict[str, Any] | None:
    record = tx.run(
        """
        MATCH (volunteer:Resident {id: $volunteer_id})-[:ASSIGNED_TO]->(chore:Task {id: $chore_id})
        RETURN chore.original_photo_path AS photo_path, chore.ai_steps AS ai_steps,
               chore.senior_refinements AS refinements, chore.ai_safety_risks AS risks
        """, chore_id=chore_id, volunteer_id=volunteer_id,
    ).single()
    return None if record is None else record.data()


@app.post("/chores/{chore_id}/volunteer-copilot-help", response_model=NIMAnswer)
async def volunteer_copilot_help(
    chore_id: str, question: CopilotQuestion, driver: Annotated[Driver, Depends(get_driver)]
) -> NIMAnswer:
    context = run_read(driver, read_copilot_context, chore_id=chore_id, volunteer_id=question.volunteer_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Assigned chore or volunteer not found")
    photo_content: dict[str, Any] = {"type": "text", "text": "Original photo is unavailable."}
    if context["photo_path"] and Path(context["photo_path"]).exists():
        image_path = Path(context["photo_path"])
        media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        photo_content = {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}
    context_text = json.dumps({"question": question.question, "ai_steps": context["ai_steps"], "senior_warnings": context["refinements"], "ai_safety_risks": context["risks"]})
    messages = [{"role": "system", "content": "You are a safety-first volunteer copilot. Treat senior warnings as binding. If the question conflicts with any warning or creates risk, set danger_flag true and clearly flag the danger."}, {"role": "user", "content": [{"type": "text", "text": context_text}, photo_content]}]
    answer = parse_json_response(await nim_text_async(messages, {"type": "json_object"}), NIMAnswer)
    return NIMAnswer.model_validate(answer)


def verify_arrival_tx(tx: ManagedTransaction, chore_id: str, senior_id: str, approved: bool) -> dict[str, Any]:
    if not approved:
        record = tx.run("""MATCH (senior:Resident {id: $senior_id})-[:POSTED]->(chore:Task {id: $chore_id}) WHERE chore.state = 'ARRIVED_PENDING_APPROVAL' SET chore.arrival_rejected_at = $at RETURN chore.id AS chore_id, chore.state AS state""", senior_id=senior_id, chore_id=chore_id, at=now()).single()
        if record is None:
            raise HTTPException(status_code=409, detail="Chore is not awaiting approval")
        raise HTTPException(status_code=403, detail="Arrival rejected; progression blocked")
    record = tx.run("""MATCH (senior:Resident {id: $senior_id})-[:POSTED]->(chore:Task {id: $chore_id}) WHERE chore.state = 'ARRIVED_PENDING_APPROVAL' SET chore.state = 'IN_PROGRESS', chore.arrival_approved_at = $at RETURN chore.id AS chore_id, chore.state AS state""", senior_id=senior_id, chore_id=chore_id, at=now()).single()
    if record is None:
        raise HTTPException(status_code=409, detail="Chore is not awaiting approval")
    return record.data()


@app.post("/chores/{chore_id}/verify-arrival")
def verify_arrival(chore_id: str, check: ArrivalCheck, driver: Annotated[Driver, Depends(get_driver)]) -> dict[str, Any]:
    return run_write(driver, verify_arrival_tx, chore_id=chore_id, senior_id=check.senior_id, approved=check.approved)


def misconduct_tx(tx: ManagedTransaction, **values: Any) -> dict[str, Any]:
    record = tx.run("""
        MATCH (reporter:Resident {id: $reporter_id}), (accused:Resident {id: $accused_id}), (chore:Task {id: $chore_id})
        CREATE (report:Report:Incident {id: $report_id, reason: $reason, description: $description, proof_path: $proof_path, proof_size: $proof_size, content_type: $content_type, immutable: true, active: true, created_at: $created_at})
        CREATE (reporter)-[:FILED_REPORT]->(report)
        CREATE (report)-[:ACCUSES]->(accused)
        CREATE (report)-[:LINKED_TO_CHORE]->(chore)
        SET chore.state = 'DISPUTED', accused.dispute_count = coalesce(accused.dispute_count, 0) + 1
        RETURN report.id AS report_id, chore.state AS chore_state, accused.dispute_count AS dispute_count
    """, **values).single()
    if record is None:
        raise HTTPException(status_code=404, detail="Reporter, accused resident, or chore not found")
    return record.data()


@app.post("/report-misconduct", status_code=status.HTTP_201_CREATED)
async def report_misconduct(
    reporter_id: Annotated[str, Form(min_length=1, max_length=100)], accused_id: Annotated[str, Form(min_length=1, max_length=100)], chore_id: Annotated[str, Form(min_length=1, max_length=100)], reason: Annotated[str, Form(min_length=1, max_length=200)], description: Annotated[str, Form(min_length=1, max_length=5000)], proof_photo: Annotated[UploadFile, File()], driver: Annotated[Driver, Depends(get_driver)],
) -> dict[str, Any]:
    report_id = str(uuid.uuid4())
    path, size = await save_upload(proof_photo, PROOF_DIRECTORY, report_id)
    try:
        return run_write(driver, misconduct_tx, reporter_id=reporter_id, accused_id=accused_id, chore_id=chore_id, reason=reason, description=description, report_id=report_id, proof_path=str(path), proof_size=size, content_type=proof_photo.content_type, created_at=now())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def approve_complete_tx(tx: ManagedTransaction, chore_id: str, senior_id: str) -> dict[str, Any]:
    record = tx.run("""
        MATCH (senior:Resident {id: $senior_id})-[:POSTED]->(chore:Task {id: $chore_id})
        MATCH (volunteer:Resident)-[:ASSIGNED_TO]->(chore)
        WHERE chore.state = 'SUBMITTED_FOR_REVIEW'
        WITH chore, volunteer, toInteger(coalesce(chore.complexity_points, 10)) AS earned
        SET chore.state = 'COMPLETED', chore.completed_at = $at, volunteer.points = coalesce(volunteer.points, 0) + earned
        RETURN chore.id AS chore_id, volunteer.id AS volunteer_id, earned AS points_earned, volunteer.points AS balance
    """, chore_id=chore_id, senior_id=senior_id, at=now()).single()
    if record is None:
        raise HTTPException(status_code=409, detail="Chore is not submitted for review")
    return record.data()


@app.post("/chores/{chore_id}/approve-complete")
def approve_complete(chore_id: str, action: UserAction, driver: Annotated[Driver, Depends(get_driver)]) -> dict[str, Any]:
    if action.chore_id != chore_id or action.action != "APPROVE_COMPLETE":
        raise HTTPException(status_code=422, detail="Action does not match completion approval")
    return run_write(driver, approve_complete_tx, chore_id=chore_id, senior_id=action.user_id)


def redeem_tx(tx: ManagedTransaction, redemption: CouponRedemption) -> dict[str, Any]:
    coupon_id = str(uuid.uuid4())
    record = tx.run("""
        MATCH (resident:Resident {id: $user_id})
        WHERE coalesce(resident.points, 0) >= $cost
        SET resident.points = resident.points - $cost
        CREATE (coupon:Coupon {id: $coupon_id, name: $name, points_cost: $cost, status: 'ISSUED', issued_at: $at})
        CREATE (resident)-[:REDEEMED {at: $at}]->(coupon)
        RETURN coupon.id AS coupon_id, coupon.name AS reward_name, resident.points AS remaining_points, coupon.status AS status
    """, user_id=redemption.user_id, cost=redemption.points_cost, name=redemption.reward_name, coupon_id=coupon_id, at=now()).single()
    if record is None:
        raise HTTPException(status_code=400, detail="User not found or insufficient points")
    return record.data()


@app.post("/redeem-rewards", status_code=status.HTTP_201_CREATED)
def redeem_rewards(redemption: CouponRedemption, driver: Annotated[Driver, Depends(get_driver)]) -> dict[str, Any]:
    return run_write(driver, redeem_tx, redemption=redemption)


LANDING_PAGE = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>HelperChore | Neo4j trust, made practical</title>
<link rel='preconnect' href='https://fonts.googleapis.com'><link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>
<link href='https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap' rel='stylesheet'>
<style>
:root{--slate:#1E293B;--paper:#F8FAFC;--orange:#FF5A1F;--muted:#9BA8B8;--line:#3B4A5D}*{box-sizing:border-box}body{margin:0;background:var(--slate);color:var(--paper);font-family:'Plus Jakarta Sans',sans-serif}a{color:inherit;text-decoration:none}.shell{max-width:1240px;margin:auto;padding:28px 5vw 90px}.nav{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:22px}.brand{font-weight:800;letter-spacing:.08em;text-transform:uppercase}.brand b{color:var(--orange)}.nav span{font-size:12px;color:var(--muted)}.hero{display:grid;grid-template-columns:1.02fr .98fr;gap:7vw;align-items:end;padding:92px 0 110px}.eyebrow{color:var(--orange);font-size:12px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}.hero h1{font:700 clamp(3.4rem,7vw,7.2rem)/.92 'Playfair Display',serif;letter-spacing:-.04em;margin:20px 0 28px}.hero p{color:#D2D9E2;max-width:510px;font-size:18px;line-height:1.7}.cta{display:inline-block;background:var(--orange);color:#1E293B;font-weight:800;padding:16px 22px;margin-top:18px}.photo{position:relative;border:1px solid #758294;padding:12px;transform:rotate(2deg)}.photo img{display:block;width:100%;height:480px;object-fit:cover;filter:saturate(.8)}.photo figcaption{background:var(--paper);color:var(--slate);font-size:11px;font-weight:800;padding:13px;position:absolute;bottom:28px;left:-24px;max-width:220px}.section{border-top:1px solid var(--line);padding:70px 0}.section-head{display:flex;justify-content:space-between;gap:30px;align-items:start}.section h2{font:700 clamp(2.2rem,4vw,4rem)/1 'Playfair Display',serif;margin:0;max-width:540px}.section-head p{color:var(--muted);max-width:330px;line-height:1.7}.graph{margin-top:58px;display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line)}.node{min-height:170px;padding:28px;border-right:1px solid var(--line);position:relative}.node:last-child{border:0}.node strong{display:block;color:var(--orange);font-size:17px;margin-bottom:22px}.node p{color:#CBD5E1;line-height:1.6;font-size:13px}.node:after{content:'→';color:var(--orange);font-size:26px;position:absolute;right:-13px;top:68px;background:var(--slate);padding:0 5px}.node:last-child:after{display:none}.workflow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;margin-top:64px}.step{padding:0 34px 0 0;margin-right:34px;border-right:1px solid var(--line)}.step:last-child{border:0}.number{font-size:12px;color:var(--orange);font-weight:800}.step h3{font:600 27px 'Playfair Display',serif;margin:24px 0 12px}.step p{color:#CBD5E1;line-height:1.65;font-size:14px}.footer{border-top:3px solid var(--orange);padding-top:22px;display:flex;justify-content:space-between;color:var(--muted);font-size:12px}@keyframes reveal{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:none}}.reveal{animation:reveal .8s ease both}.delay1{animation-delay:.12s}.delay2{animation-delay:.24s}@media(max-width:760px){.hero{grid-template-columns:1fr;padding:62px 0 80px}.photo img{height:350px}.graph,.workflow{grid-template-columns:1fr}.node{border-right:0;border-bottom:1px solid var(--line)}.node:after{display:none}.step{border-right:0;border-bottom:1px solid var(--line);padding:0 0 30px;margin:0 0 30px}.section-head{display:block}.section-head p{margin-top:25px}.footer{display:block;line-height:2.1}}
</style></head>
<body><main class='shell'>
<nav class='nav reveal'><a class='brand' href='/'>helper<b>/</b>chore</a><span>GRAPH TRUST SYSTEM · v2.0</span></nav>
<section class='hero'><div class='reveal delay1'><div class='eyebrow'>A hand-off with a memory</div><h1>Good help,<br>carefully held.</h1><p>HelperChore turns the physical work of looking after a neighbor into a clear, accountable exchange: matched by skill, checked in by consent, and carried forward with context.</p><a class='cta' href='/docs'>Open the workspace ↗</a></div><figure class='photo reveal delay2'><img src='https://images.unsplash.com/photo-1559234938-b60fff04894d?auto=format&fit=crop&w=1200&q=85' alt='A caregiver helping an older adult walk outdoors'><figcaption>Trust is a physical thing. We design for the moment it changes hands.</figcaption></figure></section>
<section class='section'><div class='section-head'><h2>The graph keeps the whole story in view.</h2><p>Every match is grounded in relationships that can be inspected: who asked, what is needed, who is qualified, and what happened next.</p></div><div class='graph'><div class='node'><strong>RESIDENT</strong><p>Senior location, volunteer skills, certifications, points, and trust history.</p></div><div class='node'><strong>TASK</strong><p>Required skills, complexity, AI draft, human warnings, and state.</p></div><div class='node'><strong>REPORT</strong><p>Immutable evidence linking an incident to the people and chore involved.</p></div></div></section>
<section class='section'><div class='eyebrow'>Augmented hand-off workflow</div><div class='workflow'><article class='step'><div class='number'>01 / DRAFTING PHASE</div><h3>See the work first.</h3><p>NVIDIA NIM reads the chore photo and proposes tools, safety risks, and practical steps. It gives the task a useful first shape.</p></article><article class='step'><div class='number'>02 / HUMAN REFINEMENT</div><h3>The senior stays in charge.</h3><p>Custom requirements and non-negotiable warnings become part of the task record, right beside the machine's draft.</p></article><article class='step'><div class='number'>03 / COPILOT EXECUTION</div><h3>Help that remembers.</h3><p>Mid-job questions are answered against the photo, steps, and warnings. A risky suggestion is flagged before it becomes an action.</p></article></div></section>
<footer class='footer'><span>HELPERCHORE / BUILT FOR NEIGHBORS</span><span>Skill · Distance · Quality · Trust</span></footer></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def landing_page() -> HTMLResponse:
    return HTMLResponse(LANDING_PAGE)


@app.get("/health")
def health(driver: Annotated[Driver, Depends(get_driver)]) -> dict[str, str]:
    try:
        with driver.session() as session:
            session.run("RETURN 1").consume()
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Neo4j is unavailable") from exc
