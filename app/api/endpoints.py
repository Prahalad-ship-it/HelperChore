from __future__ import annotations
import asyncio, base64, hashlib, hmac, json, mimetypes, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from neo4j import Driver, ManagedTransaction
from openai import OpenAI
from app.core.algorithms import fetch_match_data, rank_candidates
from app.core.config import settings
from app.core.database import driver_from_request, read, write, require_record
from app.models.schemas import *

router = APIRouter(prefix="/api")
nim = OpenAI(base_url=settings.nim_base_url, api_key=settings.nvidia_api_key or "not-configured")


def timestamp() -> str: return datetime.now(timezone.utc).isoformat()
def password_hash(password: str) -> str: return hashlib.scrypt(password.encode(), salt=b"helperchore-password-salt-v1", n=2**14, r=8, p=1).hex()
def valid_password(password: str, stored: str) -> bool: return hmac.compare_digest(password_hash(password), stored)


@router.post("/chores/{chore_id}/match", response_model=list[MatchResult])
def match_chore(chore_id: str, driver: Annotated[Driver, Depends(driver_from_request)], top_k: Annotated[int, Query(ge=1, le=100)] = 20) -> list[MatchResult]:
    data = read(driver, fetch_match_data, chore_id=chore_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Chore not found")
    if data["senior_latitude"] is None or data["senior_longitude"] is None:
        raise HTTPException(status_code=422, detail="Senior coordinates are required")
    return [MatchResult(**item) for item in rank_candidates(data["candidates"], data["senior_latitude"], data["senior_longitude"], top_k)]


def register_tx(tx: ManagedTransaction, item: RegisterRequest) -> dict[str, Any]:
    user_id = str(uuid.uuid4())
    recovery_code = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    record = tx.run("""
    CREATE (resident:Resident {id:$id,name:$name,password_hash:$password_hash,recovery_code_hash:$recovery_hash,role:$role,latitude:$latitude,longitude:$longitude,points:0,dispute_count:0,created_at:$at})
    FOREACH (skill_name IN $skills | MERGE (skill:Skill {name:skill_name}) MERGE (resident)-[:HAS_SKILL {certified: skill_name IN $certifications}]->(skill))
    RETURN resident.id AS user_id, resident.name AS name, resident.role AS role
    """, id=user_id, name=item.name.strip(), password_hash=password_hash(item.password), recovery_hash=password_hash(recovery_code), role=item.role, latitude=item.latitude, longitude=item.longitude, skills=item.skills, certifications=item.certifications, at=timestamp()).single()
    result = require_record(record, "Could not register user")
    result["recovery_code"] = recovery_code
    return result

@router.post("/auth/register", status_code=201)
def register(item: RegisterRequest, driver: Annotated[Driver, Depends(driver_from_request)]): return write(driver, register_tx, item=item)


def login_tx(tx: ManagedTransaction, item: LoginRequest) -> dict[str, Any]:
    record = tx.run("MATCH (r:Resident {name:$name}) RETURN r", name=item.name.strip()).single()
    if record is None or not valid_password(item.password, record["r"]["password_hash"]): raise HTTPException(status_code=401, detail="Invalid name or password")
    node = record["r"]
    return {"user_id": node["id"], "name": node["name"], "role": node["role"], "points": node.get("points", 0)}

@router.post("/auth/login")
def login(item: LoginRequest, driver: Annotated[Driver, Depends(driver_from_request)]): return read(driver, login_tx, item=item)

def reset_password_tx(tx: ManagedTransaction, item: PasswordResetRequest) -> dict[str, str]:
    record = tx.run("""
    MATCH (r:Resident {name:$name})
    WHERE r.recovery_code_hash IS NOT NULL
    RETURN r
    """, name=item.name.strip()).single()
    if record is None or not valid_password(item.recovery_code, record["r"]["recovery_code_hash"]):
        raise HTTPException(status_code=401, detail="Invalid name or recovery code")
    updated = tx.run("MATCH (r:Resident {name:$name}) SET r.password_hash=$password_hash RETURN r.id AS user_id", name=item.name.strip(), password_hash=password_hash(item.new_password)).single()
    return require_record(updated, "User not found")

@router.post("/auth/reset-password")
def reset_password(item: PasswordResetRequest, driver: Annotated[Driver, Depends(driver_from_request)]): return write(driver, reset_password_tx, item=item)


def nim_call(messages: list[dict[str, Any]], json_mode: bool = False) -> str:
    if not settings.nvidia_api_key: raise HTTPException(status_code=503, detail="NVIDIA_API_KEY is not configured")
    try:
        args: dict[str, Any] = {"model": settings.nim_model, "messages": messages, "temperature": .2, "max_tokens": 1200}
        if json_mode: args["response_format"] = {"type": "json_object"}
        result = nim.chat.completions.create(**args)
        return result.choices[0].message.content or ""
    except Exception as exc: raise HTTPException(status_code=502, detail="NVIDIA NIM request failed") from exc

async def save_upload(upload: UploadFile, directory: Path, identifier: str) -> tuple[Path, int]:
    directory.mkdir(parents=True, exist_ok=True); suffix = Path(upload.filename or ".bin").suffix.lower(); suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else ".bin"; path = directory / f"{identifier}{suffix}"; total = 0
    try:
        with path.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes: raise HTTPException(status_code=413, detail="File is too large")
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True); raise
    finally: await upload.close()
    return path, total

def json_model(content: str, model: type[BaseModel]) -> BaseModel:
    try: return model.model_validate_json(content.strip().removeprefix("```json").removesuffix("```").strip())
    except Exception as exc: raise HTTPException(status_code=502, detail="NIM returned invalid JSON") from exc

class Draft(BaseModel):
    tools: list[str] = []; safety_risks: list[str] = []; steps: list[str] = []; summary: str = ""
class CopilotAnswer(BaseModel):
    answer: str; danger_flag: bool = False; warning: str | None = None


def save_draft_tx(tx: ManagedTransaction, chore_id: str, photo_path: str, draft: Draft) -> dict[str, Any]:
    record = tx.run("MATCH (c:Task {id:$id}) SET c.original_photo_path=$photo,c.ai_steps=$steps,c.ai_tools=$tools,c.ai_safety_risks=$risks,c.ai_summary=$summary,c.ai_generated_at=$at RETURN c.id AS chore_id,c.ai_steps AS ai_steps", id=chore_id, photo_path=photo_path, steps=draft.steps, tools=draft.tools, risks=draft.safety_risks, summary=draft.summary, at=timestamp()).single()
    return require_record(record, "Chore not found")

@router.post("/chores/draft-ai-steps")
async def draft_ai_steps(chore_id: Annotated[str, Form(min_length=1)], image: Annotated[UploadFile, File()], driver: Annotated[Driver, Depends(driver_from_request)]):
    path, _ = await save_upload(image, settings.media_directory, str(uuid.uuid4()))
    encoded = base64.b64encode(path.read_bytes()).decode(); media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    messages = [{"role":"user","content":[{"type":"text","text":"Study this chore image. Return JSON with tools, safety_risks, steps, and summary. Use short, simple steps for a volunteer helping an older adult."},{"type":"image_url","image_url":{"url":f"data:{media};base64,{encoded}"}}]}]
    try: return write(driver, save_draft_tx, chore_id=chore_id, photo_path=str(path), draft=json_model(await asyncio.to_thread(nim_call, messages, True), Draft))
    except Exception: path.unlink(missing_ok=True); raise


def refinement_tx(tx: ManagedTransaction, chore_id: str, item: ChoreRefinement) -> dict[str, Any]:
    record = tx.run("MATCH (s:Resident {id:$senior})-[:POSTED]->(c:Task {id:$chore}) SET c.senior_refinements=coalesce(c.senior_refinements,[])+$text RETURN c.id AS chore_id,c.senior_refinements AS refinements", senior=item.senior_id, chore=chore_id, text=item.refinement_text).single()
    return require_record(record, "Senior or chore not found")

@router.post("/chores/{chore_id}/senior-refinement")
def refinement(chore_id: str, item: ChoreRefinement, driver: Annotated[Driver, Depends(driver_from_request)]): return write(driver, refinement_tx, chore_id=chore_id, item=item)


def context_tx(tx: ManagedTransaction, chore_id: str, volunteer_id: str) -> dict[str, Any] | None:
    record = tx.run("MATCH (v:Resident {id:$v})-[:ASSIGNED_TO]->(c:Task {id:$c}) RETURN c.original_photo_path AS photo,c.ai_steps AS steps,c.senior_refinements AS warnings,c.ai_safety_risks AS risks", v=volunteer_id, c=chore_id).single()
    return None if record is None else record.data()

@router.post("/chores/{chore_id}/volunteer-copilot-help", response_model=CopilotAnswer)
async def copilot(chore_id: str, item: CopilotQuestion, driver: Annotated[Driver, Depends(driver_from_request)]):
    context = read(driver, context_tx, chore_id=chore_id, volunteer_id=item.volunteer_id)
    if context is None: raise HTTPException(status_code=404, detail="Assigned chore not found")
    content: list[dict[str, Any]] = [{"type":"text","text":json.dumps({"question":item.question,"steps":context["steps"],"senior_warnings":context["warnings"],"safety_risks":context["risks"]})}]
    if context["photo"] and Path(context["photo"]).exists():
        p = Path(context["photo"]); content.append({"type":"image_url","image_url":{"url":f"data:{mimetypes.guess_type(p.name)[0] or 'image/jpeg'};base64,{base64.b64encode(p.read_bytes()).decode()}"}})
    messages = [{"role":"system","content":"You are a safety-first copilot. Senior warnings are binding. If a proposed action breaks a warning, refuse it, set danger_flag true, and explain the danger."},{"role":"user","content":content}]
    return json_model(await asyncio.to_thread(nim_call, messages, True), CopilotAnswer)


def arrival_tx(tx: ManagedTransaction, chore_id: str, item: ArrivalCheck) -> dict[str, Any]:
    if not item.approved:
        record = tx.run("MATCH (s:Resident {id:$s})-[:POSTED]->(c:Task {id:$c}) WHERE c.state='ARRIVED_PENDING_APPROVAL' SET c.arrival_rejected_at=$at RETURN c.id AS chore_id,c.state AS state", s=item.senior_id, c=chore_id, at=timestamp()).single()
        if record is None: raise HTTPException(status_code=409, detail="Chore is not awaiting approval")
        raise HTTPException(status_code=403, detail="Arrival rejected; progression blocked")
    record = tx.run("MATCH (s:Resident {id:$s})-[:POSTED]->(c:Task {id:$c}) WHERE c.state='ARRIVED_PENDING_APPROVAL' SET c.state='IN_PROGRESS',c.arrival_approved_at=$at RETURN c.id AS chore_id,c.state AS state", s=item.senior_id, c=chore_id, at=timestamp()).single()
    return require_record(record, "Chore is not awaiting approval")

@router.post("/chores/{chore_id}/verify-arrival")
def verify_arrival(chore_id: str, item: ArrivalCheck, driver: Annotated[Driver, Depends(driver_from_request)]): return write(driver, arrival_tx, chore_id=chore_id, item=item)


def report_tx(tx: ManagedTransaction, values: dict[str, Any]) -> dict[str, Any]:
    record = tx.run("""MATCH (reporter:Resident {id:$reporter_id}),(accused:Resident {id:$accused_id}),(c:Task {id:$chore_id}) CREATE (r:Report {id:$report_id,reason:$reason,description:$description,proof_path:$proof_path,proof_size:$proof_size,immutable:true,active:true,created_at:$at}) CREATE (reporter)-[:FILED_REPORT]->(r) CREATE (r)-[:ACCUSES]->(accused) CREATE (r)-[:LINKED_TO_CHORE]->(c) SET c.state='DISPUTED',accused.dispute_count=coalesce(accused.dispute_count,0)+1 RETURN r.id AS report_id,c.state AS chore_state,accused.dispute_count AS dispute_count""", **values).single()
    return require_record(record, "Reporter, accused resident, or chore not found")

@router.post("/report-misconduct", status_code=201)
async def misconduct(reporter_id: Annotated[str, Form()], accused_id: Annotated[str, Form()], chore_id: Annotated[str, Form()], reason: Annotated[str, Form()], description: Annotated[str, Form()], proof_photo: Annotated[UploadFile, File()], driver: Annotated[Driver, Depends(driver_from_request)]):
    report_id = str(uuid.uuid4()); path, size = await save_upload(proof_photo, settings.proof_directory, report_id)
    try: return write(driver, report_tx, values={"reporter_id":reporter_id,"accused_id":accused_id,"chore_id":chore_id,"reason":reason,"description":description,"report_id":report_id,"proof_path":str(path),"proof_size":size,"at":timestamp()})
    except Exception: path.unlink(missing_ok=True); raise


def completion_tx(tx: ManagedTransaction, chore_id: str, senior_id: str) -> dict[str, Any]:
    record = tx.run("MATCH (s:Resident {id:$s})-[:POSTED]->(c:Task {id:$c}),(v:Resident)-[:ASSIGNED_TO]->(c) WHERE c.state='SUBMITTED_FOR_REVIEW' WITH c,v,toInteger(coalesce(c.complexity_points,10)) AS earned SET c.state='COMPLETED',c.completed_at=$at,v.points=coalesce(v.points,0)+earned RETURN c.id AS chore_id,v.id AS volunteer_id,earned AS points_earned,v.points AS balance", s=senior_id,c=chore_id,at=timestamp()).single()
    return require_record(record, "Chore is not submitted for review")

@router.post("/chores/{chore_id}/approve-complete")
def completion(chore_id: str, item: UserAction, driver: Annotated[Driver, Depends(driver_from_request)]):
    if item.chore_id != chore_id or item.action != "APPROVE_COMPLETE": raise HTTPException(status_code=422, detail="Invalid completion action")
    return write(driver, completion_tx, chore_id=chore_id, senior_id=item.user_id)


def redeem_tx(tx: ManagedTransaction, item: CouponRedemption) -> dict[str, Any]:
    record = tx.run("MATCH (r:Resident {id:$id}) WHERE coalesce(r.points,0)>=$cost SET r.points=r.points-$cost CREATE (c:Coupon {id:$coupon,name:$name,points_cost:$cost,status:'ISSUED',issued_at:$at}) CREATE (r)-[:REDEEMED {at:$at}]->(c) RETURN c.id AS coupon_id,c.name AS reward_name,r.points AS remaining_points,c.status AS status", id=item.user_id,cost=item.points_cost,coupon=str(uuid.uuid4()),name=item.reward_name,at=timestamp()).single()
    return require_record(record, "User not found or insufficient points")

@router.post("/redeem-rewards", status_code=201)
def redeem(item: CouponRedemption, driver: Annotated[Driver, Depends(driver_from_request)]): return write(driver, redeem_tx, item=item)


def admin_tx(tx: ManagedTransaction) -> list[dict[str, Any]]:
    return [record.data() for record in tx.run("MATCH (c:Task {state:'DISPUTED'})<-[:LINKED_TO_CHORE]-(r:Report)-[:ACCUSES]->(a:Resident) RETURN c.id AS chore_id,a.name AS accused,r.reason AS reason,r.description AS description,r.proof_path AS proof_path ORDER BY r.created_at DESC")]

@router.get("/admin/disputes")
def disputes(driver: Annotated[Driver, Depends(driver_from_request)]): return read(driver, admin_tx)
