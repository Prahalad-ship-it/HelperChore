from __future__ import annotations

from typing import Any
import numpy as np
from neo4j import ManagedTransaction
from app.models.schemas import MatchResult


def haversine_miles(target_lat: float, target_lon: float, coordinates: np.ndarray) -> np.ndarray:
    earth_radius = 3958.7613
    target = np.radians(np.asarray([target_lat, target_lon], dtype=np.float64))
    points = np.radians(np.asarray(coordinates, dtype=np.float64))
    delta = points - target
    a = np.sin(delta[:, 0] / 2) ** 2 + np.cos(target[0]) * np.cos(points[:, 0]) * np.sin(delta[:, 1] / 2) ** 2
    return 2 * earth_radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rank_candidates(rows: list[dict[str, Any]], senior_lat: float, senior_lon: float, top_k: int = 20) -> list[dict[str, Any]]:
    if not rows:
        return []
    matrix = np.asarray([[r["latitude"], r["longitude"], r["certified"], r["mean_rating"] if r["mean_rating"] is not None else 4.0, r["dispute_count"] or 0] for r in rows], dtype=np.float64)
    miles = haversine_miles(senior_lat, senior_lon, matrix[:, :2])
    skill = 20 + 15 * matrix[:, 2]
    distance = np.select([miles <= 1, miles <= 5, miles <= 10], [25, 18, 10], default=2)
    rating = np.clip(matrix[:, 3], 0, 5) / 5 * 25
    trust = np.maximum(0, 15 - matrix[:, 4] * 5)
    total = skill * .35 + distance * .25 + rating * .25 + trust * .15
    order = np.argsort(-total, kind="stable")[:top_k]
    return [MatchResult(volunteer_id=str(rows[int(i)]["volunteer_id"]), name=rows[int(i)].get("name"), score=round(float(total[i]), 3), skill_score=float(skill[i]), distance_score=float(distance[i]), rating_score=float(rating[i]), trust_score=float(trust[i]), distance_miles=round(float(miles[i]), 3), mean_rating=float(matrix[i, 3]), dispute_count=int(matrix[i, 4])).model_dump() for i in order]


def fetch_match_data(tx: ManagedTransaction, chore_id: str) -> dict[str, Any] | None:
    query = """
    MATCH (senior:Resident)-[:POSTED]->(chore:Task {id:$chore_id})
    OPTIONAL MATCH (chore)-[:REQUIRES]->(required:Skill)
    WITH senior, collect(DISTINCT required.name) AS skills
    MATCH (volunteer:Resident {role:'VOLUNTEER'})-[hs:HAS_SKILL]->(skill:Skill)
    WHERE skill.name IN skills AND volunteer.latitude IS NOT NULL AND volunteer.longitude IS NOT NULL
    WITH senior, volunteer, max(CASE WHEN hs.certified = true THEN 1 ELSE 0 END) AS certified
    CALL { WITH volunteer OPTIONAL MATCH (volunteer)-[review:REVIEWS]->(:Task) RETURN coalesce(avg(review.rating),4.0) AS mean_rating }
    CALL { WITH volunteer OPTIONAL MATCH (volunteer)<-[:ACCUSES]-(report:Report) WHERE coalesce(report.active,true)=true RETURN count(report) AS dispute_count }
    RETURN senior.latitude AS senior_latitude, senior.longitude AS senior_longitude,
      collect({volunteer_id:volunteer.id,name:volunteer.name,latitude:volunteer.latitude,longitude:volunteer.longitude,certified:certified,mean_rating:mean_rating,dispute_count:dispute_count}) AS candidates
    """
    record = tx.run(query, chore_id=chore_id).single()
    return None if record is None else record.data()
