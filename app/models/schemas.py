from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

class RegisterRequest(Strict):
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=200)
    role: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    skills: list[str] = Field(default_factory=list, max_length=50)
    certifications: list[str] = Field(default_factory=list, max_length=50)
    @field_validator("role")
    @classmethod
    def valid_role(cls, value: str) -> str:
        value = value.upper()
        if value not in {"SENIOR", "VOLUNTEER"}:
            raise ValueError("role must be Senior or Volunteer")
        return value

class LoginRequest(Strict):
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=200)

class PasswordResetRequest(Strict):
    name: str = Field(min_length=1, max_length=120)
    recovery_code: str = Field(min_length=12, max_length=80)
    new_password: str = Field(min_length=8, max_length=200)

class UserAction(Strict):
    user_id: str = Field(min_length=1, max_length=100)
    chore_id: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=80)

class ChoreRefinement(Strict):
    senior_id: str = Field(min_length=1, max_length=100)
    refinement_text: str = Field(min_length=1, max_length=5000)
    @field_validator("refinement_text")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip(): raise ValueError("refinement_text cannot be blank")
        return value.strip()

class ArrivalCheck(Strict):
    senior_id: str = Field(min_length=1, max_length=100)
    approved: bool

class CopilotQuestion(Strict):
    volunteer_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=3000)

class CouponRedemption(Strict):
    user_id: str = Field(min_length=1, max_length=100)
    reward_name: str = Field(min_length=1, max_length=120)
    points_cost: int = Field(gt=0, le=1_000_000)

class MatchResult(Strict):
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
