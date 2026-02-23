from pydantic import BaseModel
from typing import Optional


class ResumeRequest(BaseModel):
    resume_text: str
    job_description: str
    candidate_name: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "Experienced Python developer with 3 years using FastAPI, PostgreSQL, Docker and AWS. Built ML pipelines with scikit-learn and pandas.",
                "job_description": "Looking for a backend engineer with Python, FastAPI, Docker, AWS, and SQL experience. Machine learning knowledge is a plus.",
                "candidate_name": "Rahul Sharma"
            }
        }