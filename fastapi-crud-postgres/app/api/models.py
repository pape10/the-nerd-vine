from pydantic import BaseModel
from typing import Optional

class StudentIn(BaseModel):
    name: str
    degree: str
    section: str

class StudentOut(StudentIn):
    s_id: int

class StudentUpdate(StudentIn):
    name: Optional[str] = None
    degree: Optional[str] = None
    section: Optional[str] = None