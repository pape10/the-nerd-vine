from pydantic import BaseModel

class Student(BaseModel):
    name: str
    degree: str
    section: str