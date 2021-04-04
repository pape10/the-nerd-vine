import logging
from typing import List
from fastapi import Header, APIRouter
from api.models import Student


log = logging.getLogger(__file__)
log.setLevel(level='WARN')




fake_student_data = [
    {
        'name': 'the-nerd-vine',
        'degree': 'grad',
        'section': 'B'
    },
    {
        'name': 'the-nerd-vine2',
        'degree': 'grad',
        'section': 'B'
    }
]

student = APIRouter()

@student.get("/",response_model=List[Student])
async def get_student():
    log.info('getting student details')
    return fake_student_data


@student.post("/",status_code=201)
async def add_student(payload: Student):
    log.info("adding students")
    student = payload.dict()
    fake_student_data.append(student)
    return {'id': len(fake_student_data) -1}
