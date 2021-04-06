import logging
from typing import List
from fastapi import Header, APIRouter , HTTPException
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
async def get_students():
    log.info('getting student details')
    return fake_student_data


@student.post("/",status_code=201)
async def add_student(payload: Student):
    log.info("adding students")
    student = payload.dict()
    fake_student_data.append(student)
    return {'id': len(fake_student_data) -1}


@student.get("/{id}",response_model=Student)
async def get_student(id: int):
    log.info('getting student detail for student with id {0}'.format(str(id)))
    student_length = len(fake_student_data)
    if 0 <= id < student_length :
        return fake_student_data[id]
    raise HTTPException(status_code=404,detail="student with given id {0} not found ".format(str(id)))

## put route
@student.put('/{id}')
async def update_student(id: int,payload: Student):
    log.info('trying to modify student data for id {0}'.format(str(int)))
    student = payload.dict()
    student_length = len(fake_student_data)
    if 0 <= id < student_length:
        fake_student_data[id] = student
        return None
    raise HTTPException(status_code=404,detail="student with given id {0} not found ".format(str(id)))


## delete route
@student.delete('/{id}')
async def delete_student(id: int):
    log.info('deleting student with id {0}'.format(str(id)))
    student_length = len(fake_student_data)
    if 0 <= id < student_length :
        del fake_student_data[id]
        return None
    raise HTTPException(status_code=404,detail="student with given id {0} not found ".format(str(id)))
