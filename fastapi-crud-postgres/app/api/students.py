import logging
from typing import List
from fastapi import Header, APIRouter , HTTPException
from api.models import StudentIn, StudentOut, StudentUpdate
from api import db_manager

log = logging.getLogger(__file__)
log.setLevel(level='WARN')



student = APIRouter()

@student.post("/",status_code=201)
async def add_student(payload: StudentIn):
    log.info("adding students")
    s_id = await db_manager.add_student(payload)
    response = {
        'id': s_id,
        **payload.dict()
    }
    return response


@student.get("/",response_model=List[StudentOut])
async def get_students():
    log.info('getting student details')
    return await db_manager.get_all_students()



@student.get("/{s_id}",response_model=StudentOut)
async def get_student(s_id: int):
    log.info('getting student detail for student with id {0}'.format(str(s_id)))
    student = await db_manager.get_student(s_id)
    if not student:
        raise HTTPException(status_code=404,detail="student with given id {0} not found ".format(str(s_id)))
    return student



## put route
@student.put('/{s_id}')
async def update_student(s_id: int,payload: StudentUpdate):
    log.info('trying to modify student data for id {0}'.format(str(s_id)))
    student = await db_manager.get_student(s_id)
    if not student:
        raise HTTPException(status_code=404,detail="student with given id {0} not found ".format(str(id)))

    update_data = payload.dict(exclude_unset=True)
    student_in_db = StudentUpdate(**student)
    updated_student = student_in_db.copy(update=update_data)

    return await db_manager.update_student(s_id, updated_student)


## delete route
@student.delete('/{s_id}')
async def delete_student(s_id: int):
    log.info('deleting student with id {0}'.format(str(s_id)))
    student = await db_manager.get_student(s_id)
    if not student:
        raise HTTPException(status_code=404,detail="student with given id {0} not found ".format(str(s_id)))
    return await db_manager.delete_student(s_id)