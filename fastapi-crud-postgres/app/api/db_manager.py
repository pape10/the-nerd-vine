from api.models import StudentIn, StudentOut, StudentUpdate
from api.db import students, database

async def add_student(payload: StudentIn):
    query = students.insert().values(**payload.dict())

    return await database.execute(query)

async def get_all_students():
    query = students.select()
    return await database.fetch_all(query=query)

async def get_student(s_id):
    query = students.select(students.c.s_id==s_id)
    return await database.fetch_one(query)

async def update_student(s_id: int,payload: StudentUpdate):
    query = (
        students
        .update()
        .where(students.c.s_id==s_id)
        .values(**payload.dict())
    )
    return await database.execute(query=query)

async def delete_student(s_id: int):
    query = students.delete().where(students.c.s_id==s_id)
    return await database.execute(query=query)