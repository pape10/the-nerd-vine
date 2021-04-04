import os
import json
import requests
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
log = logging.getLogger(__file__)
log.setLevel(level='WARN')


##initialize fastapi object
app = FastAPI(title='fastapi-demo',description='demo',version='1.0.0')


@app.on_event('startup')
async def startup():
    log.info('before connect')
    pass
    log.info('after connect')

@app.on_event('shutdown')
async def shutdown():
    log.info('before shutdown')
    pass
    log.info('after shutdown')


fake_student_data = [
    {
        'name': 'the-nerd-vine',
        'degree': 'grad',
        'section': 'B'
    },
    {
        'name': 'the-nerd-vine2',
        'degree': 'grad2',
        'section': 'B'
    }
]

class Student(BaseModel):
    name: str
    degree: str
    section: str

@app.get("/",response_model=List[Student],tags=['get'])
async def get_student():
    log.info('getting student details')
    return fake_student_data


@app.post("/",status_code=201,tags=['post'])
async def add_student(payload: Student):
    log.info("adding students")
    student = payload.dict()
    fake_student_data.append(student)
    return {'id': len(fake_student_data) -1}
