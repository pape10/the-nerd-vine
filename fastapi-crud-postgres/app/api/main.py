import os
import json
import requests
import logging
from fastapi import FastAPI
from api.students import student
from api.db import database
log = logging.getLogger(__file__)
log.setLevel(level='WARN')


##initialize fastapi object
app = FastAPI(title='fastapi-demo',description='demo',version='1.0.0')


@app.on_event('startup')
async def startup():
    log.info('before connect')
    await database.connect()
    log.info('after connect')

@app.on_event('shutdown')
async def shutdown():
    log.info('before shutdown')
    await database.disconnect()
    log.info('after shutdown')


app.include_router(student,prefix='/api/v1/students', tags=['students'])