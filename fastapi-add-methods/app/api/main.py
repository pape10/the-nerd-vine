import os
import json
import requests
import logging
from fastapi import FastAPI
from api.students import student

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


app.include_router(student,prefix='/api/v1/students', tags=['students'])