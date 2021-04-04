import os
import json
import requests
import logging
from fastapi import FastAPI

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


@app.get("/getroute",tags=["demo"])
async def getstatus(name : str):
    log.info("demo route hit for name {0}".format(name))
    ## do bunch of things
    return {'message':'yes','name':name}

@app.post("/postroute",tags=["demo"])
async def poststatus(name : str):
    log.info("demo route for post request for name {0}".format(name))
    ## do bunch of things
    return {'message':'yes','name':name}