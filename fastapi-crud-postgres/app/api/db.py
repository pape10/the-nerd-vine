import os
from sqlalchemy import (create_engine, Column, Integer, MetaData, String, Table)

from databases import Database

DATABASE_URL = os.environ['POSTGRES_DB']

engine = create_engine(DATABASE_URL)

database = Database(DATABASE_URL)

metadata = MetaData()
students = Table(
    'student_data',
    metadata,
    Column('s_id',Integer,primary_key=True),
    Column('name',String(200)),
    Column('degree',String(200)),
    Column('section',String(200)),
    schema='fastapi-demo'
)

