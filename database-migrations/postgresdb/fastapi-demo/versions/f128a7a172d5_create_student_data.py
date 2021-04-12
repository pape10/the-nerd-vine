"""create student data

Revision ID: f128a7a172d5
Revises: 
Create Date: 2021-04-13 03:48:55.814913

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import Column,String

# revision identifiers, used by Alembic.
revision = 'f128a7a172d5'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("student_data",
                    Column("name",String,nullable=False),
                    Column("degree",String,nullable=False),
                    Column("section",String,nullable=False),
                    schema="fastapi-demo")


def downgrade():
    op.drop_table("student_data",schema="fastapi-demo")


