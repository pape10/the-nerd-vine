#databse-migration-scripts

## create env
conda env create -f environment.yml


## directory structure
database-migration ---------> postgresdb1 ---------> schema1
                                          ---------> schema2  
                   ---------> sales_data
                   ---------> postgresdb3 

## alembic init 
alembic init <schemaname>

## create revision
alembic revision -m "<message>"


## setup env variables
export postgresdb=postgresql://postgres:mysecretpassword@localhost:5432/postgres

## alembic upgrades
alembic upgrade head --sql

alembic upgrade head


## alembic downgrade
alembic downgrade head:-1 --sql