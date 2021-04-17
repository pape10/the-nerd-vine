## fastapi-demo

fastapi demo

## for creating the virtual environment

```
    conda env create -f environment.yml
    conda activate <conda env name>
```


## running the application
```
    uvicorn api.main:app --reload --port 8000
```

## data structure in database
```
    s_id : int (primary key , auto increment)
    name : string
    degree : string
    section : string
```

## env variable
```
    export POSTGRES_DB=postgresql://postgres:mysecretpassword@localhost:5432/postgres
```