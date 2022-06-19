ml-project: airflow & mlflow
==============================  

  
Project Organization  
------------  
  
    ├── dags                                        <- DAG files  
    │   └── 01_generate_data.py                     <- Generate fake data for our model
    │   └── 02_train.py                             <- Training pipline  
    │   └── 03_predict.py                           <- Model inference
    │   └── utils.py                                <- Some config for airflow & mlflow
    ├── data                                        <- For data files  
    ├── docker-postgres-multiple-databases          
    │   └── create-multiple-postgresql-databases.sh <- Scripts to create multiple postgres db 
    ├── images                                      <- Imagesd for docker containers
    │   ├── airflow-docker 
    │   │   └── Dockerfile       
    │   ├── airflow-download 
    │   │   ├── Dockerfile  
    │   │   └── download.py                     
    │   ├── airflow-ml-base   
    │   │   ├── Dockerfile   
    │   │   └── requirements.txt      
    │   ├── airflow-predict
    │   │   ├── Dockerfile  
    │   │   └── predict.py          
    │   ├── airflow-preprocess 
    │   │   ├── Dockerfile  
    │   │   └── preprocess.py     
    │   ├── airflow-split 
    │   │   ├── Dockerfile  
    │   │   └── split.py        
    │   ├── airflow-train 
    │   │   ├── Dockerfile  
    │   │   └── train.py     
    │   ├── airflow-validate 
    │   │   ├── Dockerfile  
    │   │   └── val.py        
    │   └── mlflow  
    │       ├── Dockerfile   
    │       └── requirements.txt                
    ├── tests                      
    │   └── test_dags.py                            <- Test for DAGs
    ├── docker-compose.yml                          <- Docker compose config   
    └── README.md                                   <- README for using this project
  
--------  
  
### Usage for the project  
------------  

#### Build & run
```
sudo docker compose up --build
```
#### Stop & remove containers
```
^C (cntrl-C)
docker compose down
```
#### Tests
```
pytest -v
```