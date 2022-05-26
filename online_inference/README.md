ml-project: online inference  
==============================  

  
Project Organization  
------------  
  
    ├── model                      <- Trained and serialized models  
    │   └── model.pkl              <- Pre-trained model  
    │   └── transformer.pkl        <- Pre-trained transformer  
    ├── src                        <- Source code for use in this project  
    │   └── features.py            <- Scripts to turn raw data into features  
    │   └── utils                  <- Some useful utils  
    │   │   └── scaler.py          <- Custrom tansformer (from previous project)  
    │   │   └── utils.py           <- Utils to load model and set up configs  
    │   └── config.yaml            <- Config  
    ├── Dockerfile                 <- Dockerfile to create docker image  
    ├── main.py                    <- FastAPI application  
    ├── make_request.py            <- Script to generate requests for API  
    ├── README.md                  <- The top-level README for developers using this project  
    ├── requiriments.txt           <- The requirements file  
    └── test_app.py                <- Test for API  
  
--------  
  
### Usage for the project  
------------  

#### Use Docker image from Docker hub   

    docker pull tinctura/ml_app:v5
    docker run -p 8099:8099 tinctura/ml_app:v5

#### Build local container  

    docker build -t tinctura/ml_app .  
    docker run -p 8099:8099 tinctura/ml_app
    
#### Tests  

    pytest -v  

#### Make requests  

    python make_request.py

#### Docker image optimization.

At first attempt I've just built the image and it was about 1.4Gb  
What I've tried:  
- Slim version of python 3.9, then slim version of python 3.8 and stopped on python 3.6-slim  
- Removed a lot of unnecessary packages from requirements  
- Made as many layers as possible (I think that it's still possible to optimize copy commands)  

As a result I acieved about 0.4gb size at my local storage  
Great news! At docker hub my compressed image weighsts only 129mb  
So I've reduced the size more than 10 times, cool! ⚗️
