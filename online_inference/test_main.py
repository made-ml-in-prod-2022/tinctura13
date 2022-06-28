import pytest
from starlette.testclient import TestClient

from main import app
from src.features import COL_ORDER


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_read_main(client):
    response = client.get("/")
    assert  200 == response.status_code
    assert response.json() == {'message': 'Welcome to Heart Disease classificator. Check /docs'}
 

def test_health(client):
    response = client.get("/health")
    assert 200 == response.status_code
    

def test_predict_correct_request(client):
    req_data = [
            55.666,
            0.0,
            1.0,
            88.666,
            220.666,
            1.0,
            1.0,
            118.666,
            0.0,
            -0.666,
            2.0,
            1.0,
            2.0,
        ]
    req_feats = COL_ORDER
    response = client.get("predict/", json={"data": [req_data], "feature_names": req_feats})
    assert 200 == response.status_code
    assert response.json() == [{"disease": 1}]


def test_predict_incorrect_json(client):
    response = client.get("/predict/", json={"incorrect json": 0})
    assert response.status_code == 422


def test_predict_empy_data(client):
    response = client.get("/predict/", json={"data": [], "feature_names": COL_ORDER})
    assert response.status_code == 400
    assert response.json() == {"detail": "Input data list is empty"}
