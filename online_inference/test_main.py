from starlette.testclient import TestClient

from main import app
from src.features import COL_ORDER

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': 'Welcome to Heart Disease classificator. Check /docs'}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    

def test_predict_correct_request():
    with client:
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
        response = client.get(
            "/predict/", json={"data": [req_data], "feature_names": req_feats}
        )
        assert response.status_code == 200
        assert response.json() == [{"disease": 1}]


def test_predict_incorrect_json():
    with client:
        response = client.get("/predict/", json={"incorrect json": 0})
        assert response.status_code == 422
