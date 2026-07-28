from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_app_loads():
    assert app is not None


def test_triage_route_exists():
    routes = [route.path for route in app.routes]
    assert "/triage" in routes