"""Database, model-feature, chart and HTTP integration coverage."""

from pathlib import Path
import sqlite3
import sys
import numpy as np
import pytest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "report"))


@pytest.fixture
def db_path():
    return project_root / "python-package" / "employee_events" / "employee_events.db"


@pytest.fixture
def db_conn(db_path):
    connection = sqlite3.connect(db_path)
    yield connection
    connection.close()


@pytest.fixture
def table_names(db_conn):
    return [r[0] for r in db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def test_db_exists(db_path):
    assert db_path.is_file()


def test_employee_table_exists(table_names):
    assert "employee" in table_names


def test_team_table_exists(table_names):
    assert "team" in table_names


def test_employee_events_table_exists(table_names):
    assert "employee_events" in table_names


def test_notes_table_exists(table_names):
    assert "notes" in table_names


@pytest.mark.parametrize("kind", ["employee", "team"])
def test_queries_match_database(kind, db_conn):
    from employee_events import Employee, Team

    model = Employee() if kind == "employee" else Team()
    assert len(model.names()) == db_conn.execute(f"SELECT COUNT(*) FROM {kind}").fetchone()[0]
    entity_id = model.names()[0][1]
    assert model.username(entity_id)
    data = model.event_counts(entity_id)
    expected = db_conn.execute(
        f"SELECT SUM(positive_events), SUM(negative_events) FROM employee_events WHERE {kind}_id=?",
        (entity_id,),
    ).fetchone()
    assert tuple(data[["positive_events", "negative_events"]].sum()) == expected
    features = model.model_data(entity_id)
    assert list(features.columns) == ["positive_events", "negative_events"]
    assert features.shape[0] > 0
    assert list(model.notes(entity_id).columns) == ["note_date", "note"]
    with pytest.raises(ValueError):
        model.event_counts("1 OR 1=1")


@pytest.mark.parametrize("kind", ["employee", "team"])
def test_model_probability(kind):
    from employee_events import Employee, Team
    from dashboard import BarChart

    model = Employee() if kind == "employee" else Team()
    features = model.model_data(model.names()[0][1])
    pred = BarChart.predictor.predict_proba(features)
    assert pred.shape == (len(features), 2)
    assert np.all((pred >= 0) & (pred <= 1))
    assert np.allclose(pred.sum(axis=1), 1)


@pytest.fixture
def client():
    from dashboard import app
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("url", ["/", "/employee/1", "/employee/2", "/team/1", "/team/2"])
def test_reports(client, url):
    response = client.get(url)
    assert response.status_code == 200
    assert response.text.count("data:image/png;base64,") == 2
    assert "Predicted Recruitment Risk" not in response.text  # chart title is rendered into pixels
    assert "<table" in response.text
    assert 'name="profile_type"' in response.text
    assert 'name="user-selection"' in response.text


@pytest.mark.parametrize("kind", ["Employee", "Team"])
def test_dropdown_and_form(client, kind):
    response = client.get("/update_dropdown", params={"profile_type": kind})
    assert response.status_code == 200
    assert "<option" in response.text
    response = client.post(
        "/update_data", data={"profile_type": kind, "user-selection": "2"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/{kind.lower()}/2"


@pytest.mark.parametrize(
    "url,code",
    [
        ("/employee/999999", 404),
        ("/team/999999", 404),
        ("/update_dropdown?profile_type=Invalid", 400),
    ],
)
def test_invalid_profiles(client, url, code):
    assert client.get(url).status_code == code


def test_invalid_form(client):
    assert (
        client.post(
            "/update_data", data={"profile_type": "Employee", "user-selection": "invalid"}
        ).status_code
        == 400
    )
