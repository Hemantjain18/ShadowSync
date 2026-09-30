"""Tests for backend API schema and mineralogy endpoint."""

import pytest
from server.app import create_app


def test_mineralogy_endpoint_schema():
    app = create_app()
    if app is None:
        pytest.skip("Flask not installed in test environment")

    client = app.test_client()
    res = client.get("/api/mineralogy?region=apollo11")

    assert res.status_code == 200
    data = res.get_json()

    assert "region" in data
    assert "mineralogy" in data
    assert "classMapImage" in data
    assert "legend" in data

    min_report = data["mineralogy"]
    assert "source" in min_report
    assert "classes" in min_report
    assert "indices" in min_report
    assert "mean_spectrum" in min_report
    assert "quality" in min_report

    # Check JSON safety (no NaN or non-standard types)
    import json
    json_str = json.dumps(data)
    assert "NaN" not in json_str
