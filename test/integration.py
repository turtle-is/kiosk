from MockResponse import MockResponse

def test_detection_to_cart(monkeypatch):
    dummy_detection = [{
        "bbox": [10, 20, 30, 40],
        "confidence": 0.9,
        "class": "Lays",
        "class_id": 0
    }]

    mock_product = [{"category": 0, "name": "Lays", "price": "50"}]
    monkeypatch.setattr("backend_service.app.requests.get", lambda url: MockResponse(mock_product))

    from backend_service.app import update_cart, cart
    update_cart(dummy_detection)
    assert cart[0]["quantity"] == 1
