from unittest.mock import patch
from backend_service.app import update_cart, cart

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json


def test_update_cart_with_product():
    sample_detection = [{
        "bbox": [10, 20, 30, 40],
        "confidence": 0.92,
        "class": "Lays",
        "class_id": 0
    }]

    mock_product_data = [{"category": 0, "name": "Lays", "price": "50"}]

    with patch("backend_service.app.requests.get", return_value=MockResponse(mock_product_data)):
        cart.clear()
        update_cart(sample_detection)
        assert 0 in cart
