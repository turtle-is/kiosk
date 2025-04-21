class MockResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200
    def json(self):
        return self._data
