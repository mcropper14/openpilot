class AlpamayoRunner:
    def __init__(self):
        self.ready = False
        self.model = None
        self._load_stub()

    def _load_stub(self):
        self.ready = True

    def predict(self, obs):
        v_ego = obs["v_ego"]

        return {
            "valid": True,
            "text": f"stub Alpamayo | vEgo={v_ego:.2f}",
            "confidence": 0.95,
            "mode": "stub",
            "waypoints": [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
        }