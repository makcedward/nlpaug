class ModelCache:
    def __init__(self):
        self._models = {}

    def get(self, key):
        return self._models.get(key)

    def clear(self):
        self._models.clear()

    def get_or_create(self, key, factory, force_reload=False, updates=None):
        if not force_reload:
            cached = self._models.get(key)
            if cached is not None:
                self._apply_updates(cached, updates)
                return cached

        model = factory()
        self._models[key] = model
        return model

    @staticmethod
    def _apply_updates(model, updates):
        if not updates:
            return

        for attr, value in updates.items():
            setattr(model, attr, value)
