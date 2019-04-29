class BaseNER:
    def entities(self, text, type=None):
        raise NotImplementedError("This method must be redefined")
