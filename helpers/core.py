class Core:
    def __init__(self, id, ram):
        self.id = id
        self.ram = ram

    def execute(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)