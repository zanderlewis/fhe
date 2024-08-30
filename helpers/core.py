class Core:
    def __init__(self, id, ram):
        self.id = id
        self.ram = ram
        self.pc = 0
        self.running = True

    def fetch(self):
        if self.pc < len(self.ram):
            opcode = self.ram.read(self.pc)
            self.pc += 1
            return opcode
        else:
            raise Exception(f"Program counter out of bounds: {self.pc}")

    def decode_execute(self, opcode):
        # This method should be implemented in the subclass or passed as a parameter
        raise NotImplementedError("decode_execute method should be implemented in the subclass")

    def run(self):
        while self.running:
            opcode = self.fetch()
            self.decode_execute(opcode)

    def execute(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)