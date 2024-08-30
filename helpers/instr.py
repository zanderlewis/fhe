class instr:
    # Class to convert contents of .instr files to a Python array of opcodes
    def __init__(self):
        self.opcodes = []

    def load(self, path):
        if not path.endswith(".instr"):
            raise ValueError("Invalid file type. Please provide a .instr file")
        with open(path, "r") as f:
            self.opcodes = f.readlines()

    def parse(self):
        # Remove comments (#)
        self.opcodes = [x.split("#")[0] for x in self.opcodes]
        # Remove empty lines
        self.opcodes = [x.strip() for x in self.opcodes if x.strip()]
        # Split each line into individual opcodes and convert to integers
        parsed_opcodes = []
        for line in self.opcodes:
            parsed_opcodes.extend(int(x, 16) for x in line.split())
        self.opcodes = parsed_opcodes

    def get(self):
        return self.opcodes

    def __str__(self):
        return str(self.opcodes)