class instr:
    # language to convert contents of .instr files to a py array
    def __init__(self):
        self.instr = []

    def load(self, path):
        if not path.endswith(".instr"):
            raise ValueError("Invalid file type. Please provide a .instr file")
        with open(path, "r") as f:
            self.instr = f.readlines()

    def parse(self):
        # remove comments (#)
        self.instr = [x.split("#")[0] for x in self.instr]
        # remove empty lines
        self.instr = [x.strip() for x in self.instr if x.strip()]
        # Split each line into individual opcodes and convert to integers
        parsed_instr = []
        for line in self.instr:
            parsed_instr.extend(int(x, 16) for x in line.split())
        self.instr = parsed_instr

    def get(self):
        return self.instr

    def __str__(self):
        return str(self.instr)
