class RAM:
    def __init__(self, size: int = 512, init_value: int = 0):
        """Initialize the RAM with a given size (default is 512 bytes) and initial value."""
        self.memory = [init_value] * size

    def __len__(self):
        """Return the size of the memory."""
        return len(self.memory)

    def __getitem__(self, key: int):
        """Return a byte from the memory."""
        return self.read(key)

    def __setitem__(self, key: int, value: int):
        """Write a byte to the memory."""
        self.write(key, value)

    def read(self, address: int) -> int:
        """Read a byte from a specific memory address."""
        if 0 <= address < len(self.memory):
            return self.memory[address]
        else:
            raise ValueError(f"Address {address} out of bounds")

    def write(self, address: int, value: int):
        """Write a byte to a specific memory address."""
        if 0 <= address < len(self.memory):
            self.memory[address] = value & 0xFF  # Ensure 8-bit value
        else:
            raise ValueError(f"Address {address} out of bounds")

    def load_program(self, program: list):
        """Load a program into memory."""
        if len(program) > len(self.memory):
            raise ValueError("Program size exceeds memory size")
        self.memory[: len(program)] = program

    def dump(self):
        """Dump the memory content."""
        return self.memory

    def visualize(self):
        """Visualize memory content in hexadecimal format."""
        return " ".join(f"{byte:02X}" for byte in self.memory)
