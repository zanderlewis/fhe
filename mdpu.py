import numpy as np
from ram import RAM

class Core:
    def __init__(self, id, ram):
        self.id = id
        self.ram = ram

    def execute(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)


class MultiDimensionalProcessor:
    def __init__(self, shape, num_cores, ram_size, bits=16, dtype=int, debug=False):
        self.shape = shape
        self.dtype = dtype
        self.array = np.zeros(shape, dtype=dtype)
        self.ram = RAM(size=ram_size)
        self.cores = [Core(i, self.ram) for i in range(num_cores)]
        self.registers = [0] * bits
        self.pc = 0
        self.stack = []
        self.running = True
        self.debug_mode = debug
        self.bits = bits

        # Instructions (opcode to function mapping)
        self.instructions = {
            0x01: self.load_immediate,
            0x02: self.add,
            0x03: self.store,
            0x04: self.load,
            0x05: self.sub,
            0x06: self.mul,
            0x07: self.div,
            0x08: self.and_op,
            0x09: self.or_op,
            0x0A: self.xor_op,
            0x0B: self.not_op,
            0x0C: self.jmp,
            0x0D: self.jz,
            0x0E: self.jnz,
            0x0F: self.push,
            0x10: self.pop,
            0xFF: self.halt
        }

    def fetch(self):
        if self.pc < len(self.ram):
            opcode = self.ram.read(self.pc)
            self.pc += 1
            return opcode
        else:
            raise Exception(f"Program counter out of bounds: {self.pc}")

    def decode_execute(self, opcode):
        if opcode in self.instructions:
            self.instructions[opcode]()
        else:
            raise Exception(f"Unknown opcode: {opcode}")

    def load_immediate(self):
        reg = self.ram.read(self.pc)
        value = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            self.registers[reg] = value
        else:
            raise IndexError(f"Register index out of range: {reg}")

    def add(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] += self.registers[reg2]
        self.registers[reg1] &= 0xFF

    def sub(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] -= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    def mul(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] *= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    def div(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        if self.registers[reg2] == 0:
            raise Exception("Division by zero")
        self.registers[reg1] //= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    def and_op(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] &= self.registers[reg2]

    def or_op(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] |= self.registers[reg2]

    def xor_op(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] ^= self.registers[reg2]

    def not_op(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.registers[reg] = ~self.registers[reg] & 0xFF

    def jmp(self):
        addr = self.ram.read(self.pc)
        if addr < len(self.ram):
            self.pc = addr
        else:
            raise Exception(f"Jump address out of bounds: {addr}")

    def jz(self):
        addr = self.ram.read(self.pc)
        self.pc += 1
        if self.registers[0] == 0:
            if addr < len(self.ram):
                self.pc = addr
            else:
                raise Exception(f"Jump address out of bounds: {addr}")

    def jnz(self):
        addr = self.ram.read(self.pc)
        self.pc += 1
        if self.registers[0] != 0:
            if addr < len(self.ram):
                self.pc = addr
            else:
                raise Exception(f"Jump address out of bounds: {addr}")

    def push(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.stack.append(self.registers[reg])

    def pop(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        if not self.stack:
            raise Exception("Stack underflow")
        self.registers[reg] = self.stack.pop()

    def store(self):
        reg = self.ram.read(self.pc)
        addr = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            if addr < len(self.ram):
                self.ram.write(addr, self.registers[reg])
            else:
                raise Exception(f"Store address out of bounds: {addr}")
        else:
            raise IndexError(f"Register index out of range: {reg}")

    def load(self):
        reg = self.ram.read(self.pc)
        addr = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            if addr < len(self.ram):
                self.registers[reg] = self.ram.read(addr)
            else:
                raise Exception(f"Load address out of bounds: {addr}")
        else:
            raise IndexError(f"Register index out of range: {reg}")

    def halt(self):
        self.running = False

    def run(self):
        while self.running:
            opcode = self.fetch()
            self.decode_execute(opcode)
            self.debug()

    def debug(self):
        if self.debug_mode:
            print(f"PC: {self.pc}, Registers: {self.registers}, Stack: {self.stack}")

    def load_program(self, program):
        self.ram.load_program(program)

    def __repr__(self):
        return (f"MultiDimensionalProcessor(shape={self.shape}, dtype={self.dtype})\n"
                f"Array:\n{self.array}\n"
                f"RAM:\n{self.ram.visualize()}\n"
                f"Registers: {self.registers}\n"
                f"Stack: {self.stack}\n")


# Example Usage
if __name__ == "__main__":
    # Sample program that finds the sum of two numbers
    program = [
        0x01, 0x00, 0x0A,  # Load 10 into R0
        0x01, 0x01, 0x05,  # Load 5 into R1
        0x02, 0x00, 0x01,  # Add R1 to R0
        0x0D, 0x07,        # Jump to address 0x07 if R0 is zero
        0xFF               # Halt
    ]

    processor = MultiDimensionalProcessor(shape=(3, 4, 5), num_cores=4, ram_size=64, bits=16, debug=True)
    processor.load_program(program)
    processor.run()

    # Print the sum of the two numbers
    print(f"Sum: {processor.registers[0]}")
