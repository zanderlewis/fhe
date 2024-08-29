import numpy as np
import sys, time
from ram import RAM
from helpers.instr import instr


class Core:
    def __init__(self, id, ram):
        self.id = id
        self.ram = ram

    def execute(self, operation, *args, **kwargs):
        return operation(*args, **kwargs)


class MDPU:
    def __init__(self, shape, num_cores, ram_size, bits=16, dtype=int, debug=False):
        self.shape = shape
        self.dtype = dtype
        self.array = np.zeros(shape, dtype=dtype)
        self.ram = RAM(size=ram_size)
        self.cores = [Core(i, self.ram) for i in range(num_cores)]
        self.registers = [0] * bits
        self.constant_registers = [False] * bits  # Track which registers are constants
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
            0x11: self.const,
            0x12: self.swap,
            0x13: self.inc,
            0x14: self.dec,
            0xFF: self.halt,
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

    def check_constant(self, reg):
        if self.constant_registers[reg]:
            print(f'Trying to modify constant register "R{reg}" at PC {self.pc}.')
            sys.exit(1)

    # 0x01
    def load_immediate(self):
        reg = self.ram.read(self.pc)
        value = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            self.check_constant(reg)
            self.registers[reg] = value
        else:
            raise IndexError(f"Register index out of range: {reg}")

    # 0x02
    def add(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.check_constant(reg1)
        self.registers[reg1] += self.registers[reg2]
        self.registers[reg1] &= 0xFF

    # 0x05
    def sub(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.check_constant(reg1)
        self.registers[reg1] -= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    # 0x06
    def mul(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.check_constant(reg1)
        self.registers[reg1] *= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    # 0x07
    def div(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        if self.registers[reg2] == 0:
            raise Exception("Division by zero")
        self.check_constant(reg1)
        self.registers[reg1] //= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    # 0x08
    def and_op(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.check_constant(reg1)
        self.registers[reg1] &= self.registers[reg2]

    # 0x09
    def or_op(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.check_constant(reg1)
        self.registers[reg1] |= self.registers[reg2]

    # 0x0A
    def xor_op(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.check_constant(reg1)
        self.registers[reg1] ^= self.registers[reg2]

    # 0x0B
    def not_op(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.check_constant(reg)
        self.registers[reg] = ~self.registers[reg] & 0xFF

    # 0x0C
    def jmp(self):
        addr = self.ram.read(self.pc)
        if addr < len(self.ram):
            self.pc = addr
        else:
            raise Exception(f"Jump address out of bounds: {addr}")

    # 0x0D
    def jz(self):
        addr = self.ram.read(self.pc)
        self.pc += 1
        if self.registers[0] == 0:
            if addr < len(self.ram):
                self.pc = addr
            else:
                raise Exception(f"Jump address out of bounds: {addr}")

    # 0x0E
    def jnz(self):
        addr = self.ram.read(self.pc)
        self.pc += 1
        if self.registers[0] != 0:
            if addr < len(self.ram):
                self.pc = addr
            else:
                raise Exception(f"Jump address out of bounds: {addr}")

    # 0x0F
    def push(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.stack.append(self.registers[reg])

    # 0x10
    def pop(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        if not self.stack:
            raise Exception("Stack underflow")
        self.check_constant(reg)
        self.registers[reg] = self.stack.pop()

    # 0x03
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

    # 0x04
    def load(self):
        reg = self.ram.read(self.pc)
        addr = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            if addr < len(self.ram):
                self.check_constant(reg)
                self.registers[reg] = self.ram.read(addr)
            else:
                raise Exception(f"Load address out of bounds: {addr}")
        else:
            raise IndexError(f"Register index out of range: {reg}")

    # 0x11
    def const(self):
        reg = self.ram.read(self.pc)
        value = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            self.registers[reg] = value
            self.constant_registers[reg] = True  # Mark this register as a constant
        else:
            raise IndexError(f"Register index out of range: {reg}")

    # 0x12
    def swap(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg1 < len(self.registers) and reg2 < len(self.registers):
            self.registers[reg1], self.registers[reg2] = (
                self.registers[reg2],
                self.registers[reg1],
            )
        else:
            raise IndexError(f"Register index out of range: reg1={reg1}, reg2={reg2}")

    # 0x13
    def inc(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.registers[reg] += 1
        self.registers[reg] &= 0xFF

    # 0x14
    def dec(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.registers[reg] -= 1
        self.registers[reg] &= 0xFF

    # 0xFF
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
        return (
            f"MultiDimensionalProcessor(shape={self.shape}, dtype={self.dtype})\n"
            f"Array:\n{self.array}\n"
            f"RAM:\n{self.ram.visualize()}\n"
            f"Registers: {self.registers}\n"
            f"Stack: {self.stack}\n"
        )


# Example Usage
if __name__ == "__main__":
    # Initialize the instruction file
    instr_file = instr()
    instr_file.load("programs/mdpu/0.instr")

    # Parse the instruction file
    instr_file.parse()

    # Get the program
    program = instr_file.get()

    # Initialize the processor
    processor = MDPU(shape=(3, 4, 5), num_cores=4, ram_size=64, bits=16)

    # Load the program into RAM
    processor.load_program(program)

    # Record the start time
    start_time = time.time()

    # Run the processor
    processor.run()

    # Record the end time
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the final state of the registers and stack
    print(f"Registers: {processor.registers}")
    print(f"Stack: {processor.stack}")
    print(f"Execution time: {execution_time:.6f} seconds")
