import numpy as np
import time
from helpers.ram import RAM
from helpers.instr import instr

class PPU:
    def __init__(self, ram_size, bits=16, debug=False):
        self.ram = RAM(size=ram_size)
        self.registers = [0] * bits
        self.vector_registers = [np.zeros(3) for _ in range(bits)]  # 3D vectors
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
            0x08: self.vector_add,
            0x09: self.vector_sub,
            0x0A: self.vector_dot,
            0x0B: self.vector_cross,
            0x0C: self.calculate_force,
            0x0D: self.calculate_velocity,
            0x0E: self.calculate_position,
            0x0F: self.jmp,
            0x10: self.jz,
            0x11: self.jnz,
            0x12: self.push,
            0x13: self.pop,
            0x14: self.vector_normalize,
            0x15: self.vector_scale,
            0x16: self.vector_magnitude,
            0xFF: self.halt,
        }

    def fetch(self):
        if self.pc < len(self.ram.memory):
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

    # 0x01
    def load_immediate(self):
        reg = self.ram.read(self.pc)
        value = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            self.registers[reg] = value
        else:
            raise IndexError(f"Register index out of range: {reg}")

    # 0x02
    def add(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] += self.registers[reg2]
        self.registers[reg1] &= 0xFF

    # 0x05
    def sub(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] -= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    # 0x06
    def mul(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] *= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    # 0x07
    def div(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        if self.registers[reg2] == 0:
            raise Exception("Division by zero")
        self.registers[reg1] //= self.registers[reg2]
        self.registers[reg1] &= 0xFF

    # 0x08
    def vector_add(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg1 < len(self.vector_registers) and reg2 < len(self.vector_registers):
            self.vector_registers[reg1] += self.vector_registers[reg2]
        else:
            raise IndexError(f"Register index out of range: reg1={reg1}, reg2={reg2}")

    # 0x09
    def vector_sub(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg1 < len(self.vector_registers) and reg2 < len(self.vector_registers):
            self.vector_registers[reg1] -= self.vector_registers[reg2]
        else:
            raise IndexError(f"Register index out of range: reg1={reg1}, reg2={reg2}")

    # 0x0A
    def vector_dot(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] = np.dot(self.vector_registers[reg1], self.vector_registers[reg2])

    # 0x0B
    def vector_cross(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.vector_registers[reg1] = np.cross(self.vector_registers[reg1], self.vector_registers[reg2])

    # 0x0C
    def calculate_force(self):
        mass_reg = self.ram.read(self.pc)
        accel_reg = self.ram.read(self.pc + 1)
        force_reg = self.ram.read(self.pc + 2)
        self.pc += 3
        self.vector_registers[force_reg] = self.registers[mass_reg] * self.vector_registers[accel_reg]

    # 0x0D
    def calculate_velocity(self):
        init_vel_reg = self.ram.read(self.pc)
        accel_reg = self.ram.read(self.pc + 1)
        time_reg = self.ram.read(self.pc + 2)
        vel_reg = self.ram.read(self.pc + 3)
        self.pc += 4
        self.vector_registers[vel_reg] = self.vector_registers[init_vel_reg] + self.vector_registers[accel_reg] * self.registers[time_reg]

    # 0x0E
    def calculate_position(self):
        init_pos_reg = self.ram.read(self.pc)
        vel_reg = self.ram.read(self.pc + 1)
        time_reg = self.ram.read(self.pc + 2)
        pos_reg = self.ram.read(self.pc + 3)
        self.pc += 4
        self.vector_registers[pos_reg] = self.vector_registers[init_pos_reg] + self.vector_registers[vel_reg] * self.registers[time_reg]

    # 0x0F
    def jmp(self):
        addr = self.ram.read(self.pc)
        if addr < len(self.ram.memory):
            self.pc = addr
        else:
            raise Exception(f"Jump address out of bounds: {addr}")

    # 0x10
    def jz(self):
        addr = self.ram.read(self.pc)
        self.pc += 1
        if self.registers[0] == 0:
            if addr < len(self.ram.memory):
                self.pc = addr
            else:
                raise Exception(f"Jump address out of bounds: {addr}")

    # 0x11
    def jnz(self):
        addr = self.ram.read(self.pc)
        self.pc += 1
        if self.registers[0] != 0:
            if addr < len(self.ram.memory):
                self.pc = addr
            else:
                raise Exception(f"Jump address out of bounds: {addr}")

    # 0x12
    def push(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.stack.append(self.registers[reg])

    # 0x13
    def pop(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        if not self.stack:
            raise Exception("Stack underflow")
        self.registers[reg] = self.stack.pop()

    # 0x14
    def vector_normalize(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        if reg < len(self.vector_registers):
            norm = np.linalg.norm(self.vector_registers[reg])
            if norm == 0:
                print(f"Skipping normalization for zero vector in register {reg}")
                return
            self.vector_registers[reg] /= norm
        else:
            raise IndexError(f"Register index out of range: {reg}")

    # 0x15
    def vector_scale(self):
        reg = self.ram.read(self.pc)
        scale = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.vector_registers):
            self.vector_registers[reg] *= scale
        else:
            raise IndexError(f"Register index out of range: {reg}")

    # 0x16
    def vector_magnitude(self):
        reg = self.ram.read(self.pc)
        mag_reg = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.vector_registers) and mag_reg < len(self.registers):
            self.registers[mag_reg] = np.linalg.norm(self.vector_registers[reg])
        else:
            raise IndexError(f"Register index out of range: reg={reg}, mag_reg={mag_reg}")

    # 0x03
    def store(self):
        reg = self.ram.read(self.pc)
        addr = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            if addr < len(self.ram.memory):
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
            if addr < len(self.ram.memory):
                self.registers[reg] = self.ram.read(addr)
            else:
                raise Exception(f"Load address out of bounds: {addr}")
        else:
            raise IndexError(f"Register index out of range: {reg}")

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
            print(f"PC: {self.pc}, Registers: {self.registers}, Stack: {self.stack}, Vector Registers: {self.vector_registers}")

    def load_program(self, program):
        self.ram.load_program(program)

    def __repr__(self):
        return (
            f"PhysicsProcessingUnit(bits={self.bits})\n"
            f"Registers: {self.registers}\n"
            f"Vector Registers: {self.vector_registers}\n"
            f"RAM:\n{self.ram.memory}\n"
            f"Stack: {self.stack}\n"
        )

# Example Usage
if __name__ == "__main__":
    # Initialize the instruction file
    instr_file = instr()
    instr_file.load("programs/ppu/0.instr")

    # Parse the instruction file
    instr_file.parse()

    # Get the parsed instructions
    program = instr_file.get()

    # Initialize the processor
    processor = PPU(ram_size=64, bits=32, debug=True)

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

    # Print the parsed vector result
    print(f"Result: {processor.vector_registers[0]}")
    print(f"Execution time: {execution_time} seconds")
