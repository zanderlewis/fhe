import numpy as np
import time, sys, colorama
from colorama import Fore
from helpers import instr, PU

colorama.init(autoreset=True)


class PPU(PU):
    def __init__(self, ram_size, num_cores=1, bits=16, debug=False):
        super().__init__(ram_size, num_cores, bits, debug)
        self.vector_registers = [np.zeros(3) for _ in range(bits)]  # 3D vectors

        # Additional PPU-specific instructions
        self.instructions.update(
            {
                0x08: self.vector_add,
                0x09: self.vector_sub,
                0x0A: self.vector_dot,
                0x0B: self.vector_cross,
                0x0C: self.calculate_force,
                0x0D: self.calculate_velocity,
                0x0E: self.calculate_position,
                0x14: self.vector_normalize,
                0x15: self.vector_scale,
                0x16: self.vector_magnitude,
            }
        )

    def load_program(self, program):
        for i, instruction in enumerate(program):
            if i < len(self.ram.memory):
                self.ram.write(i, instruction)
            else:
                print(f"{Fore.RED}Program size exceeds RAM capacity at index: {i}")
                sys.exit(1)

    def run(self):
        self.pc = 0
        self.running = True
        while self.running:
            opcode = self.ram.read(self.pc)
            self.pc += 1
            if opcode in self.instructions:
                self.instructions[opcode]()
            else:
                print(f"{Fore.RED}Unknown opcode {opcode} at PC: {self.pc - 1}")
                sys.exit(1)

    # 0x08
    def vector_add(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg1 < len(self.vector_registers) and reg2 < len(self.vector_registers):
            self.vector_registers[reg1] += self.vector_registers[reg2]
        else:
            print(f"{Fore.RED}Register index out of range: reg1={reg1}, reg2={reg2}")
            sys.exit(1)

    # 0x09
    def vector_sub(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg1 < len(self.vector_registers) and reg2 < len(self.vector_registers):
            self.vector_registers[reg1] -= self.vector_registers[reg2]
        else:
            print(f"{Fore.RED}Register index out of range: reg1={reg1}, reg2={reg2}")
            sys.exit(1)

    # 0x0A
    def vector_dot(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] = np.dot(
            self.vector_registers[reg1], self.vector_registers[reg2]
        )

    # 0x0B
    def vector_cross(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.vector_registers[reg1] = np.cross(
            self.vector_registers[reg1], self.vector_registers[reg2]
        )

    # 0x0C
    def calculate_force(self):
        mass_reg = self.ram.read(self.pc)
        accel_reg = self.ram.read(self.pc + 1)
        force_reg = self.ram.read(self.pc + 2)
        self.pc += 3
        self.vector_registers[force_reg] = (
            self.registers[mass_reg] * self.vector_registers[accel_reg]
        )

    # 0x0D
    def calculate_velocity(self):
        init_vel_reg = self.ram.read(self.pc)
        accel_reg = self.ram.read(self.pc + 1)
        time_reg = self.ram.read(self.pc + 2)
        vel_reg = self.ram.read(self.pc + 3)
        self.pc += 4
        self.vector_registers[vel_reg] = (
            self.vector_registers[init_vel_reg]
            + self.vector_registers[accel_reg] * self.registers[time_reg]
        )

    # 0x0E
    def calculate_position(self):
        init_pos_reg = self.ram.read(self.pc)
        vel_reg = self.ram.read(self.pc + 1)
        time_reg = self.ram.read(self.pc + 2)
        pos_reg = self.ram.read(self.pc + 3)
        self.pc += 4
        self.vector_registers[pos_reg] = (
            self.vector_registers[init_pos_reg]
            + self.vector_registers[vel_reg] * self.registers[time_reg]
        )

    # 0x14
    def vector_normalize(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        if reg < len(self.vector_registers):
            norm = np.linalg.norm(self.vector_registers[reg])
            if norm == 0:
                print(
                    f"{Fore.YELLOW}Skipping normalization for zero vector in register {reg}"
                )
                return
            self.vector_registers[reg] /= norm
        else:
            print(f"{Fore.RED}Register index out of range: {reg}")
            sys.exit(1)

    # 0x15
    def vector_scale(self):
        reg = self.ram.read(self.pc)
        scale = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.vector_registers):
            self.vector_registers[reg] *= scale
        else:
            print(f"{Fore.RED}Register index out of range: {reg}")
            sys.exit(1)

    # 0x16
    def vector_magnitude(self):
        reg = self.ram.read(self.pc)
        mag_reg = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.vector_registers) and mag_reg < len(self.registers):
            self.registers[mag_reg] = np.linalg.norm(self.vector_registers[reg])
        else:
            print(
                f"{Fore.RED}Register index out of range: reg={reg}, mag_reg={mag_reg}"
            )
            sys.exit(1)

    def __repr__(self):
        return (
            f"PPU(bits={self.bits})\n"
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
    processor = PPU(ram_size=256, num_cores=1, bits=16, debug=False)

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
    print(f"{Fore.GREEN}Execution time: {execution_time} seconds")
