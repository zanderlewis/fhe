from helpers import instr, PU
from colorama import Fore
import time
import numpy as np
import sys


class MDPU(PU):
    def __init__(
        self, shape, ram_size, core_id=1, num_cores=1, bits=16, dtype=int, debug=False
    ):
        super().__init__(ram_size, num_cores, bits, debug)
        self.shape = shape
        self.dtype = dtype
        self.array = np.zeros(shape, dtype=dtype)
        self.core_id = core_id

    def load_program(self, program):
        for i, instruction in enumerate(program):
            if i < len(self.ram.memory):
                self.ram.write(i, instruction)
            else:
                print(f"{Fore.RED}Program size exceeds RAM capacity at index: {i}")
                sys.exit(1)

    def run(self, mode="sequential"):
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

    def __repr__(self):
        return (
            f"MDPU(shape={self.shape}, dtype={self.dtype})\n"
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

    # Initialize the processor with core_id
    processor = MDPU(shape=(3, 4, 5), ram_size=64, bits=32)

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
    print(f"{Fore.GREEN}Execution time: {execution_time:.24f} seconds")
