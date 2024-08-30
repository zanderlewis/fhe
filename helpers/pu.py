import numpy as np
import time, sys, colorama
from colorama import Fore, Style
from helpers.ram import RAM
from helpers.core import Core

colorama.init(autoreset=True)

class PU:
    def __init__(self, ram_size, num_cores=1, bits=16, debug=False):
        self.ram = RAM(size=ram_size)
        self.registers = [0] * bits
        self.pc = 0
        self.stack = []
        self.running = True
        self.debug_mode = debug
        self.bits = bits
        self.cores = [Core(i, self.ram) for i in range(num_cores)]

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

    def load_immediate(self):
        reg = self.ram.read(self.pc)
        value = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg] = value

    def add(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1] += self.registers[reg2]
        self.registers[reg1] &= 0xFF

    def store(self):
        reg = self.ram.read(self.pc)
        addr = self.ram.read(self.pc + 1)
        self.pc += 2
        self.ram.write(addr, self.registers[reg])

    def load(self):
        addr = self.ram.read(self.pc)
        reg = self.ram.read(self.pc + 1)
        self.pc += 2
        if reg < len(self.registers):
            self.registers[reg] = self.ram.read(addr)
        else:
            print(f"{Fore.RED}Register index out of range: {reg}")
            sys.exit(1)

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
        if self.registers[reg2] != 0:
            self.registers[reg1] //= self.registers[reg2]
        else:
            print(f"{Fore.RED}Division by zero error at PC: {self.pc - 2}")
            sys.exit(1)
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
        self.pc = addr

    def jz(self):
        addr = self.ram.read(self.pc)
        self.pc += 1
        if self.registers[0] == 0:
            self.pc = addr

    def jnz(self):
        addr = self.ram.read(self.pc)
        self.pc += 1
        if self.registers[0] != 0:
            self.pc = addr

    def push(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.stack.append(self.registers[reg])

    def pop(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.registers[reg] = self.stack.pop()

    def const(self):
        reg = self.ram.read(self.pc)
        value = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg] = value

    def swap(self):
        reg1 = self.ram.read(self.pc)
        reg2 = self.ram.read(self.pc + 1)
        self.pc += 2
        self.registers[reg1], self.registers[reg2] = self.registers[reg2], self.registers[reg1]

    def inc(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.registers[reg] += 1
        self.registers[reg] &= 0xFF

    def dec(self):
        reg = self.ram.read(self.pc)
        self.pc += 1
        self.registers[reg] -= 1
        self.registers[reg] &= 0xFF

    def halt(self):
        self.running = False