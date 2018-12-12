import os
import re
import numpy as np


class SobolGenerator:
    def __init__(self, n_variables):

        self.n_variables = n_variables
        self.bits = 52
        self.scale = 1.0 / np.power(2.0, self.bits)

        p = re.compile(" +")

        def build_direction(line):
            arr = list(map(int, p.split(line.strip())))
            a = arr[2]
            m = arr[3:]
            m.insert(0, 0)
            s = len(m) - 1
            direction = np.zeros((self.bits + 1,), int)

            for i in range(1, s + 1):
                direction[i] = (m[i] << (self.bits - i))

            for i in range(s + 1, self.bits + 1):
                direction[i] = direction[i - s] ^ (direction[i - s] >> s)
                for k in range(1, s):
                    x = a >> (s - 1 - k)
                    y = x & 1
                    z = y * direction[i - k]

                    direction[i] = direction[i] ^ z  # (((a >> (s - 1 - k)) & 1) * direction[i - k])

            return direction

        self.directions = np.zeros((n_variables, self.bits + 1), int)
        with open(f"{os.path.dirname(os.path.realpath(__file__))}/resources/new-joe-kuo-7.21201", "r") as f:
            head = [next(f) for x in range(n_variables)]
            for i, line in enumerate(head[1:]):
                self.directions[i + 1] = build_direction(line)

        one_direction = [1 << (self.bits - i) for i in range(1, self.bits + 1)]
        one_direction.insert(0, 0)

        self.directions[0] = np.array(one_direction)

    def generate(self, n_paths):

        arr = np.zeros((n_paths, self.n_variables), float)
        x = np.zeros((self.n_variables,), int)
        count = 1
        var_range = range(self.n_variables)
        for i_path in range(n_paths):
            c = 1
            value = count - 1
            while (value & 1) == 1:
                value >>= 1
                c += 1

            for i_var in var_range:
                x[i_var] ^= self.directions[i_var, c]
                arr[i_path, i_var] = x[i_var] * self.scale

            count += 1

        return np.transpose(arr)
