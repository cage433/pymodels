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
            # print(f"Direction for {line}")
            arr = list(map(int, p.split(line.strip())))
            a = arr[2]
            m = arr[3:]
            m.insert(0, 0)
            s = len(m) - 1
            # print(f"Primitive {arr[0]}, a {a}, m {np.array(m)}")
            direction = np.zeros((self.bits + 1,), int)

            for i in range(1, s + 1):
                direction[i] = (m[i] << (self.bits - i))

            for i in range(s + 1, self.bits + 1):
                direction[i] = direction[i - s] ^ (direction[i - s] >> s)
                for k in range(1, s):
                    x = a >> (s - 1 - k)
                    y = x & 1
                    z = y * direction[i - k]
                    # print(f"i {i}, k {k}, x {x}, y {y}, z {z}")

                    direction[i] = direction[i] ^ z  # (((a >> (s - 1 - k)) & 1) * direction[i - k])

            # print(f"Direction form {line} is {np.array(direction)}")
            return list(direction)

        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(f"{dir_path}/resources/new-joe-kuo-7.21201", "r") as f:
            self.directions = [build_direction(x) for x in f.readlines()[1:n_variables]]

        one_direction = [1 << (self.bits - i) for i in range(1, self.bits + 1)]
        one_direction.insert(0, 0)

        self.directions.insert(0, one_direction)

    def generate(self, n_paths):

        arr = np.zeros((n_paths, self.n_variables), float)
        x = np.zeros((self.n_variables,), int)
        count = 1
        for i_path in range(n_paths):
            c = 1
            value = count - 1
            while (value & 1) == 1:
                value = value >> 1
                c += 1

            for i_var in range(self.n_variables):
                x[i_var] = x[i_var] ^ self.directions[i_var][c]
                arr[i_path][i_var] = x[i_var] * self.scale

            count += 1

        return np.transpose(arr)
