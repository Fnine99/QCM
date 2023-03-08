from scipy.stats import qmc
import numpy as np

class QuasiRandomSequence:
    def __init__(self, data) -> None:
        """ 
        initialize the Quasi Monte-Carlo class
        with the sequence that will be randomized
        """
        self.seed = np.array(data)
        self.size = len(self.seed)
        self.count = 0
        self.sobol_matrix = self.get_sobol_matrix()

    def get_sobol_matrix(self):
        max_bits = np.ceil(np.log2(self.n))
        matrix = np.zeros((self.n, max_bits), dtype=np.uint8)
        matrix[:, 0] = 1
        for i in range(1, max_bits):
            for j in range(self.n):
                matrix[j, i] = matrix[j, i-1] ^ (matrix[j, i-1] >> i-1)
        return matrix

    def generate_sequence(self, iterations):
        quasi_random_sequence = np.zeros((iterations, self.n))
        for i in range(iterations):
            direction = np.zeros(self.n)
            for j in range(self.count.bit_length()):
                direction ^= self.sobol_matrix[:, j] * ((self.count >> j) & 1)
            quasi_random_sequence[i, :] = self.seed + direction
            self.count += 1
        return quasi_random_sequence

if __name__ == '__main__':
    data  = [5.0,6.0,3.0,4.0,5.0,5.0,3.0,2.0,5.0,5.0,2.0,0.4,5.0,]
    test = QMC(data)
    print(test.generate_sequence(5))
