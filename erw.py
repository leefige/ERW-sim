import numpy as np

class ERW(object):
    def __init__(self, p, q, M=None):
        self.p = p
        self.q = q
        if M is not None:
            assert M > 0
        self.M = M
        self.pool = []

    def reset(self):
        self.pool.clear()

    def steps(self):
        return len(self.pool)

    def where(self):
        return self.get_S(self.steps())

    def get_X(self, t):
        if t == 0:
            return 0
        elif t > 0 and t <= self.steps():
            return self.pool[t-1]
        elif t < 0:
            return RuntimeError("Error: step cannot be negative")
        else:
            raise RuntimeError("Error: step too large")

    def get_S(self, t):
        if t == 0:
            return 0
        elif t > 0 and t <= self.steps():
            return np.sum(self.pool[:t])
        elif t < 0:
            return RuntimeError("Error: step cannot be negative")
        else:
            raise RuntimeError("Error: step too large")

    def walk(self):
        if self.steps() == 0:
            X = 1 if np.random.rand() < self.q else -1
        else:
            if self.M is not None and self.steps() >= self.M:
                t = np.random.randint(0, self.M)
            else:
                t = np.random.randint(0, self.steps())
            X = self.pool[t] if np.random.rand() < self.p else -self.pool[t]
        self.pool.append(X)
