class _Counter:
    def __init__(self):
        self.n = 0
    def add(self, k: int): 
        self.n += int(k)
    def value(self): 
        return self.n
    def reset(self):
        self.n = 0

token_counter = _Counter()
