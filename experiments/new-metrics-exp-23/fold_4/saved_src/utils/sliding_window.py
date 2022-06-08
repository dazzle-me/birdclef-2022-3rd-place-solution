from abc import abstractmethod


class SlidingWindow():
    def __init__(self, size=10):
        self.list = []
        self.size = 10

    def update(self, value):
        if len(self.list) == self.size:
            self.list.pop(0)
        self.list.append(value)
    
    @abstractmethod
    def compute(self):
        pass

class AverageWindow(SlidingWindow):
    def __init__(self, size=10):
        super().__init__(size)
    
    def compute(self):
        return sum(self.list) / len(self.list)