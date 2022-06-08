class AverageMeter():
    def __init__(self):
        self.count = 0.0
        self.average = 0.0
        self.sum = 0.0
        
    def update(self, value, count):
        self.sum += value
        self.count += count
        self.average = self.sum / self.count