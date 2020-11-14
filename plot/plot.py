import matplotlib.pyplot as plt

class plot():
    def __init__(self, epochs, period):
        self.epochs = epochs
        self.period = period
        self.data = {'epoch': [], 'steps': []}

    def update(self, current_epoch, steps):
        if(current_epoch % self.period == 0):
            self.data['epoch'].append(current_epoch)
            self.data['steps'].append(stepds)

    def show(self):
        plt.plot(self.data['epoch'], self.data['steps'])
        plt.show()
