import matplotlib.pyplot as plt

class plot():
    def __init__(self, epochs, period):
        self.epochs = epochs
        self.period = period
        self.rewards = []
        self.data = {'epoch': [], 'min': [], 'max': [],'avg': []}

    def update(self, current_epoch, rewards):
        self.rewards.append(rewards)

        if((current_epoch+1) % self.period == 0):
            self.data['epoch'].append(current_epoch)
            self.data['min'].append(min(self.rewards))
            self.data['max'].append(max(self.rewards))
            self.data['avg'].append(sum(self.rewards)/len(self.rewards))
            self.rewards = []

    def show(self):
        plt.plot(self.data['epoch'], self.data['max'], label='max')
        plt.plot(self.data['epoch'], self.data['avg'], label='avg')
        plt.plot(self.data['epoch'], self.data['min'], label='min')
        plt.legend(loc=2)

        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Rewards per Epoch')
        plt.grid(True)

        plt.show()
