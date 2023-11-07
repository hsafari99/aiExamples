import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
N = len(dataset)
d = len(dataset.columns)
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, i]
    if reward == 1:
        numbers_of_rewards_1[ad] += reward
    else:
        numbers_of_rewards_0[ad] += reward
    total_reward += reward

plt.hist(ads_selected)
plt.title('Histogram of ad selections')
plt.xlabel('Ad index')
plt.ylabel('Number of time each ad was selected')
plt.show()
