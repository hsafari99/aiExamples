import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, len(dataset)):
    # 20 is max of purchased products (hard coded to make ;line lighter)
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

from apyori import apriori
# 0.003 is coming from 3 *7/ 7501 (3 purchases of a products in a week(7) divided by total no. of transactions(7501))
#  0.2 and 3 are best practice (kind of)
# 2 stands for number of products
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

results = list(rules)
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs,supports))

results_in_dataframe = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])
print(results_in_dataframe.sort_values(by='Support', ascending=0))
