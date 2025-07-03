import math
import random

#RandomSearch


def random_search(alfa, max_iter, k, n):
    f_min = math.inf
    alpha_min = None
    for i in range(max_iter):
        alfa = random_alpha(k*n)
        x = gamma(alfa, k, n)




def gamma(alfa, k, n):
    x = [int(alfa[i:i+k], 2) for i in range(0, k*n, k)]
    return x

def random_alpha(size):
    return "".join([str(random.randint(0, 1)) for i in range(size)])

