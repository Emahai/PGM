import math
from math import comb
import mpmath as mp

# parameters
n, p = 1000, 1/6
mu = n*p
sigma = math.sqrt(n*p*(1-p))

# (a) normal approx with continuity correction
z1 = (149.5 - mu)/sigma
z2 = (200.5 - mu)/sigma
approx = 0.5*(1+math.erf(z2/math.sqrt(2))) - 0.5*(1+math.erf(z1/math.sqrt(2)))

# (b) exact binomial for Y<150
n, p = 1000, mp.mpf(1)/6

exact_mp = mp.nsum(lambda k: mp.binomial(n, int(k)) * p**k * (1-p)**(n-k), [0, 149])
print("Exact P(Y<150) (mpmath) =", exact_mp)

print("Normal approx P(150<=X<=200)=", approx)
print("Exact P(Y<150)=", exact_mp)