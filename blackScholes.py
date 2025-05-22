import math
from scipy.stats import norm

assetPrice = 100.00;
strikePrice = 100.00; 
timeToMaturity = 1; 
volatility = 0.25;
riskFreeRate = 0.1; 

def calculateCallPrice(S, X, T, r, sigma):
    d1 = (math.log(S/X) + (r + (sigma**2)/2)*T) / (sigma*T**0.5)
    d2 = d1 - sigma*T**0.5
    return S*norm.cdf(d1) - X*math.exp(-r*T)*norm.cdf(d2) 

def calculatePutPrice(S, X, T, r, sigma):
    d1 = (math.log(S/X) + (r + (sigma**2)/2)*T) / (sigma*T**0.5)
    d2 = d1 - sigma*T**0.5
    return X*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

callPrice = calculateCallPrice(assetPrice, strikePrice, timeToMaturity, riskFreeRate, volatility)
putPrice = calculatePutPrice(assetPrice, strikePrice, timeToMaturity, riskFreeRate, volatility)