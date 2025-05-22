
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

def calculateOptionPrice(S, X, T, r, sigma, side="C"):
    if(size=="C"):
        d1 = (math.log(S/X) + (r + (sigma**2)/2)*T) / (sigma*T**0.5)
        d2 = d1 - sigma*T**0.5
        return S*norm.cdf(d1) - X*math.exp(-r*T)*norm.cdf(d2) 
    else if (size=="P"):
        d1 = (math.log(S/X) + (r + (sigma**2)/2)*T) / (sigma*T**0.5)
        d2 = d1 - sigma*T**0.5
        return X*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    else:
        print("Please enter either C for Call or P for Put")
        
callPrice = calculateCallPrice(assetPrice, strikePrice, timeToMaturity, riskFreeRate, volatility)
putPrice = calculatePutPrice(assetPrice, strikePrice, timeToMaturity, riskFreeRate, volatility)

#asduho asuhdouh asoudh aoush dus ofuhousa houh douah dosuhoasuh d
#Wijsahdu hasiuhd uashdiuhsaoudhiu hiuash diuhiu haiuh iuhai sudhiuah sd
# uashdiu hds aiui uashiudhiuash diuh i sadhi
#O sdauih diusha iusdh
# sauduhasdiuh asdu sauidh saiuhd iuahs diuhsa hdiu d d hdh dhhd  hdd 
# ushadiu h39h3h ush aiudh98289 dh9sah d9uh|
#OI8hs 8da h8d9ha 9fh9ahf98 ha89sfha89hf
