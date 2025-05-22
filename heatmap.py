import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from blackScholes import calculateCallPrice, calculatePutPrice

# Get user inputs for ranges
min_vol = float(input("Enter minimum volatility (e.g., 0.1): "))
max_vol = float(input("Enter maximum volatility (e.g., 0.4): "))
min_spot = float(input("Enter minimum spot price (e.g., 80): "))
max_spot = float(input("Enter maximum spot price (e.g., 120): "))

# Get user inputs for buy prices (optional)
call_buy_price = input("Enter call option buy price: ")
put_buy_price = input("Enter put option buy price: ")

# Convert buy prices to float if provided
call_buy_price = float(call_buy_price) if call_buy_price else None
put_buy_price = float(put_buy_price) if put_buy_price else None

# Define the ranges for spot prices and volatilities
spot_prices = np.linspace(min_spot, max_spot, 10)
volatilities = np.linspace(min_vol, max_vol, 10)

# Create matrices to store call and put prices/profits
call_values = np.zeros((10, 10))
put_values = np.zeros((10, 10))

# Calculate values for each combination
for i, vol in enumerate(volatilities):
    for j, spot in enumerate(spot_prices):
        call_price = calculateCallPrice(spot, 100, 1, 0.1, vol)
        put_price = calculatePutPrice(spot, 100, 1, 0.1, vol)
        
        # If buy price is provided, calculate profit/loss
        if call_buy_price is not None:
            call_values[i, j] = call_price - call_buy_price
        else:
            call_values[i, j] = call_price
            
        if put_buy_price is not None:
            put_values[i, j] = put_price - put_buy_price
        else:
            put_values[i, j] = put_price

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot call options heatmap
sns.heatmap(call_values, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn",  # Red for negative, Green for positive
            center=0 if call_buy_price is not None else None,
            xticklabels=[f"{price:.0f}" for price in spot_prices],
            yticklabels=[f"{vol:.2f}" for vol in volatilities],
            ax=ax1)
ax1.set_title("Call Options" + (" Profit/Loss" if call_buy_price is not None else " Prices"))
ax1.set_xlabel("Spot Price")
ax1.set_ylabel("Volatility")

# Plot put options heatmap
sns.heatmap(put_values, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn",  # Red for negative, Green for positive
            center=0 if put_buy_price is not None else None,
            xticklabels=[f"{price:.0f}" for price in spot_prices],
            yticklabels=[f"{vol:.2f}" for vol in volatilities],
            ax=ax2)
ax2.set_title("Put Options" + (" Profit/Loss" if put_buy_price is not None else " Prices"))
ax2.set_xlabel("Spot Price")
ax2.set_ylabel("Volatility")

plt.tight_layout()
plt.show()
