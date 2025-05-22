from flask import Flask, render_template, request, jsonify, send_from_directory
from blackScholes import calculateCallPrice, calculatePutPrice
from monteCarlo import MonteCarloOptionPricer
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    model = data.get('model', 'blackScholes')
    
    # Extract common parameters
    S0 = float(data['assetPrice'])
    K = float(data['strikePrice'])
    T = float(data['timeToMaturity'])
    r = float(data['riskFreeRate'])
    sigma = float(data['volatility'])
    
    if model == 'blackScholes':
        # Calculate Black-Scholes prices
        call_price = calculateCallPrice(S0, K, T, r, sigma)
        put_price = calculatePutPrice(S0, K, T, r, sigma)
        
        return jsonify({
            'callPrice': call_price,
            'putPrice': put_price
        })
    else:  # Monte Carlo
        # Extract Monte Carlo specific parameters
        n_simulations = int(data.get('nSimulations', 100000))
        
        # Create Monte Carlo pricer instance
        pricer = MonteCarloOptionPricer(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            n_simulations=n_simulations
        )
        
        # Calculate European option prices
        european_call = pricer.price_european_call()
        european_put = pricer.price_european_put()
        
        # Calculate American option prices
        american_call = pricer.price_american_call()
        american_put = pricer.price_american_put()
        
        # Calculate Greeks for European call
        greeks = pricer.calculate_greeks(option_type='call', style='european')
        
        # Use 95% confidence level (z-score = 1.96)
        confidence_level = 1.96
        european_call['conf_interval'] = confidence_level * european_call['std_error']
        european_put['conf_interval'] = confidence_level * european_put['std_error']
        american_call['conf_interval'] = confidence_level * american_call['std_error']
        american_put['conf_interval'] = confidence_level * american_put['std_error']
        
        return jsonify({
            'european': {
                'call': {
                    'price': european_call['price'],
                    'std_error': european_call['std_error'],
                    'conf_interval': european_call['conf_interval'],
                    'computation_time': european_call['computation_time']
                },
                'put': {
                    'price': european_put['price'],
                    'std_error': european_put['std_error'],
                    'conf_interval': european_put['conf_interval'],
                    'computation_time': european_put['computation_time']
                }
            },
            'american': {
                'call': {
                    'price': american_call['price'],
                    'std_error': american_call['std_error'],
                    'conf_interval': american_call['conf_interval'],
                    'computation_time': american_call['computation_time']
                },
                'put': {
                    'price': american_put['price'],
                    'std_error': american_put['std_error'],
                    'conf_interval': american_put['conf_interval'],
                    'computation_time': american_put['computation_time']
                }
            },
            'greeks': greeks
        })

@app.route('/heatmap-data', methods=['POST'])
def heatmap_data():
    data = request.get_json()
    min_vol = float(data['minVol'])
    max_vol = float(data['maxVol'])
    min_spot = float(data['minSpot'])
    max_spot = float(data['maxSpot'])
    call_buy_price = data.get('callBuyPrice')
    put_buy_price = data.get('putBuyPrice')
    call_buy_price = float(call_buy_price) if call_buy_price not in (None, "") else None
    put_buy_price = float(put_buy_price) if put_buy_price not in (None, "") else None

    # Use fixed strike, T, r for now (could be extended to be dynamic)
    strike = 100
    T = 1
    r = 0.1

    spot_prices = np.linspace(min_spot, max_spot, 10)
    volatilities = np.linspace(min_vol, max_vol, 10)
    call_values = np.zeros((10, 10))
    put_values = np.zeros((10, 10))

    for i, vol in enumerate(volatilities):
        for j, spot in enumerate(spot_prices):
            call_price = calculateCallPrice(spot, strike, T, r, vol)
            put_price = calculatePutPrice(spot, strike, T, r, vol)
            if call_buy_price is not None:
                call_values[i, j] = call_price - call_buy_price
            else:
                call_values[i, j] = call_price
            if put_buy_price is not None:
                put_values[i, j] = put_price - put_buy_price
            else:
                put_values[i, j] = put_price

    return jsonify({
        'call': call_values.tolist(),
        'put': put_values.tolist(),
        'spotLabels': [f"{x:.0f}" for x in spot_prices],
        'volLabels': [f"{x:.2f}" for x in volatilities]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 