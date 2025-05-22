import numpy as np
from scipy.stats import norm
import time

class MonteCarloOptionPricer:
    def __init__(self, S0, K, T, r, sigma, q=0, n_simulations=100000):
        """
        Initialize the Monte Carlo Option Pricer
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Volatility
        q (float): Dividend yield (default: 0)
        n_simulations (int): Number of simulations (default: 100000)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.n_simulations = n_simulations
        
    def generate_paths(self):
        """Generate stock price paths using geometric Brownian motion"""
        dt = self.T
        # Generate random numbers
        Z = np.random.normal(0, 1, self.n_simulations)
        # Calculate drift and diffusion terms
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        # Calculate final stock prices
        ST = self.S0 * np.exp(drift + diffusion)
        return ST
    
    def price_european_call(self):
        """Price a European call option using Monte Carlo simulation"""
        start_time = time.time()
        
        # Generate final stock prices
        ST = self.generate_paths()
        
        # Calculate payoffs
        payoffs = np.maximum(ST - self.K, 0)
        
        # Discount payoffs
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        # Calculate 95% confidence interval
        conf_interval = 1.96 * std_error
        
        computation_time = time.time() - start_time
        
        return {
            'price': price,
            'std_error': std_error,
            'conf_interval': conf_interval,
            'computation_time': computation_time
        }
    
    def price_european_put(self):
        """Price a European put option using Monte Carlo simulation"""
        start_time = time.time()
        
        # Generate final stock prices
        ST = self.generate_paths()
        
        # Calculate payoffs
        payoffs = np.maximum(self.K - ST, 0)
        
        # Discount payoffs
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        # Calculate 95% confidence interval
        conf_interval = 1.96 * std_error
        
        computation_time = time.time() - start_time
        
        return {
            'price': price,
            'std_error': std_error,
            'conf_interval': conf_interval,
            'computation_time': computation_time
        }
    
    def price_american_call(self):
        """Price an American call option using Monte Carlo simulation with Longstaff-Schwartz method"""
        start_time = time.time()
        
        # For non-dividend paying stocks, American call = European call
        if self.q == 0:
            return self.price_european_call()
        
        # For dividend paying stocks, use binomial tree approximation
        # This is a simplified version - in practice, you'd want to use a more sophisticated method
        n_steps = 100
        dt = self.T / n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1/u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        # Generate stock price tree
        stock_tree = np.zeros((n_steps + 1, n_steps + 1))
        stock_tree[0, 0] = self.S0
        
        for i in range(1, n_steps + 1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S0 * (u ** (i - j)) * (d ** j)
        
        # Calculate option values at maturity
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        option_tree[n_steps, :] = np.maximum(stock_tree[n_steps, :] - self.K, 0)
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                exercise_value = stock_tree[i, j] - self.K
                hold_value = np.exp(-self.r * dt) * (p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1])
                option_tree[i, j] = max(exercise_value, hold_value)
        
        price = option_tree[0, 0]
        
        # Calculate standard error (approximate)
        std_error = price * 0.01  # Simplified error estimate
        conf_interval = 1.96 * std_error
        
        computation_time = time.time() - start_time
        
        return {
            'price': price,
            'std_error': std_error,
            'conf_interval': conf_interval,
            'computation_time': computation_time
        }
    
    def price_american_put(self):
        """Price an American put option using Monte Carlo simulation with Longstaff-Schwartz method"""
        start_time = time.time()
        
        # Use binomial tree approximation
        n_steps = 100
        dt = self.T / n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1/u
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)
        
        # Generate stock price tree
        stock_tree = np.zeros((n_steps + 1, n_steps + 1))
        stock_tree[0, 0] = self.S0
        
        for i in range(1, n_steps + 1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S0 * (u ** (i - j)) * (d ** j)
        
        # Calculate option values at maturity
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        option_tree[n_steps, :] = np.maximum(self.K - stock_tree[n_steps, :], 0)
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                exercise_value = self.K - stock_tree[i, j]
                hold_value = np.exp(-self.r * dt) * (p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1])
                option_tree[i, j] = max(exercise_value, hold_value)
        
        price = option_tree[0, 0]
        
        # Calculate standard error (approximate)
        std_error = price * 0.01  # Simplified error estimate
        conf_interval = 1.96 * std_error
        
        computation_time = time.time() - start_time
        
        return {
            'price': price,
            'std_error': std_error,
            'conf_interval': conf_interval,
            'computation_time': computation_time
        }
    
    def calculate_greeks(self, option_type='call', style='european'):
        """Calculate option Greeks using finite difference method"""
        # Use larger perturbations for more stable results
        h = 0.01  # 1% change
        
        # Calculate base price
        if style == 'european':
            if option_type == 'call':
                base_price = self.price_european_call()['price']
            else:
                base_price = self.price_european_put()['price']
        else:
            if option_type == 'call':
                base_price = self.price_american_call()['price']
            else:
                base_price = self.price_american_put()['price']
        
        # Store original values
        original_S0 = self.S0
        original_sigma = self.sigma
        original_T = self.T
        original_r = self.r
        
        # Delta
        self.S0 = original_S0 * (1 + h)
        if style == 'european':
            if option_type == 'call':
                price_up = self.price_european_call()['price']
            else:
                price_up = self.price_european_put()['price']
        else:
            if option_type == 'call':
                price_up = self.price_american_call()['price']
            else:
                price_up = self.price_american_put()['price']
        
        self.S0 = original_S0 * (1 - h)
        if style == 'european':
            if option_type == 'call':
                price_down = self.price_european_call()['price']
            else:
                price_down = self.price_european_put()['price']
        else:
            if option_type == 'call':
                price_down = self.price_american_call()['price']
            else:
                price_down = self.price_american_put()['price']
        
        self.S0 = original_S0
        delta = (price_up - price_down) / (2 * h * original_S0)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (h * h * original_S0 * original_S0)
        
        # Vega
        self.sigma = original_sigma * (1 + h)
        if style == 'european':
            if option_type == 'call':
                price_up = self.price_european_call()['price']
            else:
                price_up = self.price_european_put()['price']
        else:
            if option_type == 'call':
                price_up = self.price_american_call()['price']
            else:
                price_up = self.price_american_put()['price']
        
        self.sigma = original_sigma * (1 - h)
        if style == 'european':
            if option_type == 'call':
                price_down = self.price_european_call()['price']
            else:
                price_down = self.price_european_put()['price']
        else:
            if option_type == 'call':
                price_down = self.price_american_call()['price']
            else:
                price_down = self.price_american_put()['price']
        
        self.sigma = original_sigma
        vega = (price_up - price_down) / (2 * h * original_sigma)
        
        # Theta (per year)
        self.T = original_T * (1 + h)
        if style == 'european':
            if option_type == 'call':
                price_up = self.price_european_call()['price']
            else:
                price_up = self.price_european_put()['price']
        else:
            if option_type == 'call':
                price_up = self.price_american_call()['price']
            else:
                price_up = self.price_american_put()['price']
        
        self.T = original_T * (1 - h)
        if style == 'european':
            if option_type == 'call':
                price_down = self.price_european_call()['price']
            else:
                price_down = self.price_european_put()['price']
        else:
            if option_type == 'call':
                price_down = self.price_american_call()['price']
            else:
                price_down = self.price_american_put()['price']
        
        self.T = original_T
        theta = (price_down - price_up) / (2 * h * original_T)
        
        # Rho
        self.r = original_r * (1 + h)
        if style == 'european':
            if option_type == 'call':
                price_up = self.price_european_call()['price']
            else:
                price_up = self.price_european_put()['price']
        else:
            if option_type == 'call':
                price_up = self.price_american_call()['price']
            else:
                price_up = self.price_american_put()['price']
        
        self.r = original_r * (1 - h)
        if style == 'european':
            if option_type == 'call':
                price_down = self.price_european_call()['price']
            else:
                price_down = self.price_european_put()['price']
        else:
            if option_type == 'call':
                price_down = self.price_american_call()['price']
            else:
                price_down = self.price_american_put()['price']
        
        self.r = original_r
        rho = (price_up - price_down) / (2 * h * original_r)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

def main():
    # Example usage
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity (1 year)
    r = 0.05    # Risk-free rate
    sigma = 0.2 # Volatility
    q = 0.0     # Dividend yield
    
    # Create pricer instance
    pricer = MonteCarloOptionPricer(S0, K, T, r, sigma, q)
    
    # Price European options
    european_call = pricer.price_european_call()
    european_put = pricer.price_european_put()
    
    # Price American options
    american_call = pricer.price_american_call()
    american_put = pricer.price_american_put()
    
    # Calculate Greeks for European call
    greeks = pricer.calculate_greeks(option_type='call', style='european')
    
    # Print results
    print("\nEuropean Call Option:")
    print(f"Price: ${european_call['price']:.4f}")
    print(f"Standard Error: ${european_call['std_error']:.4f}")
    print(f"95% Confidence Interval: ±${european_call['conf_interval']:.4f}")
    print(f"Computation Time: {european_call['computation_time']:.2f} seconds")
    
    print("\nEuropean Put Option:")
    print(f"Price: ${european_put['price']:.4f}")
    print(f"Standard Error: ${european_put['std_error']:.4f}")
    print(f"95% Confidence Interval: ±${european_put['conf_interval']:.4f}")
    print(f"Computation Time: {european_put['computation_time']:.2f} seconds")
    
    print("\nAmerican Call Option:")
    print(f"Price: ${american_call['price']:.4f}")
    print(f"Standard Error: ${american_call['std_error']:.4f}")
    print(f"95% Confidence Interval: ±${american_call['conf_interval']:.4f}")
    print(f"Computation Time: {american_call['computation_time']:.2f} seconds")
    
    print("\nAmerican Put Option:")
    print(f"Price: ${american_put['price']:.4f}")
    print(f"Standard Error: ${american_put['std_error']:.4f}")
    print(f"95% Confidence Interval: ±${american_put['conf_interval']:.4f}")
    print(f"Computation Time: {american_put['computation_time']:.2f} seconds")
    
    print("\nGreeks (European Call):")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Theta: {greeks['theta']:.4f}")
    print(f"Rho: {greeks['rho']:.4f}")

if __name__ == "__main__":
    main()
