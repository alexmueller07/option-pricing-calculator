# Option Pricing Calculator

A web application for calculating option prices using both Black-Scholes and Monte Carlo simulation methods. This tool is designed for anyone interested in quantitative finance.

## Live Demo

[Try the live demo here](https://option-pricing-calculator.onrender.com/)
Note: Will take a little bit to boot up as I am running it on a free hosting service. Once it boots up it will run fast. 

## Recordings Showing all Important Parts of Project Running

[Add screenshots or GIF of the application in action here]

## Features

- **Multiple Pricing Models**:
  - Black-Scholes model for European options
  - Monte Carlo simulation for both European and American options
- **Greeks Calculation**
- **Interactive Heatmap Visualization With Profit and Loss**
- **Confidence Intervals for Monte Carlo Results**
- **Computation Time Tracking for Monte Carlo Simulation**

## Usage

1. Select your preferred pricing model (Black-Scholes or Monte Carlo)
2. Enter the required parameters:
   - Asset Price (S₀)
   - Strike Price (K)
   - Time to Maturity (T)
   - Risk-Free Rate (r)
   - Volatility (σ)
   - Number of Simulations (for Monte Carlo)
3. It will automatically calculate and display the answers on the right side of the screen.
4. If you are using Black-Scholes model use the heatmap feature to visualize price variations and look at your P/L based on buy price.
5. If you are using Monte Carlo look at the greeks at the bottom and look at the various bits of information it gives.
   
## Technical Details

- This is a purely python program. HTML was only used for creating the webpage to display what the python backend calculated 
- Built with Flask
- Uses NumPy for numerical computations
- Implements both Black-Scholes and Monte Carlo pricing methods
- Features a responsive web interface using HTML and CSS
  
## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/option-pricing-calculator.git
cd option-pricing-calculator
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Contributing

Feel free to submit issues and enhancement requests! Also feel free to email me at amueller.code@gmail.com with anything.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Alexander Mueller

- GitHub: [alexmueller07](https://github.com/alexmueller07)
- LinkedIn: [Alexander Mueller](https://www.linkedin.com/in/alexander-mueller-021658307/)
- Email: amueller.code@gmail.com
