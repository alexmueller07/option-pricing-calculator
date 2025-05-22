# Option Pricing Calculator

A sophisticated web application for calculating option prices using both Black-Scholes and Monte Carlo simulation methods. This tool is designed for financial analysts, traders, and students interested in quantitative finance.

## Features

- **Multiple Pricing Models**:
  - Black-Scholes model for European options
  - Monte Carlo simulation for both European and American options
- **Comprehensive Greeks Calculation**
- **Interactive Heatmap Visualization**
- **Real-time Price Updates**
- **Confidence Intervals for Monte Carlo Results**
- **Computation Time Tracking**

## Live Demo

[Try the live demo here](https://your-deployment-url.com)

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

## Usage

1. Select your preferred pricing model (Black-Scholes or Monte Carlo)
2. Enter the required parameters:
   - Asset Price (S₀)
   - Strike Price (K)
   - Time to Maturity (T)
   - Risk-Free Rate (r)
   - Volatility (σ)
   - Number of Simulations (for Monte Carlo)
3. Click "Calculate" to get the option prices
4. Use the heatmap feature to visualize price variations

## Screenshots

[Add screenshots or GIF of the application in action here]

## Technical Details

- Built with Flask
- Uses NumPy for numerical computations
- Implements both Black-Scholes and Monte Carlo pricing methods
- Features a responsive web interface

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Alexander Mueller

- GitHub: [alexmueller07](https://github.com/alexmueller07)
- LinkedIn: [Alexander Mueller](https://www.linkedin.com/in/alexander-mueller-021658307/)
- Email: amueller.code@gmail.com
