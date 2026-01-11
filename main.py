import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load historical prices from CSV
# Expected CSV format: columns 'Date' and 'Close' (or 'Adj Close')
csv_file = "NKE_historical_data.csv"
df = pd.read_csv(csv_file)

# Try to find the price column (common names)
price_col = next((col for col in ['close', 'Adj Close', 'Price'] if col in df.columns), None)
if price_col is None:
    raise ValueError("CSV must contain 'Close', 'Adj Close', or 'Price' column")

prices = df[price_col].values
S0 = prices[-1]  # Starting price (most recent)

# 2. Calculate daily returns and estimate parameters
returns = np.diff(prices) / prices[:-1]
mu = np.mean(returns)  # Average daily return (drift)
sigma = np.std(returns)  # Daily volatility

print(f"Historical Stats:")
print(f"  Starting Price: ${S0:.2f}")
print(f"  Daily Drift (mu): {mu:.6f}")
print(f"  Daily Volatility (sigma): {sigma:.6f}\n")

# 3. Set up Monte Carlo simulation parameters
N_PATHS = 1000  # Number of simulation paths
N_DAYS = 252    # Trading days to simulate (~1 year)

# Generate random price paths using geometric Brownian motion
# dS = mu*S*dt + sigma*S*dW
# S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
dt = 1  # Daily time step
drift = (mu - 0.5 * sigma**2) * dt
diffusion = sigma * np.sqrt(dt)

# Generate all random shocks at once
Z = np.random.standard_normal((N_DAYS, N_PATHS))
daily_returns = np.exp(drift + diffusion * Z)

# Calculate price paths
paths = np.zeros((N_DAYS + 1, N_PATHS))
paths[0] = S0
for t in range(1, N_DAYS + 1):
    paths[t] = paths[t-1] * daily_returns[t-1]

# 4. Plot simulated paths
colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
plt.figure(figsize=(12, 6))
for i in range(N_PATHS):
    color = colors[i % len(colors)]
    plt.plot(paths[:, i], alpha=0.3, color=color, linewidth=0.5)

plt.axhline(S0, color='black', linestyle='--', linewidth=2, label=f'Starting Price: ${S0:.2f}')
plt.xlabel('Trading Days')
plt.ylabel('Stock Price ($)')
plt.title('Monte Carlo Stock Price Simulation (Geometric Brownian Motion)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Calculate summary statistics for final prices
final_prices = paths[-1]
mean_final = np.mean(final_prices)
percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])

print(f"Final Price Statistics (Day {N_DAYS}):")
print(f"  Mean: ${mean_final:.2f}")
print(f"  Median (50th percentile): ${percentiles[2]:.2f}")
print(f"  5th percentile: ${percentiles[0]:.2f}")
print(f"  25th percentile: ${percentiles[1]:.2f}")
print(f"  75th percentile: ${percentiles[3]:.2f}")
print(f"  95th percentile: ${percentiles[4]:.2f}")
print(f"\nExpected Return: {((mean_final / S0) - 1) * 100:.2f}%")
