# HW1b
# 1. Construct same monthly return series 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the Excel file
df = pd.read_excel('HSI.xlsx', index_col=0)

# Interpolate missing entries
df_interpolated = df.interpolate(method='spline', order=3)

# Create monthly DataFrame and calculate monthly returns
monthly_df = df_interpolated.resample('ME').last()
monthly_returns = monthly_df.pct_change()

monthly_returns

# 2. Find A, B, and C from equations in lecture 2 
# Expected returns 
R = monthly_returns.mean()

# Covariance matrix
omega = monthly_returns.cov()

# Inverse of cov matrix
omega_inv = np.linalg.inv(omega)

# Indicator Function
indicator = np.ones(len(R))

# Calculate A, B, C
A = R.transpose() @ omega_inv @ R
B = R.transpose() @ omega_inv @ indicator
C = indicator.transpose() @ omega_inv @ indicator

print(f'A = {A:.5f}')
print(f'B = {B:.5f}')
print(f'C = {C:.5f}')

# 3. Efficient frontier without risk-free rate
# Params
mu_p_range = np.arange(0.005, 0.105, 0.005)
weights = []
stdev = []

for mu_p in mu_p_range:
    lambda_val = (mu_p * C - B) * (A * C - B**2)**-1
    gamma_val = (A - B * mu_p) * (A * C - B**2)**-1

    alpha_p = (lambda_val * omega_inv @ R) + (gamma_val * omega_inv @ indicator)

    weights.append(alpha_p)
    stdev_p = np.sqrt(alpha_p.transpose() @ omega @ alpha_p)
    stdev.append(stdev_p)

# Make weights list into dataframe
weights_df = pd.DataFrame(weights, index=[f"{mu_p:.3f}" for mu_p in mu_p_range], columns=monthly_returns.columns)

plt.plot(stdev, mu_p_range, color='red')
plt.title('Efficient Frontier')
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.show()

# 4. Efficient frontier with risk-free rate
# Params
rf_annual = 0.02
rf_rate = (1 + rf_annual) ** (1/52) - 1
weights_rf = []

# Portfolio of risky assests = tangency portfolio
excess_returns = R - rf_rate * indicator
numerator = omega_inv @ excess_returns
denominator = indicator.transpose() @ omega_inv @ excess_returns
w_q = numerator / denominator  

# Verify tangency portfolio equates to 1
weight_sum = np.sum(w_q)
print(f'Sum of tangency portfolio weights: {weight_sum:.5f}')

# Expected returns and standard deviation of tangency portfolio
mu_q = w_q.transpose() @ R
stdev_q = np.sqrt(w_q.transpose() @ omega @ w_q)

print(f'Tangency portfolio expected return: {mu_q:.5f}')
print(f'Tangency portfolio standard deviation: {stdev_q:.5f}')

for mu_p in mu_p_range:
    # Tangency portfolio proportion invested
    w_in_q = (mu_p - rf_rate) / (mu_q - rf_rate)

    # Weight in each asset = proportion in tangency portfolio * tangency portfolio weight
    weights_in_assets = w_in_q * w_q
    
    # Calculate weight in risk-free asset (1 - sum of weights in risky assets)
    weight_in_rf = 1 - w_in_q
    
    # Combine all weights (including risk-free asset)
    all_weights = np.append(weights_in_assets, weight_in_rf)
    
    weights_rf.append(all_weights)

# Make weights with risk-free list into dataframe
columns_rf = list(monthly_returns.columns) + ['Risk-Free Asset']
weights_rf_df = pd.DataFrame(
    weights_rf, 
    index=[f"{mu_p:.3f}" for mu_p in mu_p_range], 
    columns=columns_rf
)

# Calculate CML points for plotting
stdev_rf = []
for mu_p in mu_p_range:
    w_in_q = (mu_p - rf_rate) / (mu_q - rf_rate)
    stdev_p = w_in_q * stdev_q
    stdev_rf.append(stdev_p)
    
plt.plot(stdev, mu_p_range, color='red', label='Efficient frontier with no Risk-free')
plt.plot(stdev_rf, mu_p_range, 'blue', label='Efficient frontier with Risk-free (CML)')
plt.title('Efficient Frontier')
plt.xlabel('Standard Deviation')
plt.ylabel('Expected Return')
plt.legend()
plt.show()

# 5. Weights of portfolios in csv
# Without risk-free asset 3.
weights_df.to_csv('weights_withoutRf.csv')

# With risk-free asset 4.
weights_rf_df.to_csv('weights_withRf.csv')