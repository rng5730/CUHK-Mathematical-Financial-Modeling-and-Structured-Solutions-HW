# 1. read HW2.xlsx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

# Read the Excel file into DataFrames
file_path = 'HW2.xlsx'
equity_df = pd.read_excel(file_path, sheet_name='equity', index_col=0)
factor_df = pd.read_excel(file_path, sheet_name='factor', index_col=0)
print("Equity DataFrame Shape:", equity_df.shape)
print("Factor DataFrame Shape:", factor_df.shape)

# Convert DataFrames to NumPy arrays
equity_array = equity_df.values
factor_array = factor_df.values

# Calculate simple returns
equity_returns = equity_array[1:] / equity_array[:-1] - 1
factor_returns = factor_array[1:] / factor_array[:-1] - 1

# Create DataFrames for the returns
equity_returns_df = pd.DataFrame(
    equity_returns, 
    index=equity_df.index[1:], 
    columns=equity_df.columns
)

factor_returns_df = pd.DataFrame(
    factor_returns, 
    index=factor_df.index[1:], 
    columns=factor_df.columns
)

# Display the first few rows of the returns
print("\nEquity Returns (first 5 rows):")
print(equity_returns_df.head())

print("\nFactor Returns (first 5 rows):")
print(factor_returns_df.head())

# Basic statistics of the returns
print("\nEquity Returns Statistics:")
print(equity_returns_df.describe())

print("\nFactor Returns Statistics:")
print(factor_returns_df.describe())

# 2. Setting parameters
reqExp = 0.8 # Required explanatory power
reqCorr = 0.4 # Required minimum correlation
reqFcorr = 0.7 # Maximum allowed between-factor correlation

# 3. PCA analysis on equity returns
# Convert to numpy array for PCA
equity_returns_array = equity_returns_df.values

# Standardize returns for PCA
equity_returns_mean = np.mean(equity_returns_array)
equity_returns_std = np.std(equity_returns_array)
equity_returns_std_array = (equity_returns_array - equity_returns_mean) / equity_returns_std

# Calculate covariance matrix
cov_matrix = np.cov(equity_returns_std_array, rowvar=False)

# Get the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
x = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[x]
eigenvectors = eigenvectors[:, x]

# Calculate explained variance ratio
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Find minimum number of PCs to meet required explanatory power
num_pcs = np.where(cumulative_variance_ratio >= reqExp)[0][0] + 1
print(f"Number of PCs needed to explain {reqExp*100}% of variance: {num_pcs}")

# Calculate principal components (PC scores)
pc_scores = equity_returns_std_array @ eigenvectors

# 4. Find relevant factors that represent the principal components
selected_factors = []
selected_factor_indices = []
selected_factor_corrs = []

# For each principal component loop
for pc_x in range(num_pcs):
    pc = pc_scores[:, pc_x]
    # Calculate correlation of each factor with the PC
    for factor_x, factor_name in enumerate(factor_returns_df.columns):
        factor_returns = factor_returns_df.iloc[:, factor_x].values
        
        # Calculate correlation
        corr, _ = pearsonr(pc, factor_returns)
        abs_corr = abs(corr)
        
        # Check if correlation meets reqCorr threshold
        if abs_corr > reqCorr:
            # For the first PC, add if correlation is high enough
            if pc_x == 0:
                # Check if this factor correlates too highly with any already selected factors
                too_correlated = False
                for selected_x in selected_factor_indices:
                    selected_factor = factor_returns_df.iloc[:, selected_x].values
                    factor_corr, _ = pearsonr(factor_returns, selected_factor)
                    if abs(factor_corr) > reqFcorr:
                        too_correlated = True
                        break
                
                if not too_correlated:
                    selected_factors.append(factor_name)
                    selected_factor_indices.append(factor_x)
                    selected_factor_corrs.append((pc_x, factor_name, corr))
            
            # For subsequent PCs, check additional criteria
            else:
                # Skip if factor is already selected
                if factor_x in selected_factor_indices:
                    continue
                
                # Check correlation with existing factors
                too_correlated = False
                for selected_x in selected_factor_indices:
                    selected_factor = factor_returns_df.iloc[:, selected_x].values
                    factor_corr, _ = pearsonr(factor_returns, selected_factor)
                    if abs(factor_corr) > reqFcorr:
                        too_correlated = True
                        break
                
                if not too_correlated:
                    selected_factors.append(factor_name)
                    selected_factor_indices.append(factor_x)
                    selected_factor_corrs.append((pc_x, factor_name, corr))

print(f"Selected factors: {selected_factors}")
print(f"Selected factor correlations with PCs: {selected_factor_corrs}")

# 5. Normalize (standardize) 
# Create a DataFrame with standardized factor returns
selected_factor_returns = factor_returns_df.iloc[:, selected_factor_indices]
standardized_factors = (selected_factor_returns - selected_factor_returns.mean()) / selected_factor_returns.std()

# Standardize equity returns
standardized_equity = (equity_returns_df - equity_returns_df.mean()) / equity_returns_df.std()

# 6. Run OLS for each equity index over standardized factors
beta_dict = {}
tvalue_dict = {}
rsq_dict = {}

# Add a constant term to the standardized factors for the regression
X = sm.add_constant(standardized_factors)

# Run regression for each equity index
for equity_x, equity_name in enumerate(equity_returns_df.columns):
    # Get standardized returns for this equity index
    y = standardized_equity.iloc[:, equity_x]
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    # Store results
    beta_dict[equity_name] = model.params.values[1:]  # Skip the constant term
    tvalue_dict[equity_name] = model.tvalues.values[1:]  # Skip the constant term
    rsq_dict[equity_name] = model.rsquared
    
# Create and save DataFrames for beta, t-value, and R-squared
beta_df = pd.DataFrame(beta_dict, index=selected_factors).transpose()
tvalue_df = pd.DataFrame(tvalue_dict, index=selected_factors).transpose()
rsq_df = pd.DataFrame({'R-squared': rsq_dict}).transpose()

# Save to CSV
beta_df.to_csv('beta.csv')
tvalue_df.to_csv('tvalue.csv')
rsq_df.to_csv('Rsq.csv')