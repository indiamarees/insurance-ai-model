import os

import numpy as np
import pandas as pd

# Use script directory so output is saved next to this file, regardless of the current working directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, "insurance_data.csv")

# Set a random seed for reproducibility
np.random.seed(42)

n_records = 1000

# Generate mock insurance claim data
claim_ids = [f"CLM{idx:04d}" for idx in range(1, n_records + 1)]
customer_ages = np.random.randint(18, 81, size=n_records)
policy_types = np.random.choice(["Gold", "Silver", "Bronze"], size=n_records, p=[0.25, 0.45, 0.30])
claim_amounts = np.round(np.random.uniform(500.0, 25000.0, size=n_records), 2)
past_claims_count = np.random.poisson(lam=1.2, size=n_records)

# Create a Claim_Status with ~20% denied claims (1 = Denied, 0 = Approved)
claim_status = np.random.choice([0, 1], size=n_records, p=[0.80, 0.20])

insurance_data = pd.DataFrame({
    "Claim_ID": claim_ids,
    "Customer_Age": customer_ages,
    "Policy_Type": policy_types,
    "Claim_Amount": claim_amounts,
    "Past_Claims_Count": past_claims_count,
    "Claim_Status": claim_status,
})

insurance_data.to_csv(output_file, index=False)
print(f"Generated {output_file} with 1,000 mock claims.")
print(f"\nDataset shape: {insurance_data.shape}")
print(f"Approved claims: {(insurance_data['Claim_Status'] == 0).sum()} ({(insurance_data['Claim_Status'] == 0).sum() / len(insurance_data) * 100:.1f}%)")
print(f"Denied claims: {(insurance_data['Claim_Status'] == 1).sum()} ({(insurance_data['Claim_Status'] == 1).sum() / len(insurance_data) * 100:.1f}%)")
