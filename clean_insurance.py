import pandas as pd
import os

# Define the file name
input_file = "insurance_data.csv"

# 1. Sanity Check: Does the file even exist?
if not os.path.exists(input_file):
    print(f"CRITICAL ERROR: The file '{input_file}' was not found in {os.getcwd()}")
else:
    print(f"File found! Reading {input_file}...")
    
    # 2. Use a robust read method
    try:
        df = pd.read_csv(input_file)
        
        # 3. Clean the data
        # Fill missing numeric values with the mean
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Fill missing text values with 'Unknown'
        text_cols = df.select_dtypes(include=['object']).columns
        df[text_cols] = df[text_cols].fillna('Unknown')
        
        # 4. Save the result
        df.to_csv("cleaned_insurance_data.csv", index=False)
        print("SUCCESS: Data cleaned and saved to 'cleaned_insurance_data.csv'")
        
    except Exception as e:
        print(f"An error occurred while reading: {e}")
