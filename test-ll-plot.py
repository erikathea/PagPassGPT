import pandas as pd
import matplotlib.pyplot as plt

# List of CSV filenames to load
file_prefixes = ['pass','beta','16t09i88','L4N1S3','L4N3S1','L4N2S2','S1L3S1N1*']
csv_files = [f"test-{prefix}.csv" for prefix in file_prefixes]

# List to store DataFrames
dataframes = []

# Load each CSV file
for csv_file in csv_files:
    print(f"Loading {csv_file}...")
    with open(csv_file, 'r') as file:
        lines = file.readlines()

    # Process the lines
    data = []
    for line in lines[1:]:  # Skip the header line
        parts = line.strip().split(' - ', 1)
        if len(parts) == 2:
            _, fields = parts
            data.append(fields.split('\t'))

    # Create DataFrame for this file
    columns = ['datetime', 'orig_pw', 'orig_pattern', 'orig_zxcvbn_score', 'orig_zxcvbn_guesses', 
               'orig_log_likelihood', 'pw_variant', 'pattern', 'pw_zxcvbn_score', 
               'pw_zxcvbn_guesses', 'pw_log_likelihood', 'edit_distance', 'similarity_ratio']
    df = pd.DataFrame(data, columns=columns)

    # Convert necessary columns to numeric
    numeric_columns = ['orig_zxcvbn_score', 'orig_zxcvbn_guesses', 'orig_log_likelihood', 
                       'pw_zxcvbn_score', 'pw_zxcvbn_guesses', 'pw_log_likelihood', 
                       'edit_distance', 'similarity_ratio']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add this DataFrame to the list
    dataframes.append(df)

# Combine all DataFrames into a single DataFrame
df_combined = pd.concat(dataframes, ignore_index=True)

# Debugging output for combined DataFrame
print("Combined DataFrame Debugging Values:")
print("Loglikelihood:", df_combined['pw_log_likelihood'].describe())

# Number of unique original passwords
unique_orig_pw_count = df_combined['orig_pw'].nunique()
print(f"Number of unique 'orig_pw': {unique_orig_pw_count}")
print(f"Number of unique 'pw_variant': {df_combined['pw_variant'].nunique()}")

# Create the scatter plot
log_likelihood_values = df_combined['pw_log_likelihood']

plt.figure(figsize=(10, 6))
plt.hist(log_likelihood_values, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=2.0, color='green', linestyle='--', label='< 2.0: Very similar')
plt.axvline(x=3.25, color='orange', linestyle='--', label='2.0 - 3.25: Similar')
plt.axvline(x=4.0, color='red', linestyle='--', label='> 4.0: Less similar or outliers')
plt.xlabel('Log Likelihood')
plt.ylabel('Frequency')
plt.title('Distribution of Log Likelihood Values (Combined Data)')
plt.legend()

plt.tight_layout()
plt.show()
