import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the file
with open('test-L4N2S2.csv', 'r') as file:
    lines = file.readlines()

# Process the lines
data = []
for line in lines[1:]:  # Skip the header line
    parts = line.strip().split(' - ', 1)
    if len(parts) == 2:
        _, fields = parts
        data.append(fields.split('\t'))

# Create DataFrame
columns = ['datetime','orig_pw', 'orig_pattern', 'orig_zxcvbn_score', 'orig_zxcvbn_guesses', 
           'orig_log_likelihood', 'pw_variant', 'pattern', 'pw_zxcvbn_score', 
           'pw_zxcvbn_guesses', 'pw_log_likelihood', 'edit_distance', 'similarity_ratio']
df = pd.DataFrame(data, columns=columns)

# Convert necessary columns to numeric
numeric_columns = ['orig_zxcvbn_score', 'orig_zxcvbn_guesses', 'orig_log_likelihood', 
                   'pw_zxcvbn_score', 'pw_zxcvbn_guesses', 'pw_log_likelihood', 
                   'edit_distance', 'similarity_ratio']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter rows where edit_distance
df_filtered = df.copy() #df[df['edit_distance'] == 9].copy()

# Corrected weights for the composite similarity score
w1, w2, w3 = 0.25, 0.4, 0.35

# Compute the new columns for the filtered data
df_filtered.loc[:, 'normalized_editDistance'] = 1 - (df_filtered['edit_distance'] / 32)
abs_diff = np.abs(df_filtered['pw_zxcvbn_score'] - df_filtered['orig_zxcvbn_score'])
max_diff = 4  # Maximum possible difference for zxcvbn scores
df_filtered.loc[:, 'normalized_zxcvbnScore'] = 1 - (abs_diff / max_diff)
df_filtered.loc[:, 'normalized_logLikelihood'] = np.exp(-(np.abs(df_filtered['pw_log_likelihood'] - df_filtered['orig_log_likelihood'])))

# Print intermediate values for debugging
print("Debugging Values:")
print("Normalized Edit Distance:", df_filtered['normalized_editDistance'].describe())
print("Normalized zxcvbn Score:", df_filtered['normalized_zxcvbnScore'].describe())
print("Normalized Log Likelihood:", df_filtered['normalized_logLikelihood'].describe())

# Calculate composite similarity score and add it to the DataFrame
df_filtered.loc[:, 'composite_similarity_score'] = (
    (w1 * df_filtered['normalized_editDistance']) + 
    (w2 * df_filtered['normalized_logLikelihood']) + 
    (w3 * df_filtered['normalized_zxcvbnScore'])
)

# Check for any scores greater than 1.0 and print relevant rows
if (df_filtered['composite_similarity_score'] > 1.0).any():
    print("Warning: Composite similarity score exceeded 1.0")
    print(df_filtered.loc[df_filtered['composite_similarity_score'] > 1.0, ['pw_variant', 'normalized_editDistance', 'normalized_zxcvbnScore', 'normalized_logLikelihood', 'composite_similarity_score']])

print("composite_similarity_score:", df_filtered['composite_similarity_score'].describe())

# Print the number of unique "orig_pw"
unique_orig_pw_count = df_filtered['orig_pw'].nunique()
print(f"Number of unique 'orig_pw': {unique_orig_pw_count}")
print(f"Number of unique 'pw_variant': {df_filtered['pw_variant'].nunique()}")

# Create the scatter plot
plt.figure(figsize=(18,10))

plt.scatter(df_filtered.index, df_filtered['normalized_editDistance'], label='norm_editDistance', color='blue', s=10, alpha=0.5)
plt.scatter(df_filtered.index, df_filtered['normalized_zxcvbnScore'], label='norm_zxcvbnScore', color='red', s=10, alpha=0.5)
plt.scatter(df_filtered.index, df_filtered['normalized_logLikelihood'], label='norm_LogLikelihood', color='orange', s=10, alpha=0.5)
plt.scatter(df_filtered.index, df_filtered['composite_similarity_score'], label='composite_similarity_score', color='green', s=20, alpha=1)

plt.ylim(0, 1.1)  # Set y-axis limits
plt.xlabel('Index')
plt.ylabel('Score')
plt.title('L4N2S2 Random Passwords')
plt.legend()

plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
