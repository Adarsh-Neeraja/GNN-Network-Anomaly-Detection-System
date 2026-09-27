import pandas as pd

# 1. Define your file names
giant_file = "cicids2017_cleaned.csv" 
safe_mini_file = "network_data_safe.csv"

# 2. Create a list to hold our small data bites
sampled_chunks = []

print("Starting to process the giant file safely...")
print("Reading in chunks of 50,000 rows...")

# 3. CHUNKING: Read the giant file 50,000 rows at a time to save RAM
try:
    for i, chunk in enumerate(pd.read_csv(giant_file, chunksize=50000)):
        
        # Take a random 5% sample of this specific chunk (about 2,500 rows)
        # random_state=42 ensures we get the exact same random mix every time
        small_chunk = chunk.sample(frac=0.05, random_state=42)
        
        # Add this tiny piece to our list
        sampled_chunks.append(small_chunk)
        
        print(f"Processed chunk {i+1}...")
        
        # Stop early once we have 5 chunks (about 12,500 rows total)
        if len(sampled_chunks) >= 5:
            break

    # 4. Glue our tiny pieces together into one small DataFrame
    mini_dataset = pd.concat(sampled_chunks)

    # 5. Save it as a brand new CSV!
    mini_dataset.to_csv(safe_mini_file, index=False)

    print("\n✅ Success!")
    print(f"Created a safe mini-dataset with {len(mini_dataset)} rows.")
    print(f"File saved as: '{safe_mini_file}'")
    print("You can now safely delete the giant 'cicids2017_cleaned.csv' file!")

except FileNotFoundError:
    print(f"❌ Error: Could not find the file '{giant_file}'.")
    print("Please make sure the giant CSV is extracted and is in the exact same folder as this script.")