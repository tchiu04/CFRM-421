import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('cont&Kukanov/l1_day.csv')

# Print basic info about the data
print("\nBasic DataFrame Info:")
print(df.info())

# Look at venue sizes
print("\nVenue Size Analysis:")
print("===================")

# Get unique publishers
publishers = df['publisher_id'].unique()
print(f"\nNumber of unique publishers: {len(publishers)}")

# For each publisher, show size statistics
for pub in publishers:
    pub_data = df[df['publisher_id'] == pub]
    sizes = pub_data['ask_sz_00']
    
    print(f"\nPublisher {pub}:")
    print(f"  Average size: {sizes.mean():.2f}")
    print(f"  Max size: {sizes.max()}")
    print(f"  Min size: {sizes.min()}")
    print(f"  Total messages: {len(pub_data)}")
    
    # Show size distribution
    size_counts = sizes.value_counts().head()
    print("  Most common sizes:")
    for size, count in size_counts.items():
        print(f"    {size} shares: {count} times")

# Look specifically for the 8920 size venue
print("\nLooking for venue with 8920 shares:")
large_venue = df[df['ask_sz_00'] == 8920]
if not large_venue.empty:
    print("\nFound 8920 share venue:")
    print(f"Publisher ID: {large_venue['publisher_id'].iloc[0]}")
    print(f"Number of occurrences: {len(large_venue)}")
    print("\nSample of these messages:")
    print(large_venue[['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']].head())
else:
    print("No venue found with exactly 8920 shares")

# Show distribution of all sizes
print("\nSize Distribution Summary:")
size_stats = df['ask_sz_00'].describe()
print(size_stats)

# Show most common sizes
print("\nMost Common Sizes:")
size_counts = df['ask_sz_00'].value_counts().head(10)
print(size_counts)



