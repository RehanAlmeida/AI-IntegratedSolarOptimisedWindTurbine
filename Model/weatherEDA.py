import pandas as pd

# Load the CSV file
file_path = "bandraweatherdataset.csv"
df = pd.read_csv(file_path)

# Convert 'device_date_time' to datetime
df['device_date_time'] = pd.to_datetime(df['device_date_time'])

# Extract total unique dates
total_dates = df['device_date_time'].dt.date.nunique()

# Descriptive statistics for numerical columns
stats = df[['wind_speed', 'humidity', 'temperature', 'wind_direction']].agg(['min', 'max', 'mean', 'median', lambda x: x.mode().iloc[0]])

# Unique values for categorical columns
unique_values = {
    "city_name": df['city_name'].nunique(),
    "locality_name": df['locality_name'].nunique(),
    "rain_intensity": df['rain_intensity'].nunique(),
    "rain_accumulation": df['rain_accumulation'].nunique()
}

# Display results
print(f"Total unique dates: {total_dates}\n")
print("Descriptive Statistics:")
print(stats.rename(index={"<lambda_0>": "mode"}), "\n")
print("Unique values in categorical columns:")
for key, value in unique_values.items():
    print(f"{key}: {value}")
