import pandas as pd
import numpy as np
from geopy.distance import great_circle

# Load dataset
df = pd.read_csv("amazon_delivery.csv")

# Drop rows where Order_Time or Pickup_Time is missing
df = df.dropna(subset=['Order_Time', 'Pickup_Time'])

# Handle missing values (use assignment instead of inplace on chained access)
df['Agent_Rating'] = df['Agent_Rating'].fillna(df['Agent_Rating'].mean())
df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])

# Convert to datetime
df['Order_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Order_Time'], errors='coerce')
df['Pickup_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Pickup_Time'], errors='coerce')

# Drop rows where datetime parsing failed
df = df.dropna(subset=['Order_DateTime', 'Pickup_DateTime'])

# Time-based features
df['Pickup_Delay_Minutes'] = (df['Pickup_DateTime'] - df['Order_DateTime']).dt.total_seconds() / 60
df['Order_Hour'] = df['Order_DateTime'].dt.hour
df['Order_Weekday'] = df['Order_DateTime'].dt.weekday

# Geospatial distance calculation
def compute_distance(row):
    store = (row['Store_Latitude'], row['Store_Longitude'])
    drop = (row['Drop_Latitude'], row['Drop_Longitude'])
    return great_circle(store, drop).km

df['Distance_km'] = df.apply(compute_distance, axis=1)

# Encode categorical variables
categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Drop unnecessary columns
final_df = df_encoded.drop(columns=[
    'Order_ID', 'Order_Date', 'Order_Time', 'Pickup_Time',
    'Order_DateTime', 'Pickup_DateTime'
])

#Preview the final DataFrame
print(final_df.head())

#Save the final processed DataFrame to a CSV file
final_df.to_csv("processed_delivery_data.csv", index=False)

print("Processed data saved to 'processed_delivery_data.csv'")

