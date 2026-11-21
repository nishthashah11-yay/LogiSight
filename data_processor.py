"""
Data Processing Module for LogiSight Analytics Last-Mile Delivery Dashboard
Handles data ingestion, cleaning, feature engineering, and statistical aggregation.
Strictly preserves raw data integrity while creating robust analytical features.
"""

import os
import numpy as np
import pandas as pd


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).
    Uses absolute coordinates to handle negative coordinate entries.
    Returns distance in kilometers.
    """
    lat1_r = np.radians(np.abs(lat1))
    lon1_r = np.radians(np.abs(lon1))
    lat2_r = np.radians(np.abs(lat2))
    lon2_r = np.radians(np.abs(lon2))

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def load_raw_data(filepath=None):
    """
    Load the raw delivery dataset without any modifications.
    """
    if filepath is None:
        # Default search paths
        candidate_paths = [
            "Last mile Delivery Data.csv",
            os.path.join(os.path.dirname(__file__), "Last mile Delivery Data.csv"),
            r"d:\Users\mann\Downloads\Last mile Delivery Data.csv",
            r"d:\Users\mann\Downloads\Math For AI II FA 2\Last mile Delivery Data.csv"
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                filepath = p
                break
    
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(f"Delivery dataset not found at paths: {candidate_paths}")
    
    return pd.read_csv(filepath)


def clean_and_prepare_data(raw_df, late_threshold_mode="stat", custom_sla_mins=150):
    """
    Cleans and prepares data according to Stage 4 requirements:
    - Standardizes text columns and removes whitespace
    - Handles missing values sensibly:
        * Agent_Rating: imputed with median rating
        * Weather: imputed with mode
        * Traffic: cleans 'NaN' string representations and imputes with mode
    - Feature Engineering:
        * Distance_KM: Haversine distance from store to drop
        * Agent_Age_Group: '<25', '25-40', '40+'
        * Prep_Time_Min: duration between order and pickup
        * Order_Date: datetime parsing, Month, Day_of_Week, Day_Name
        * Late Delivery Indicator (Is_Late): Delivery_Time > threshold (mean + 1 std or custom SLA)
        * Delivery_Speed_KMPH: estimated delivery speed
    """
    df = raw_df.copy()

    # 1. Clean string columns (strip trailing whitespace and fix 'NaN' string artifacts)
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'nan': np.nan, 'NaN': np.nan, '': np.nan})

    # Standardize vehicle naming for presentation
    df['Vehicle'] = df['Vehicle'].replace({
        'bicycle': 'Bicycle',
        'motorcycle': 'Motorcycle',
        'scooter': 'Scooter',
        'van': 'Van'
    })

    # Standardize Area naming
    df['Area'] = df['Area'].replace({
        'Metropolitian': 'Metropolitan',
        'Metropolitian ': 'Metropolitan',
        'Urban ': 'Urban',
        'Semi-Urban ': 'Semi-Urban',
        'Other': 'Other'
    })

    # 2. Sensible Imputation of Missing Values
    # Agent_Rating: median imputation
    median_rating = df['Agent_Rating'].median()
    df['Agent_Rating_Imputed'] = df['Agent_Rating'].isnull()
    df['Agent_Rating'] = df['Agent_Rating'].fillna(median_rating)

    # Weather: mode imputation
    mode_weather = df['Weather'].mode()[0] if not df['Weather'].mode().empty else 'Sunny'
    df['Weather'] = df['Weather'].fillna(mode_weather)

    # Traffic: mode imputation
    mode_traffic = df['Traffic'].mode()[0] if not df['Traffic'].mode().empty else 'Low'
    df['Traffic'] = df['Traffic'].fillna(mode_traffic)

    # 3. Feature Engineering: Haversine Distance (km)
    df['Distance_KM'] = calculate_haversine_distance(
        df['Store_Latitude'], df['Store_Longitude'],
        df['Drop_Latitude'], df['Drop_Longitude']
    )
    # Floor distance to reasonable bounds (0.5 km minimum if zero)
    df['Distance_KM'] = df['Distance_KM'].apply(lambda x: 0.5 if x < 0.1 else round(x, 2))

    # 4. Feature Engineering: Agent Age Group
    # Rubric specified: (<25, 25-40, 40+)
    bins = [0, 24, 40, 120]
    labels = ['<25', '25–40', '40+']
    df['Agent_Age_Group'] = pd.cut(df['Agent_Age'], bins=bins, labels=labels, right=True)

    # 5. Feature Engineering: Order Date & Time Processing
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['Order_Month'] = df['Order_Date'].dt.month
    df['Order_Month_Name'] = df['Order_Date'].dt.strftime('%B %Y')
    df['Order_Day_Name'] = df['Order_Date'].dt.day_name()
    df['Order_Day_Of_Week'] = df['Order_Date'].dt.dayofweek

    # Parse Order_Time & Pickup_Time
    order_time_clean = df['Order_Time'].astype(str).str.strip().replace({'nan': np.nan, 'NaN': np.nan})
    pickup_time_clean = df['Pickup_Time'].astype(str).str.strip().replace({'nan': np.nan, 'NaN': np.nan})

    # Calculate Order Hour
    df['Order_Hour'] = pd.to_datetime(order_time_clean, format='%H:%M:%S', errors='coerce').dt.hour
    df['Order_Hour'] = df['Order_Hour'].fillna(12).astype(int)

    # Calculate Order Preparation Time (order-to-pickup minutes)
    t_order = pd.to_timedelta(order_time_clean.fillna('12:00:00'))
    t_pickup = pd.to_timedelta(pickup_time_clean.fillna('12:10:00'))
    prep_diff = (t_pickup - t_order).dt.total_seconds() / 60.0
    # Handle wrap-around across midnight
    prep_diff = prep_diff.apply(lambda x: x + 1440 if x < 0 else x)
    df['Prep_Time_Min'] = prep_diff.clip(lower=0, upper=60).round(1)

    # 6. Feature Engineering: Late Delivery Indicator
    # Rubric: "deliveries whose times exceed the mean plus one standard deviation"
    overall_mean = df['Delivery_Time'].mean()
    overall_std = df['Delivery_Time'].std()
    stat_threshold = round(overall_mean + overall_std, 2)

    if late_threshold_mode == "stat":
        threshold = stat_threshold
    else:
        threshold = custom_sla_mins

    df['Late_Threshold_Used'] = threshold
    df['Is_Late'] = (df['Delivery_Time'] > threshold).astype(int)
    df['Delivery_Status'] = df['Is_Late'].map({1: 'Delayed (Late)', 0: 'On-Time'})

    # 7. Estimated Delivery Speed (km/h)
    time_hours = df['Delivery_Time'] / 60.0
    df['Delivery_Speed_KMPH'] = (df['Distance_KM'] / time_hours).round(2)

    return df, stat_threshold, overall_mean, overall_std


def get_kpi_summary(df):
    """
    Returns executive KPI metrics matching the storyboard:
    - Total Deliveries
    - Average Delivery Time
    - Delay Rate (%)
    - Average Agent Rating
    - Delayed Deliveries Count
    - On-Time Deliveries Count
    """
    total = len(df)
    if total == 0:
        return {
            'total_deliveries': 0,
            'avg_delivery_time': 0.0,
            'delay_rate': 0.0,
            'avg_rating': 0.0,
            'delayed_count': 0,
            'ontime_count': 0
        }

    avg_time = df['Delivery_Time'].mean()
    delayed_count = int(df['Is_Late'].sum())
    delay_rate = (delayed_count / total) * 100.0
    avg_rating = df['Agent_Rating'].mean()
    ontime_count = total - delayed_count

    return {
        'total_deliveries': total,
        'avg_delivery_time': round(avg_time, 1),
        'delay_rate': round(delay_rate, 1),
        'avg_rating': round(avg_rating, 2),
        'delayed_count': delayed_count,
        'ontime_count': ontime_count
    }


if __name__ == "__main__":
    raw = load_raw_data()
    print(f"Loaded raw dataset with {len(raw)} rows and {len(raw.columns)} columns.")
    cleaned, thresh, m, s = clean_and_prepare_data(raw)
    print(f"Processed dataset: Mean={m:.2f}, Std={s:.2f}, Late Threshold={thresh:.2f}")
    kpis = get_kpi_summary(cleaned)
    print("KPIs:", kpis)
