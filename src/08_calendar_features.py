
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# Load calendar dataset
calendar = pd.read_csv("data/raw/calendar.csv")

# Convert date column
calendar["date"] = pd.to_datetime(calendar["date"])

# Create calendar features
calendar["day_of_week"] = calendar["date"].dt.day_of_week
calendar["is_weekend"] = (
    calendar["day_of_week"] >= 5
).astype(int)

calendar["month"] = calendar["date"].dt.month
calendar["year"] = calendar["date"].dt.year
calendar["day_of_month"] = calendar["date"].dt.day
calendar["quarter"] = calendar["date"].dt.quarter

# Create event indicators
calendar["is_event"] = (
    calendar["event_name_1"].notna()
).astype(int)

calendar["event_name_1"] = (
    calendar["event_name_1"].fillna("No Event")
)

calendar["event_type_1"] = (
    calendar["event_type_1"].fillna("No Event")
)

# Save calendar features
calendar.to_csv(
    "data/processed/calendar_features.csv",
    index=False
)

print("=" * 60)
print("CALENDAR FEATURE ENGINEERING")
print("=" * 60)

print("\nNew features created:")
print([
    "day_of_week",
    "is_weekend",
    "month",
    "year",
    "day_of_month",
    "quarter",
    "is_event"
])

print("\nSample output:")
print(
    calendar[
        [
            "date",
            "weekday",
            "day_of_week",
            "is_weekend",
            "month",
            "is_event",
            "event_name_1"
        ]
    ].head(15).to_string(index=False)
)

print("\nSaved: data/processed/calendar_features.csv")
print("\nCalendar feature engineering completed!")
