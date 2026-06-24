import pandas as pd
import numpy as np
df=pd.read_csv('C:\MLProject\HealthInsuranceProject\data\insurance_100k_dataset.csv')

tier_1_cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai",
    "Kolkata", "Hyderabad", "Pune"
]

tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi",
    "Visakhapatnam", "Coimbatore", "Bhopal", "Nagpur", "Vadodara",
    "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati",
    "Thiruvananthapuram", "Ludhiana", "Nashik", "Allahabad",
    "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem",
    "Vijayawada", "Tiruchirappalli", "Bhavnagar", "Gwalior",
    "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode",
    "Warangal", "Kolhapur", "Bilaspur", "Jalandhar",
    "Noida", "Guntur", "Asansol", "Siliguri"
]

def age_group(age):
    if age<25:
        return 'young'
    elif age<45:
        return 'adult'
    elif age<60:
        return 'middle_aged'
    return 'senior'

def lifestyle_risk(row):
    if row['smoker'] and row['bmi'] >30:
        return 'high'
    elif row['smoker'] or row['bmi']>27:
        return 'medium'
    else:
        return 'low'

def city_tier(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    else:
        return 3
    

def create_features(df):
    df_feat=df.copy()

    df_feat['bmi']=df_feat['weight']/(df_feat['height']**2)
    df_feat['age_group']=df_feat['age'].apply(age_group)
    df_feat['lifestyle_risk']=df_feat.apply(lifestyle_risk,axis=1)
    df_feat['city_tier']=df_feat['city'].apply(city_tier)

    df_feat = df_feat.drop(
        columns=['age','weight','height','smoker','city']
    )
    return df_feat