import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv("datasets/healthcare_iot/clean_healthcare_iot.csv")

print(df.head())

print("\nTarget_Blood_Pressure:")
print(df["Target_Blood_Pressure"].value_counts())

print("\nTarget_Heart_Rate:")
print(df["Target_Heart_Rate"].value_counts())

print("\nTarget_Health_Status:")
print(df["Target_Health_Status"].value_counts())