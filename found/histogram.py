import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv("po2_data.csv")

# Set the style for Seaborn plots
sns.set(style="whitegrid")
# Create histograms for age, motor_updrs, and total_updrs
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(data["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")

plt.subplot(1, 3, 2)
sns.histplot(data["motor_updrs"], bins=20, kde=True)
plt.title("Motor UPDRS Distribution")
plt.xlabel("Motor UPDRS")

plt.subplot(1, 3, 3)
sns.histplot(data["total_updrs"], bins=20, kde=True)
plt.title("Total UPDRS Distribution")
plt.xlabel("Total UPDRS")

plt.tight_layout()
plt.show()