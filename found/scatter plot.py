import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv("po2_data.csv")

# Set the style for Seaborn plots
sns.set(style="whitegrid")

# Create scatterplots for motor_updrs vs. test_time and total_updrs vs. test_time
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.scatterplot(data=data, x="test_time", y="motor_updrs", hue="sex")
plt.title("Motor UPDRS vs. Test Time")
plt.xlabel("Test Time")
plt.ylabel("Motor UPDRS")

plt.subplot(1, 2, 2)
sns.scatterplot(data=data, x="test_time", y="total_updrs", hue="sex")
plt.title("Total UPDRS vs. Test Time")
plt.xlabel("Test Time")
plt.ylabel("Total UPDRS")

plt.tight_layout()
plt.show()

