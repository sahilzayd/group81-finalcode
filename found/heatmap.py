import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv("po2_data.csv")

# Set the style for Seaborn plots
sns.set(style="whitegrid")
# Create a heatmap to visualize correlations between numerical variables
correlation_matrix = data.corr()
plt.figure(figsize=(12, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()