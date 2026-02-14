# writer :Ali pourfereydoon 
# student- master- computer-enginnering- istanbul atlas - university
# FP-Growth Algorithm for Market Basket Analysis ,  Dataset: Groceries Market Basket Dataset
# Purpose: Discover frequent itemsets and association rules

# Step 1: Mount Google Drive
# -----------------------------
# Mount Google Drive to access the dataset stored in the user's Drive , at first you must stored the dataset in your google drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Load and Parse Dataset
# -----------------------------
# Each line in the file represents a single transaction and Items are separated by commas and stored as a list
transactions = []

with open("/content/drive/MyDrive/groceries.csv", "r") as file:
    for line in file:
        # Clean the line and split items
        items = line.strip().split(",")
        transactions.append(items)

# Display dataset information to verify correct loading
print("Total number of transactions:", len(transactions))
print("First 5 transactions:")
print(transactions[:5])


# Step 3: One-Hot Encoding
# ------------------------------------------------------------------------------------------
# FP-Growth (via mlxtend) requires transactions in binary format and Encoder converts the transaction list into True/False values
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

# Convert encoded data into a Pandas DataFrame and Rows represent transactions and columns represent grocery items
df = pd.DataFrame(te_array, columns=te.columns_)

# Display the shape of the encoded dataset
print("One-hot encoded data shape:", df.shape)
df.head()


# Frequent Itemset Mining using FP-Growth algorithems
# ------------------------------------------------------------
# FP-Growth identifies frequent itemset without generating candidates and making it faster and more memory-efficient than Apriorie algorithms
from mlxtend.frequent_patterns import fpgrowth

frequent_itemsets = fpgrowth(
    df,
    min_support=0.02,      # Minimum support threshold is (2%)
    use_colnames=True     # Display item names instead of column indices
)

# Output the number of frequent itemsets discovered
print("Number of frequent itemsets:", len(frequent_itemsets))
frequent_itemsets.head()


# Association Rule Generation
# -----------------------------------------------------------------------------------------------------------------------------
# Generate association rules from frequent itemsets and using confidence as the evaluation metric
from mlxtend.frequent_patterns import association_rules

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3      # Minimum confidence threshold (30%)
)

# Sort rules by lift to highlight strong associations
rules = rules.sort_values(by="lift", ascending=False)

# Display top association rules with key metrics
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head()


#Strong Rule Filtering
# -----------------------------------------------------------------------------------------------------------------
# Select rules with high confidence and lift to focus on meaningful purchasing patterns
strong_rules = rules[
    (rules['confidence'] > 0.5) &   # High reliability
    (rules['lift'] > 1.2)           # Strong dependency between items
]

# Display top strong association rules
strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head()


# Step 7: Visualization
# ---------------------------------------------------------------------------------------------------------
# Scatter plot to visualize the relationship between support and confidence for FP-Growth association rules
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence of Association Rules (FP-Growth)')
plt.grid(True)
plt.show()
