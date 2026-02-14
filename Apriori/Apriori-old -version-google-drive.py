# writer Ali pourfereydoon 
# student - computer engineering - istanbul-atlas -university
# course : data mining
# Apriori Algorithm for Market Basket Analysis Dataset: Groceries Market Basket Dataset and Purpose: Discover frequent itemsets and association rules


# Mount Google Drive
# **************************************************************************************************************
# Mounting Google Drive allow us to access external files,
# such as the groceries dataset stored in the user's Drive. in this project you must stored the dataset in your google drive

from google.colab import drive
drive.mount('/content/drive')



# Load and Parse Dataset
# **********************************************************************************************************
# Each line in the dataset represents one transaction Items within a transaction are separated by comma
transactions = []

with open("/content/drive/MyDrive/groceries.csv", "r") as file:
    for line in file:
        # Remove extra spaces and split items into a list
        items = line.strip().split(",")
        transactions.append(items)

# Print dataset size to confirm sucessful loading
print("Total number of transactions:", len(transactions))


# One-Hot Encoding
# -----------------------------
# Apriori requires a binary input format (True/False) Encoder converts transactions into this format

from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

# Convert encoded array into a Pandas DataFrame and Rows represent transactions and columns represent items
df = pd.DataFrame(te_array, columns=te.columns_)

# Display shape of the encoded dataset
print("One-hot encoded data shape:", df.shape)


# Frequent Itemset Mining
# ------------------------------------------------------------------------------------------------------
# The Apriori algorithm identifies itemsets that meet the minimum support threshold
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(
    df,
    min_support=0.02,      # Itemset must appear in at least 2% of transactions
    use_colnames=True     # Display item names instead of column indices
)

# Output the total number of frequent itemsets found
print("Number of frequent itemsets:", len(frequent_itemsets))



# Association Rule Generation
# -----------------------------------------------------------------------------------------
# Association rules are generated from frequent itemsets using confidence as the primary evaluation metric

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3      # Minimum confidence of 30%
)

# Sort rules based on lift to highlight strong associations
rules = rules.sort_values(by="lift", ascending=False)

# Display the top rules with key evaluation metrics
print(
    rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    .head()
)



# Strong Rule Filtering
# ----------------------------------------------------------------------------------------
# Filter rules with both high confidence and lift values to focus on the most meaningful associations

strong_rules = rules[
    (rules['confidence'] > 0.5) &   # High reliability
    (rules['lift'] > 1.2)           # Strong dependency between items
]

# Display top strong association rules
print(
    strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    .head()
)



# Visualization
# ---------------------------------------------------------------------------------------
# Scatter plot to visualize the relationship between support and confidence for all association rules
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence of Association Rules (Apriori Algorithm)')
plt.grid(True)
plt.show()
