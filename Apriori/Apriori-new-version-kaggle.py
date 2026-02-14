# writer Ali pourfereydoon 
# student - computer engineering - istanbul-atlas -university
# course : data mining
# Apriori Algorithm for Market Basket Analysis Dataset: Groceries Market Basket Dataset and Purpose:
#  Discover frequent itemsets and association rules

# get dataset directly from the kaggle website
# ******************************************************************************************************

# at first login to the kaggle acount (Step 1: Log in to Kaggle)
# Step 2: Go to Account Settings
# Step 3: Create the API Token
# download kaggle.json
# After downloading kaggle.json, go to the google colab and upload it to Colab


!pip install kaggle

from google.colab import files
files.upload()  # upload kaggle.json

# *****************************************************************************
# Then configure Kaggle:

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d irfanasrullah/groceries

!unzip groceries.zip

# Load Groceries Market Basket Dataset (line by line)
# Load and Parse Dataset
# -----------------------------------------------------------------------------------------------------------------------
# Each line in the dataset represents one transaction Items within a transaction are separated by comma

transactions = []
with open("groceries.csv", "r") as file:
    for line in file:
 # Remove extra spaces and split items into a list
        items = line.strip().split(",")
        transactions.append(items)
 # Print dataset size to confirm sucessful loading


print("Total number of transactions:", len(transactions))

# One-Hot Encoding
# -----------------------------
# Apriori requires a binary input format (True/False) Encoder converts transactions into this format

print("First 5 transactions:")
print(transactions[:5])

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
print(df.head())

from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(
    df,
    min_support=0.02,
    use_colnames=True
)

# Output the total number of frequent itemsets found

print("Number of frequent itemsets:", len(frequent_itemsets))


# Association Rule Generation
# -----------------------------------------------------------------------------------------
# Association rules are generated from frequent itemsets using confidence as the primary evaluation metric

frequent_itemsets.head()

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3
)

rules = rules.sort_values(by="lift", ascending=False)

# Sort rules based on lift to highlight strong associations

rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head()

# Strong Rule Filtering
# ----------------------------------------------------------------------------------------
# Filter rules with both high confidence and lift values to focus on the most meaningful associations
strong_rules = rules[
    (rules['confidence'] > 0.5) &
    (rules['lift'] > 1.2)
]

strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head()


# Visualization
# ---------------------------------------------------------------------------------------
# Scatter plot to visualize the relationship between support and confidence for all association rules

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence of Association Rules (Apriori Algorithm)')
plt.grid(True)
plt.show()