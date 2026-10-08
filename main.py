import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------
# STEP 1: Train a small NLP model to understand user replies
# (e.g. "yeah", "nope", "maybe" → yes / no / maybe)
# ---------------------------------------------------------------
resp_df = pd.read_csv("responses.csv", skipinitialspace=True)
resp_df.columns = resp_df.columns.str.strip()

vectorizer = CountVectorizer()                          # converts text to numbers
nb_model   = MultinomialNB()                            # simple text classifier
nb_model.fit(
    vectorizer.fit_transform(resp_df["text"]),          # input: raw sentences
    resp_df["label"]                                    # output: yes / no / maybe
)

def interpret(text):
    """Turn a freeform reply into yes / no / maybe."""
    return nb_model.predict(vectorizer.transform([text]))[0]


# ---------------------------------------------------------------
# STEP 2: Load the food dataset and encode text → numbers
# Machine learning models only understand numbers, not words,
# so we convert each category (e.g. "spicy") to an integer.
# ---------------------------------------------------------------
df = pd.read_csv("food_dataset.csv")
df.columns = df.columns.str.strip()

# Make everything lowercase so "Spicy" and "spicy" are treated the same
df = df.apply(lambda col: col.map(
    lambda x: str(x).strip().lower() if isinstance(x, str) else x
))

FOOD_COL = "Food"
features = [col for col in df.columns if col != FOOD_COL]  # all columns except Food

# Encode every column: "spicy" → 0, "sweet" → 1, etc.
encoders   = {}       # we save each encoder so we can decode answers later
encoded_df = df.copy()

for col in df.columns:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(df[col])
    encoders[col]   = le                                # save for later use

X = encoded_df[features]   # features  (what we ask about)
y = encoded_df[FOOD_COL]   # target    (what we want to guess)


# ---------------------------------------------------------------
# STEP 3: Train the Decision Tree
# The tree automatically learns WHICH question to ask first
# and how to branch based on the answer — just like Akinator.
# ---------------------------------------------------------------
tree = DecisionTreeClassifier(criterion="entropy")  # entropy = info gain splitting
tree.fit(X, y)


# ---------------------------------------------------------------
# STEP 4: Walk the tree interactively
#
# A decision tree is just a series of if/else questions.
# We start at the root (node 0) and move left or right
# depending on the user's answer, until we reach a leaf
# node that tells us the food.
#
#         [Is it spicy?]          ← node 0 (root)
#        /              \
#    [Yes]             [No]
#   [Biryani]      [Is it sweet?]  ← node 2
#                  /          \
#              [Yes]          [No]
#             [Cake]         [Rice]
# ---------------------------------------------------------------
def run_akinator():
    print("\nThink of a food and I'll guess it!\n")

    node    = 0   # always start at the root of the tree
    q_count = 0

    while True:

        # tree_.feature[node] == -2 means this is a LEAF node (no more questions)
        if tree.tree_.feature[node] == -2:
            # value[node] holds class counts → argmax gives the winning class
            food_encoded = tree.tree_.value[node].argmax()
            food_name    = encoders[FOOD_COL].inverse_transform([food_encoded])[0]
            print(f"\n🎯 Your food is: {food_name.title()}")
            return

        # --- Ask the question this node wants to ask ---
        feat_index = tree.tree_.feature[node]      # which feature to split on
        feat_name  = features[feat_index]           # human-readable name
        threshold  = tree.tree_.threshold[node]     # the split boundary

        # Show the user what values are valid for this feature
        unique_vals = sorted(df[feat_name].unique())
        hint        = f" [{' / '.join(unique_vals[:6])}]" if len(unique_vals) <= 6 else ""

        q_count += 1
        raw_input = input(f"Q{q_count}. {feat_name.capitalize()}{hint}: ").strip().lower()

        # Use NLP to handle fuzzy answers like "yeah" or "nah"
        nlp_label = interpret(raw_input)

        # Try to convert the answer to a number the tree understands
        try:
            answer      = raw_input if raw_input in unique_vals else nlp_label
            encoded_val = encoders[feat_name].transform([answer])[0]
        except ValueError:
            print(f"   ⚠️  Couldn't understand '{raw_input}', skipping.")
            node = tree.tree_.children_right[node]  # default: go right
            continue

        # --- Move to the next node ---
        # Left branch  = answer is <= threshold
        # Right branch = answer is >  threshold
        if encoded_val <= threshold:
            node = tree.tree_.children_left[node]
        else:
            node = tree.tree_.children_right[node]

        print("   ✓ Got it.")


if __name__ == "__main__":
    run_akinator()