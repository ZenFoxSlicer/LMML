# SWITZERLAND: read a word from a decision tree.
# A decision tree is a chain of questions about the input columns.
# We train it to predict whether a slab is released, then visit its questions.
# Each question contributes the first letter of the column it uses.
# Visit order: current question, left branch, right branch.
# The finished answer is written to train-output.txt.

"""The Weak Layer: recover the hidden word from a fitted decision tree's structure.

The tree's split questions, read in the order the tree asks them (root first,
preorder), each contribute the first letter of the feature they test.
"""
import pandas as pd  # pandas, for loading the CSV into a DataFrame
from sklearn.tree import DecisionTreeClassifier  # the decision tree model we fit

TARGET = "slab_released"  # name of the label column (1 = avalanche slab released, 0 = not)

df = pd.read_csv("train-input.csv")  # load the snowpack measurements
# Separate what we know (X: measurements) from what we want to predict (y).
# Do not leave the answer column among the inputs: that would give the answer away.
X = df.drop(columns=[TARGET])  # features: every column except the label
y = df[TARGET]  # labels: the slab_released column

# Training chooses questions that separate the two possible answers.
# random_state fixes the random choices so repeated runs build the same tree.
clf = DecisionTreeClassifier(random_state=0).fit(X, y)  # fit a full-depth tree; fixed seed makes the shape reproducible
# The tree stores its information in arrays. The same position in each array
# refers to the same node, which is either a question or a final answer.
tree = clf.tree_  # low-level tree structure (parallel arrays indexed by node id)
names = list(X.columns)  # feature names, indexed the same way as tree.feature

letters = []  # collects one letter per split node, in visit order

# This function calls itself to explore smaller branches. That is recursion.
# A leaf is a final answer, so it contributes no question letter.
def walk(node):  # recursive preorder traversal starting at a given node id
    left, right = tree.children_left[node], tree.children_right[node]  # ids of the two child nodes
    if left == right:  # leaf: sklearn marks both children as -1, so they are equal
        return  # leaves ask no question, so they contribute no letter
    # tree.feature[node] gives a column number. names turns it into a column name.
    # [0] takes its first character, and upper() makes that character uppercase.
    letters.append(names[tree.feature[node]][0].upper())  # first letter of the feature this node splits on
    walk(left)  # visit the left subtree next (preorder: node, left, right)
    walk(right)  # then visit the right subtree

# Begin at the first question. The recursive calls handle the rest of the tree.
walk(0)  # start at the root, node 0
word = "".join(letters)  # glue the letters into the hidden word

with open("train-output.txt", "w") as f:  # open the output file for writing (overwrites it)
    f.write(f"SIGMOID_{word}\n")  # write the answer in the expected SIGMOID_<WORD> format
