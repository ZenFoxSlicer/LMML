"""The Weak Layer: recover the hidden word from a fitted decision tree's structure.

The tree's split questions, read in the order the tree asks them (root first,
preorder), each contribute the first letter of the feature they test.
"""
import pandas as pd  # pandas, for loading the CSV into a DataFrame
from sklearn.tree import DecisionTreeClassifier  # the decision tree model we fit

TARGET = "slab_released"  # name of the label column (1 = avalanche slab released, 0 = not)

df = pd.read_csv("train-input.csv")  # load the snowpack measurements
X = df.drop(columns=[TARGET])  # features: every column except the label
y = df[TARGET]  # labels: the slab_released column

clf = DecisionTreeClassifier(random_state=0).fit(X, y)  # fit a full-depth tree; fixed seed makes the shape reproducible
tree = clf.tree_  # low-level tree structure (parallel arrays indexed by node id)
names = list(X.columns)  # feature names, indexed the same way as tree.feature

letters = []  # collects one letter per split node, in visit order

def walk(node):  # recursive preorder traversal starting at a given node id
    left, right = tree.children_left[node], tree.children_right[node]  # ids of the two child nodes
    if left == right:  # leaf: sklearn marks both children as -1, so they are equal
        return  # leaves ask no question, so they contribute no letter
    letters.append(names[tree.feature[node]][0].upper())  # first letter of the feature this node splits on
    walk(left)  # visit the left subtree next (preorder: node, left, right)
    walk(right)  # then visit the right subtree

walk(0)  # start at the root, node 0
word = "".join(letters)  # glue the letters into the hidden word

with open("train-output.txt", "w") as f:  # open the output file for writing (overwrites it)
    f.write(f"SIGMOID_{word}\n")  # write the answer in the expected SIGMOID_<WORD> format
