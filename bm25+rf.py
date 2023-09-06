#bm25
#-------vectorizing test and train combinedly
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize an empty list to store lines from both files
corpus = []

# Read lines from the first file and append them to the corpus list
with open('cleaned_text.txt', 'r') as file1:
    corpus.extend(file1.readlines())

# Read lines from the second file and append them to the corpus list
with open('test_cleaned_text.txt', 'r') as file2:
    corpus.extend(file2.readlines())

# Initialize the TF-IDF vectorizer
from rank_bm25 import BM25Okapi
import numpy as np

tokenized_corpus = [doc.split() for doc in corpus]

# Initialize BM25Okapi with the tokenized corpus
bm25 = BM25Okapi(tokenized_corpus)

# Create a list of queries, where each query is a single document from the corpus
queries = tokenized_corpus

# Calculate BM25 scores for the queries (documents)
bm25_scores = [bm25.get_scores(query) for query in queries]

# Convert the BM25 scores into a NumPy array
bm25_matrix = np.array(bm25_scores)

# Convert the BM25 matrix to a NumPy array
bm25_ = np.array(bm25_scores)
num_lines_file = len(open('cleaned_text.txt').readlines())
print(num_lines_file)
num_lines_file1 = len(open('test_cleaned_text.txt').readlines())
print(num_lines_file1)


#---------------------

# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer()
# X = cv.fit_transform(corpus)
# docs = X.toarray()
# X_train = docs[:num_lines_file][:]
# X_test = docs[num_lines_file:][:]
# print(X_train.shape)
X_train = bm25_[:num_lines_file][:]
# X_test = bm25_[num_lines_file:][:]

from sklearn.model_selection import train_test_split


with open('encoded_labels.txt', 'r') as file:
    y_train = np.array([int(line.strip()) for line in file])

# print(y_train.shape)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
SEED=123

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=SEED)

# Define a grid of hyperparameters to search
param_grid = {
    'n_estimators': [100],       # Number of trees in the forest
    # 'max_depth': [None, 10, 20, 30],       # Maximum depth of each tree
    # 'min_samples_split': [2, 5, 10],       # Minimum samples required to split a node
    # 'min_samples_leaf': [1, 2, 4],         # Minimum samples required for a leaf node
    # 'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for the best split
}

# Create a GridSearchCV object with the Random Forest classifier and parameter grid
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Use the best hyperparameters to create a new Random Forest classifier
best_rf = RandomForestClassifier(random_state=SEED, **best_params)

# Fit the best model to the training data
best_rf.fit(X_train, y_train)

# # Predict on the training and testing data using the best model
y_pred_train_rf = best_rf.predict(X_train)
y_pred_test_rf = best_rf.predict(X_test)

# Evaluate the performance of the best Random Forest model
print("Best Model - Training Accuracy score:", accuracy_score(y_train, y_pred_train_rf))
print("Best Model - Testing Accuracy score:", accuracy_score(y_test, y_pred_test_rf))


