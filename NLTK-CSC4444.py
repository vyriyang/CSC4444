# Importing necessary packages
import nltk
from nltk.corpus import movie_reviews
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.svm import SVC
from sklearn import model_selection
from nltk.classify.scikitlearn import SklearnClassifier
import pickle  # Importing pickle to save the model

# Downloading necessary NLTK data
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')  # For the word_tokenize function

# Displaying basic information about the 'movie_reviews' corpus
print("Total number of words:", len(movie_reviews.words()))
print("Categories:", movie_reviews.categories())

# Calculating and displaying the total frequency of words
words_freq_dist = FreqDist(movie_reviews.words())
print("Total frequency of words:", words_freq_dist)
print("Frequency of the word 'good':", words_freq_dist['good'])
print("Frequency of the word 'bad':", words_freq_dist['bad'])

# Loading the default stopwords list for English and adding most frequent punctuation
stop_words = set(stopwords.words('english') + [',', 'the', '.', 'a', 'and', 'of', 'to', "'", 'is', 'in', 's', '"', 'it', 'that', '-',')','(','film','one','movie','?','like',':','even','good', 'much','also','get','two',';','first','--','see','!','way','could'])

# Preprocessing text: tokenizing, removing stopwords, and stemming
porter_stemmer = PorterStemmer()

def preprocess(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmed_tokens = [porter_stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Converting movie reviews to a pandas DataFrame for easier processing
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
df = pd.DataFrame(documents, columns=['Review', 'Category'])
df['Review'] = df['Review'].apply(lambda x: ' '.join(x))

# Apply the preprocessing function
df['Processed'] = df['Review'].apply(preprocess)

# Filter out stopwords from the original frequency distribution
filtered_words_freq_dist = FreqDist(word for word in movie_reviews.words() if word.lower() not in stop_words)

# Displaying 15 most common words after removing stopwords
print("15 most common words (excluding stopwords):", filtered_words_freq_dist.most_common(15))


#print("All file ids:", movie_reviews.fileids())
#print("File ids of positive reviews:", movie_reviews.fileids('pos'))
#print("File ids of negative reviews:", movie_reviews.fileids('neg'))


'''
# Working with a specific review
#print("Words in a specific negative review:", movie_reviews.words('neg/cv001_19502.txt'))


# Example of feature extraction for a specific review
feature = {}
review = movie_reviews.words('neg/cv954_19932.txt')
for word in feature_vector:
    feature[word] = word in review
print([word for word in feature_vector if feature[word]])

'''

# Preparing the feature set
all_words = nltk.FreqDist(movie_reviews.words())
feature_vector = list(all_words)[:250]  # Using the top 4000 words

# Preparing documents as a list of (word_list, category)
documents = [(movie_reviews.words(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Function to find features in a document
def find_feature(word_list):
    feature = {word: (word in word_list) for word in feature_vector}
    return feature

# Checking the feature finding function
#print(find_feature(documents[0][0]))

# Creating feature sets for all documents
feature_sets = [(find_feature(word_list), category) for (word_list, category) in documents]

# Splitting data into training and testing sets
train_set, test_set = model_selection.train_test_split(feature_sets, test_size=0.25)
print(f"Training set size: {len(train_set)}")
print(f"Testing set size: {len(test_set)}")


# Training the model
model = SklearnClassifier(SVC(kernel='linear'))
model.train(train_set)

# Testing the model and printing the accuracy
accuracy = nltk.classify.accuracy(model, test_set)*100
print(f"SVC Accuracy: {accuracy}")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(train_set)
    accuracy = nltk.classify.accuracy(nltk_model, test_set)*100
    print("{} Accuracy: {}".format(name, accuracy))
    
    
# Ensemble methods - Voting classifier
from sklearn.ensemble import VotingClassifier

# Define names and classifiers again just for clarity
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),  # Specifying the number of trees in the forest
    LogisticRegression(solver='lbfgs', max_iter=200),  # Specifying solver and increasing max_iter
    SGDClassifier(max_iter=1000),  # Increasing max_iter to ensure convergence
    MultinomialNB(),
    SVC(kernel='linear')
]

# Convert zip to a list before passing it to VotingClassifier
models = list(zip(names, classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators=models, voting='hard', n_jobs=-1))
nltk_ensemble.train(train_set)
accuracy = nltk.classify.accuracy(nltk_ensemble, test_set)*100
print("Voting Classifier Accuracy: {:.2f}%".format(accuracy))

# Make class label prediction for testing set
txt_features, labels = zip(*test_set)

prediction = nltk_ensemble.classify_many(txt_features)

# Print a confusion matrix and a classification report
print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index=[['actual', 'actual'], ['neg', 'pos']],
    columns=[['predicted', 'predicted'], ['neg', 'pos']])



# Saving the model to disk
with open('movie_reviews_model_full.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully.")




# Example of loading the model
# with open('movie_reviews_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)
# loaded_model_accuracy = nltk.classify.accuracy(loaded_model, test_set)
# print(f"Loaded model accuracy: {loaded_model_accuracy}")

