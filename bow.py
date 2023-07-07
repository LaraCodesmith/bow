from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer

# Preparation of a text corpus
filename = 'path/example.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# Removal of punctuation marks and numbers
text = re.sub(r'[^a-zA-Z]', ' ', text)

# Converting the text to lowercase
text = text.lower()

# Tokenization of text into words
tokens = word_tokenize(text)

# Removal of stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_text = [' '.join([lemmatizer.lemmatize(token) for token in filtered_tokens])]

# Creating a vector representation
vectorizer = CountVectorizer()
vectorizer.fit(lemmatized_text)
vector = vectorizer.transform(lemmatized_text)

print(vector.toarray())