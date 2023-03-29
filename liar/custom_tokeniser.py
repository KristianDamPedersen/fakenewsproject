import nltk
from nltk.stem import WordNetLemmatizer
import contractions
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# Custom tokenizer with lemmatization
def custom_tokenizer(text):
    lower_text = text.lower()

    # Tokenize with RegexpTokenizer
    tokenizer = RegexpTokenizer(r"\b[a-z]+\b")
    tokenized_words = tokenizer.tokenize(lower_text)

    # Load English stopwords
    stopwords_set = set(stopwords.words("english"))

    # Remove stopwords
    filtered_words = [word for word in tokenized_words if word not in stopwords_set]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return lemmatized_words
