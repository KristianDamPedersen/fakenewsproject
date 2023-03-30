import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# Custom tokenizer with lemmatisation.
def custom_tokenizer(text):
    # We first lower the string
    lower_text = text.lower()

    # Tokenise with RegexpTokenizer which is matching on any word separated by some sort of border
    tokenizer = RegexpTokenizer(r"\b[a-z]+\b")
    tokenized_words = tokenizer.tokenize(lower_text)

    # Load English stopwords
    stopwords_set = set(stopwords.words("english"))

    # Remove stop words
    filtered_words = [word for word in tokenized_words if word not in stopwords_set]

    # Use WordNet to lemmatise the words into their base form
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return lemmatized_words
