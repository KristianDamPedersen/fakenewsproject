import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# Custom tokenizer with lemmatisation
def custom_tokenizer(text):
    lower_text = text.lower()

    # Tokenise with RegexpTokenizer, this regex matches a word separated by some sort of border
    tokenizer = RegexpTokenizer(r"\b[a-z]+\b")
    tokenized_words = tokenizer.tokenize(lower_text)

    # Load English stopwords
    stopwords_set = set(stopwords.words("english"))

    # Remove stopwords
    filtered_words = [word for word in tokenized_words if word not in stopwords_set]

    # Lemmatise using WordNet
    lemmatiser = WordNetLemmatizer()
    lemmatised_words = [lemmatiser.lemmatize(word) for word in filtered_words]
    return lemmatised_words
