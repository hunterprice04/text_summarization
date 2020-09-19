import pandas as pd
import nltk
from Summary import Summary

# if the nltk package has not been downloaded, download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading nltk punkt")
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize


def main():
    # create the pandas data frame for the training data
    df = pd.read_csv("artificial_intelligence.csv")

    # read the text from the pd dataframe and break into a list of sentences
    sentences = []
    for s in df['article_text']:
        sentences.append(sent_tokenize(s))

    # flatten list in to 1d list of sentences
    text = [y for x in sentences for y in x]

    summary = Summary(text, 10).summary
    print(summary)


if __name__ == '__main__':
    main()