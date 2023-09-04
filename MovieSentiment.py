# Aim: Classifying the positivity of movies, based on their plots.

# We will use the cinemagoer API to scrape the plots of random movies, and use NLP techniques to understand the sentiment behind each plot description.
# To begin, you must install the cinemagoer module/package.
# "pip install git+https://github.com/cinemagoer/cinemagoer"


# Imports

import pandas as pd
import string
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from imdb import Cinemagoer

ia = Cinemagoer()  # Generate an instance

# Create dataframe of plots:

searchTerm = input("Enter Movie Genre/Keyword to search on IMDB (Hint: Dystopia or epic): ")
NoOfTerms = int(input("Enter how many movies to find: "))

random_movies = ia.get_keyword(searchTerm)
movies = random_movies[:NoOfTerms]

movie_names = []
movie_ids = []
plots = []

for movie in movies:
    movie_ids.append(movie.movieID)

for id_value in movie_ids:
    mv = ia.get_movie(id_value)
    plots.append(mv['plot'])
    movie_names.append((mv['title']))

data = {'Movie_ID': movie_ids, 'Movie_Name': movie_names, 'Plot': plots}
df = pd.DataFrame(data)

df.to_csv('plots.csv')
df = pd.read_csv('plots.csv', header=None)
df = df.rename(columns=df.iloc[0]).drop(df.index[0])

text_entries = df.Plot.values.tolist()


def remove_punctuation(text_entries):
    """
    Aim: Remove punctuation from the list of texts
    Input: List of texts/strings
    Output: Texts entries with no punctuation
    """
    no_punct_text_entries = []
    for text in text_entries:
        test_str = text.translate(str.maketrans('', '', string.punctuation))
        no_punct_text_entries.append(test_str)
    return no_punct_text_entries


def remove_numbers_from_text(text_entries):
    """
    Aim: Remove numbers from the list of texts
    Input: List of texts/strings
    Output: Texts entries with no numbers
    """
    no_numbers_text_entries = []

    for text in text_entries:
        pattern = r'[0-9]'
        new_text = re.sub(pattern, '', text)
        no_numbers_text_entries.append(new_text)

    return no_numbers_text_entries


def lemmatise_texts(text_entries):
    """
    Aim: Lemmatise the list of texts
    Input: List of texts/strings
    Output: Texts entries lemmatised
    """
    lemmatised_text_entries = []
    nltk.download('wordnet')
    nltk.download('punkt')
    lemmatizer = WordNetLemmatizer()
    for text in text_entries:
        text_tokens = word_tokenize(text)
        text_lemm = [lemmatizer.lemmatize(word.lower()) for word in text_tokens]
        lemmatised_text_entries.append(' '.join(text_lemm))

    return lemmatised_text_entries


def stop_word_texts(text_entries):
    """
    Aim: Remove stop words (e.g. and, to etc.) from the list of texts
    Input: List of texts/strings
    Output: Texts entries with no stop words
    """
    nltk.download('stopwords')
    stop_words_text_entries = []
    for text in text_entries:
        text_tokens = word_tokenize(text)
        tokens = [word for word in text_tokens if not word in set(stopwords.words('english'))]
        stop_words_text_entries.append(' '.join(tokens))

    return stop_words_text_entries


def get_polarity(text):
    """
    Aim: Get the polarity text strings
    Input: A string
    Output: Polarity value of the string. (Neutral, Positive etc)
    """
    textblob = TextBlob(str(text))
    pol = textblob.sentiment.polarity
    if (pol == 0):
        return "Neutral"
    elif (pol > 0 and pol <= 0.3):
        return "Weakly Positive"
    elif (pol > 0.3 and pol <= 0.6):
        return "Positive"
    elif (pol > 0.6 and pol <= 1):
        return "Strongly Positive"
    elif (pol > -0.3 and pol <= 0):
        return "Weakly Negative"
    elif (pol > -0.6 and pol <= -0.3):
        return "Negative"
    elif (pol > -1 and pol <= -0.6):
        return "Strongly Negative"


# Put all the above functions together:

def polarity_of_entries(df, col_name, text_entries):
    """
    Aim: Clean the text entries of a data frame, and get the polarity of each text.
    Input: The df containing the texts column, Column name of the text, Column values as a list
    Output: Dataframe with the polarity column added.
    """
    result = remove_punctuation(text_entries)
    result = remove_numbers_from_text(result)
    result = lemmatise_texts(result)
    result = stop_word_texts(result)

    df[col_name] = result

    df['polarity'] = df[col_name].apply(get_polarity)
    return df


result_df = polarity_of_entries(df, 'Plot', text_entries)


def percentage(part, whole):
    """ Finding percentage """
    temp = 100 * float(part) / float(whole)
    return format(temp, '.2f')


def print_results(result_df, NoOfTerms, searchTerm):
    """
    Aim: Print the results overview
    Input: Datframe containing polarity, How many movies we are searching, The search term we used at the begining
    Output: Results overview - graph, percentages
    """

    neutral = 0
    wpositive = 0
    spositive = 0
    positive = 0
    negative = 0
    wnegative = 0
    snegative = 0
    polarity = 0

    for i in range(1, len(result_df)):
        textblob = TextBlob(str(result_df['Plot'][i]))
        polarity += textblob.sentiment.polarity
        pol = textblob.sentiment.polarity
        if (pol == 0):  # adding reaction of how people are reacting to find average later
            neutral += 1
        elif (pol > 0 and pol <= 0.3):
            wpositive += 1
        elif (pol > 0.3 and pol <= 0.6):
            positive += 1
        elif (pol > 0.6 and pol <= 1):
            spositive += 1
        elif (pol > -0.3 and pol <= 0):
            wnegative += 1
        elif (pol > -0.6 and pol <= -0.3):
            negative += 1
        elif (pol > -1 and pol <= -0.6):
            snegative += 1

    positive = percentage(positive, NoOfTerms)
    wpositive = percentage(wpositive, NoOfTerms)
    spositive = percentage(spositive, NoOfTerms)
    negative = percentage(negative, NoOfTerms)
    wnegative = percentage(wnegative, NoOfTerms)
    snegative = percentage(snegative, NoOfTerms)
    neutral = percentage(neutral, NoOfTerms)

    print(f"Finding How Positive The Top {NoOfTerms} {searchTerm} Movies Are On IMDB.")
    print()
    print("-----------------------------------------------------------------------------------------")
    print()
    print(f"On Average, The Top {NoOfTerms} {searchTerm} movies on IMDB are")

    if (polarity == 0):
        print("Neutral")
    elif (polarity > 0 and polarity <= 0.3):
        print("Weakly Positive")
    elif (polarity > 0.3 and polarity <= 0.6):
        print("Positive")
    elif (polarity > 0.6 and polarity <= 1):
        print("Strongly Positive")
    elif (polarity > -0.3 and polarity <= 0):
        print("Weakly Negative")
    elif (polarity > -0.6 and polarity <= -0.3):
        print("Negative")
    elif (polarity > -1 and polarity <= -0.6):
        print("Strongly Negative")

    print()
    print("------------------------------------------------------------------------------------------")
    print()
    print("Detailed Report: ")
    print(str(positive) + "% of plots are positive")
    print(str(wpositive) + "% of plots are weakly positive")
    print(str(spositive) + "% of plots are strongly positive")
    print(str(negative) + "% of plots are negative")
    print(str(wnegative) + "% of plots are weakly negative")
    print(str(snegative) + "% of plots are strongly negative")
    print(str(neutral) + "% of plots are neutral")

    sizes = [positive, wpositive, spositive, neutral, negative, wnegative, snegative]
    colors = ['yellowgreen', 'lightgreen', 'darkgreen', 'gold', 'red', 'lightsalmon', 'darkred']
    labels = ['Positive [' + str(positive) + '%]', 'Weakly Positive [' + str(wpositive) + '%]',
              'Strongly Positive [' + str(spositive) + '%]', 'Neutral [' + str(neutral) + '%]',
              'Negative [' + str(negative) + '%]', 'Weakly Negative [' + str(wnegative) + '%]',
              'Strongly Negative [' + str(snegative) + '%]']

    plt.figure(figsize=(10, 10))
    plt.pie(sizes, colors=colors)

    plt.legend(labels, loc="best")
    plt.title('Positivity of the movies analysed, based on their IMDB plots')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


print(result_df)
print_results(result_df, NoOfTerms, searchTerm)
