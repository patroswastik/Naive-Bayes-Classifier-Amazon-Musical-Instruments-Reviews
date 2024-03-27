import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


class NaiveBayes:
    def __init__(self, jsonFile):
        self.data = pd.read_json(jsonFile)
        self.df = pd.read_json(jsonFile)
        self.vocabulary = defaultdict(lambda: 0)
        self.vocabulary_output = defaultdict(lambda: 0)
        self.stopwords = set(stopwords.words("english"))
        self.tokenizer = nltk.tokenize.regexp.RegexpTokenizer(r"""(?x)                   # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |(?:[+/\-@&*])         # special characters with meanings
            """)
        self.overall_vc = None

    def eda(self):
        def print_columns():
            print(self.data.columns)

        def print_describe():
            print(self.data.describe())

        def print_info():
            print(self.data.info())

        def print_head():
            print(self.data.head())

        def print_count():
            print(self.data.count())

        def print_unique_values():
            print(self.data["overall"].unique())

        def print_is_null_count():
            print(self.data.isNull().sum())

        def plot_count_values_dist():
            value_count_arr = self.data["overall"].value_counts()
            self.overall_vc = value_count_arr
            ax = sns.barplot(x=value_count_arr.index, y=value_count_arr.values)
            ax.bar_label(ax.containers[0])
            plt.show()

        # print_head()
        plot_count_values_dist()

    def preprocessing(self):
        def drop_columns():
            self.df = self.df.drop(["reviewerID", "asin", "reviewerName", "unixReviewTime", "reviewTime", "summary"],
                                   axis=1)
            # print(self.df.head())

        def fix_unbalanced_data():
            pass

        def use_helpful():
            pass

        drop_columns()

    def vocabulary_creation(self):
        for review, output in zip(self.df["reviewText"], self.df["overall"]):
            review = review.lower()  # Lower-casing to avoid duplication
            tokens = self.tokenizer.tokenize(review)
            for token in tokens:
                self.vocabulary[token] += 1

    def train_test_split(self, train_size=0.8):
        X = self.df.drop(["overall"], axis=1)
        y = pd.DataFrame(self.df["overall"])
        return tts(X, y, train_size=train_size)

    def fit(self, x_train, y_train):
        for review, output in zip(x_train["reviewText"], y_train["overall"]):
            review = review.lower()  # Lower-casing to avoid duplication
            tokens = self.tokenizer.tokenize(review)
            for token in tokens:  # Not removing stopwords as mentioned in class that it won't usually help
                self.vocabulary_output[(token, output)] += 1

    def predict(self, x_input):
        output = []
        for each_row in x_input:
            print(each_row)
            analysis_log = {}
            max_prob_key, max_prob_val = 0, 0
            for i, overall_count in zip(self.overall_vc.index, self.overall_vc.values):
                analysis_log[i] = 0
                for review in each_row["reviewText"]:
                    review = review.lower()
                    tokens = self.tokenizer.tokenize(review)
                    for token in tokens:
                        analysis_log[i] += self.vocabulary_output[(token, i)] + 1 / overall_count + len(self.vocabulary)
                if analysis_log[i] > max_prob_val: max_prob_key = i; max_prob_val = analysis_log[i]
            output.append(max_prob_key)
        return output


if __name__ == '__main__':
    json_file = "Musical_Instruments_5.json"
    nb = NaiveBayes(json_file)
    nb.eda()
    nb.preprocessing()
    nb.vocabulary_creation()
    x_train, x_test, y_train, y_test = nb.train_test_split(train_size=0.8)
    nb.fit(x_train, y_train)
    print(x_test)
    y_pred = nb.predict(x_test[1:])
    print(y_pred)

