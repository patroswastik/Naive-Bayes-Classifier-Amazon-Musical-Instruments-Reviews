import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import sys
import pickle


class NaiveBayes(object):
    def __init__(self, jsonFile):
        self.probability_class = None
        self.data = pd.read_json(jsonFile)
        self.df = pd.read_json(jsonFile)
        self.vocabulary = defaultdict(lambda: 0)
        self.vocabulary_output = defaultdict(lambda: 0)
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = nltk.tokenize.regexp.RegexpTokenizer(r"""(?x)                   # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |(?:[+/\-@&*])         # special characters with meanings
            """)
        self.overall_vc = None
        self.reduced_class = False

        print(f"Patro, Swastik, A20547200 solution:")
        print(f"Training set size: {sys.argv[1]} %  ")

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

        def print_total_dataset():
            print(len(self.data))

        def print_is_null_count():
            print(self.data.isNull().sum())

        def get_value_counts():
            self.overall_vc = self.data["overall"].value_counts()

        def plot_count_values_dist_bar():
            value_count_arr = self.overall_vc
            ax = sns.barplot(x=value_count_arr.index, y=value_count_arr.values)
            ax.bar_label(ax.containers[0])
            plt.show()

        def plot_count_values_dist_pie():
            value_count_arr = self.overall_vc
            value_count_arr.plot.pie(autopct="%.2f")

        # print_head()
        print_total_dataset()
        get_value_counts()
        # plot_count_values_dist_bar()
        plot_count_values_dist_pie()

    def preprocessing(self):
        def drop_columns():
            self.df = self.df.drop(["reviewerID", "asin", "reviewerName", "unixReviewTime", "reviewTime", "summary"],
                                   axis=1)

        def reducing_classes():
            self.reduced_classes = True
            self.df.loc[self.df["overall"] < 4, "overall"] = 4
            # value_count_arr = self.df["overall"].value_counts()
            # print(value_count_arr)
            # ax = sns.barplot(x=value_count_arr.index, y=value_count_arr.values)
            # ax.bar_label(ax.containers[0])
            # plt.show()

        def oversampling_data():
            pass

        def undersampling_data():
            pass

        def remove_stopwords():
            pass

        def use_helpful():
            to_remove = []
            for index, review in enumerate(self.df["helpful"]):
                if review[1] == 0: to_remove.append(index)
            self.df = self.df.drop(to_remove)

        def use_lemmatization():
            pass

        drop_columns()
        # reducing_classes()
        use_helpful()
        # oversampling_data()

    def vocabulary_creation(self):
        for review, output in zip(self.df["reviewText"], self.df["overall"]):
            review = review.lower()  # Lower-casing to avoid duplication
            tokens = self.tokenizer.tokenize(review)
            for token in tokens:
                self.vocabulary[token] += 1

    def train_test_split(self, train_size=0.8):
        X = self.df.drop(["overall"], axis=1)
        y = pd.DataFrame(self.df["overall"])
        return tts(X, y, train_size=train_size, shuffle=False)

    def fit(self, x_train, y_train):
        for review, output in zip(x_train["reviewText"], y_train["overall"]):
            review = review.lower()  # Lower-casing to avoid duplication
            tokens = self.tokenizer.tokenize(review)
            for token in tokens:  # Not removing stopwords as mentioned in class that it won't usually help
                self.vocabulary_output[(token, output)] += 1

    def predict(self, x_input):
        output = []
        total_len = len(self.df)
        for each_row in x_input["reviewText"]:
            analysis_log = {}
            max_prob_key, max_prob_val = 0, 0
            self.probability_class = defaultdict(lambda x : 0)
            for i, overall_count in zip(self.overall_vc.index, self.overall_vc.values):
                analysis_log[i] = np.log(overall_count / total_len)
                review = each_row.lower()
                tokens = self.tokenizer.tokenize(review)
                for token in tokens:
                    analysis_log[i] += np.log((self.vocabulary_output[(token, i)] + 1) / (overall_count + len(self.vocabulary)))
                analysis_log[i] = np.exp(analysis_log[i])
                self.probability_class[i] = analysis_log[i]
                if analysis_log[i] > max_prob_val: max_prob_key = i; max_prob_val = analysis_log[i]
            output.append(1 if max_prob_key == 0 else max_prob_key)
        return output, self.probability_class

    def user_input(self):
        print("Enter your sentence:")
        print("Sentence S:")
        sentence = str(input())
        user_input = pd.DataFrame({"reviewText": [sentence]})
        ans, each_prob = self.predict(user_input)
        print(f"was classified as {ans[0]}.")
        for key, value in each_prob.items():
            print(f"P({key} | S) = {value}")

        repeat = str(input("Do you want to enter another sentence [Y/N]? "))
        if repeat == "Y" or repeat == "y":
            self.user_input()

    def plot_confusion_matrix(self, confusion_mat):
        display = ConfusionMatrixDisplay(confusion_mat, display_labels=[1, 2, 3, 4, 5] if not self.reduced_class else [4, 5])
        display.plot()
        plt.show()



if __name__ == '__main__':

    json_file = "Musical_Instruments_5.json"
    nb = NaiveBayes(json_file)

    nb.eda()
    nb.preprocessing()
    nb.vocabulary_creation()

    x_train, x_test, y_train, y_test = nb.train_test_split(train_size=int(sys.argv[1]) / 100)
    nb.fit(x_train, y_train)
    y_pred, _ = nb.predict(x_test)

    # Confusion Matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    nb.plot_confusion_matrix(confusion_mat)

    plt.show()

    nb.user_input()