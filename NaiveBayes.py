import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import sys
import pickle

# nltk.download('wordnet')

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
        self.remove_stopwords_flag = False
        self.lemmatizer_flag = False

        print(f"Patro, Swastik, A20547200 solution:")
        print(f"Training set size: {sys.argv[1]} %  \n")

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
            ax.set(title="Total Dataset Distribution")
            plt.show()

        def plot_count_values_dist_pie():
            value_count_arr = self.overall_vc
            value_count_arr.plot.pie(autopct="%.2f")
            plt.show()

        # print_head()
        # print_total_dataset()
        get_value_counts()
        # plot_count_values_dist_bar()
        # plot_count_values_dist_pie()

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

        def remove_stopwords():
            self.remove_stopwords_flag = True

        def use_helpful():
            to_remove = []
            for index, review in enumerate(self.df["helpful"]):
                if review[1] == 0: to_remove.append(index)
            self.df = self.df.drop(to_remove)

        def use_lemmatization():
            self.lemmatizer_flag = True

        drop_columns()
        # use_helpful()
        remove_stopwords()
        use_lemmatization()

    def vocabulary_creation(self):
        for review, output in zip(self.df["reviewText"], self.df["overall"]):
            review = review.lower()  # Lower-casing to avoid duplication
            tokens = self.tokenizer.tokenize(review)
            for token in tokens:
                self.vocabulary[token] += 1

    def train_test_split(self, train_size=0.8):
        X = self.df.drop(["overall"], axis=1)
        y = pd.DataFrame(self.df["overall"])
        x_train, _, y_train, _ = tts(X, y, train_size=train_size, shuffle=False)
        _, x_test, _, y_test = tts(X, y, test_size=0.2, shuffle=False)
        return x_train, x_test, y_train, y_test

    def fit(self, x_train, y_train):
        print("Training classifier…")
        for review, output in zip(x_train["reviewText"], y_train["overall"]):
            review = review.lower()  # Lower-casing to avoid duplication
            tokens = self.tokenizer.tokenize(review)
            for token in tokens:
                if (self.remove_stopwords_flag and token not in self.stopwords) or (not self.remove_stopwords_flag):
                    if self.lemmatizer_flag:
                        token = self.lemmatizer.lemmatize(token)
                    self.vocabulary_output[(token, output)] += 1

    def predict(self, x_input):
        output = []
        output_scores = []
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
                    if (self.remove_stopwords_flag and token not in self.stopwords) or (not self.remove_stopwords_flag):
                        if self.lemmatizer_flag:
                            token = self.lemmatizer.lemmatize(token)
                        analysis_log[i] += np.log((self.vocabulary_output[(token, i)] + 1) / (overall_count + len(self.vocabulary)))
                analysis_log[i] = np.exp(analysis_log[i])
                self.probability_class[i] = analysis_log[i]
                if analysis_log[i] > max_prob_val: max_prob_key = i; max_prob_val = analysis_log[i]
            output.append(1 if max_prob_key == 0 else max_prob_key)
            output_scores.append(list(self.probability_class.values()))
        return output, self.probability_class, np.array(output_scores)

    def evalulate(self, confusion_mat):
        print("Testing classifier…")
        print("Test results / metrics:\n")
        np.seterr(invalid='ignore')
        tp = np.diag(confusion_mat)
        fp = (np.sum(confusion_mat, axis=0) - np.diag(confusion_mat))
        fn = (np.sum(confusion_mat, axis=1) - np.diag(confusion_mat))
        tn = (np.sum(confusion_mat) - (fp + fn + tp))
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        negative_predictive_value = tn / (tn + fn)
        accuracy = np.sum(tp + tn) / np.sum(confusion_mat)
        f_score = (2 * precision * recall) / (precision + recall)

        print(f'''Number of true positives: {tp}
Number of true negatives: {tn}
Number of false positives: {fp}
Number of false negatives: {fn}
Sensitivity (recall): {recall}
Specificity: {specificity}
Precision: {precision}
Negative predictive value: {negative_predictive_value}
Accuracy: {accuracy}
F-score: {f_score}''')

    def plot_roc_curve(self, y_test, y_score):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(5):  # Assuming 5 classes
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def user_input(self):
        print("\nEnter your sentence: \n")
        print("Sentence S:")
        sentence = str(input())
        user_input = pd.DataFrame({"reviewText": [sentence]})
        ans, each_prob, y_scores = self.predict(user_input)
        print(f"was classified as {ans[0]}.")
        for key, value in each_prob.items():
            print(f"P({key} | S) = {value}")

        repeat = str(input("Do you want to enter another sentence [Y/N]? "))
        if repeat == "Y" or repeat == "y":
            self.user_input()

    def plot_confusion_matrix(self, confusion_mat):
        display = ConfusionMatrixDisplay(confusion_mat, display_labels=[1, 2, 3, 4, 5])
        display.plot()
        plt.show()


if __name__ == '__main__':

    json_file = "Musical_Instruments_5.json"
    nb = NaiveBayes(json_file)

    nb.eda()
    nb.preprocessing()
    nb.vocabulary_creation()

    train_size = sys.argv[1]

    x_train, x_test, y_train, y_test = nb.train_test_split(train_size=int(sys.argv[1]) / 100)

    nb.fit(x_train, y_train)
    y_pred, _, y_scores = nb.predict(x_test)

    # Confusion Matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    nb.evalulate(confusion_mat)

    nb.user_input()