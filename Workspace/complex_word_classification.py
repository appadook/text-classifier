"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Kurtik Appadoo

I affirm that I will carry out my academic endeavors with full academic honesty, and I rely on my fellow students to do the same.

Complete this file for parts 2-4 of the project.

"""

from collections import defaultdict
import gzip
import numpy as np
import nltk
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC as SVM

from syllables import count_syllables

from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate

def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    
    words, labels = load_file(data_file)
    y_pred = [1] * len(words)
    evaluate(y_pred, labels)


### 2.2: Word length thresholding

def _make_predictions(words, threshold):
    """Private method to make predictions based on the given length threshold."""
    y_pred = []
    for word in words:
        if len(word) > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    best_threshold = 0
    best_fscore = 0
    max_length = 9

    for threshold in range(1, max_length + 1):
        y_pred = _make_predictions(train_words, threshold)
        fscore = get_fscore(y_pred, train_labels)
        if fscore > best_fscore:
            best_fscore = fscore
            best_threshold = threshold  
    
    train_predictions = _make_predictions(train_words, best_threshold)
    dev_predictions = _make_predictions(dev_words, best_threshold)

    print(f"Best length threshold: {best_threshold}")
    print("\nTraining data results:")
    evaluate(train_predictions, train_labels)
    print("\nDevelopment data results:")
    evaluate(dev_predictions, dev_labels)
        


### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """
    max_count = max(counts.values())
    min_count = min(counts.values())
    print(f"Maximum frequency count: {max_count}")
    print(f"Minimum frequency count: {min_count}")

    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    best_threshold = 0
    best_fscore = None
    best_result = None

    # thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # Need to pick better thresholds
    
    log_min = np.log10(min(counts.values()) + 1)  
    log_max = np.log10(max(counts.values()))
    thresholds = np.logspace(log_min, log_max, num=10, base=10)
    thresholds = [int(t) for t in thresholds]  

    for threshold in thresholds:
        y_pred = []
        for word in train_words:
            if counts[word] > threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        fscore = get_fscore(y_pred, train_labels)
        if best_fscore is None or fscore > best_fscore:
            best_fscore = fscore
            best_threshold = threshold
            best_result = y_pred
    
    print(f"Best frequency threshold: {best_threshold}")
    print("\nTraining data results:")
    evaluate(best_result, train_labels)
    print("\nDevelopment data results:")
    y_pred = []
    for word in dev_words:
        if counts[word] > best_threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    evaluate(y_pred, dev_labels)



### 3.1: Naive Bayes

def _get_word_length_features(words):
    """Get the length of each word."""
    length_feature = []
    for word in words:
        length_feature.append(len(word))
    return length_feature

def _get_word_frequency_features(words, counts):
    """Get the frequency of each word."""
    frequency_feature = []
    for word in words:
        frequency_feature.append(counts[word])
    return frequency_feature

def _get_feature_array(train_words, counts):
    feature_array = []
    train_length_features = _get_word_length_features(train_words)
    train_frequency_features = _get_word_frequency_features(train_words, counts)
    for i in range(len(train_length_features)):
        feature_array.append([train_length_features[i], train_frequency_features[i]])
    return np.array(feature_array)

def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """

    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)
    train_feat = _get_feature_array(train_words, counts)
    train_labels = np.array(train_labels)

    train_mean = train_feat.mean(axis=0)
    train_std = train_feat.std(axis=0)
    train_feat_normalized = (train_feat - train_mean) / train_std

    clf = GaussianNB()
    clf.fit(train_feat_normalized, train_labels)

    dev_feat = _get_feature_array(dev_words, counts)
    dev_labels = np.array(dev_labels)

    # dev_mean = dev_feat.mean(axis=0)
    # dev_std = dev_feat.std(axis=0)
    dev_feat_normalized = (dev_feat - train_mean) / train_std

    train_pred = clf.predict(train_feat_normalized)
    dev_pred = clf.predict(dev_feat_normalized)

    print("Training Set Metrics:")
    evaluate(train_pred, train_labels)
    print("\nDevelopment Set Metrics:")
    evaluate(dev_pred, dev_labels)


### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)
    train_feat = _get_feature_array(train_words, counts)
    train_labels = np.array(train_labels)

    train_mean = train_feat.mean(axis=0)
    train_std = train_feat.std(axis=0)
    train_feat_normalized = (train_feat - train_mean) / train_std

    clf = LogisticRegression()
    clf.fit(train_feat_normalized, train_labels)
    dev_feat = _get_feature_array(dev_words, counts)
    dev_labels = np.array(dev_labels)

    # dev_mean = dev_feat.mean(axis=0)
    # dev_std = dev_feat.std(axis=0)
    dev_feat_normalized = (dev_feat - train_mean) / train_std

    train_pred = clf.predict(train_feat_normalized)
    dev_pred = clf.predict(dev_feat_normalized)

    print("Training Set Metrics:")
    evaluate(train_pred, train_labels)
    print("\nDevelopment Set Metrics:")
    evaluate(dev_pred, dev_labels)


### 3.3: Build your own classifier

def _get_syllable_count_features(words):
    """Get the syllable count of each word."""
    syllable_feature = []
    for word in words:
        syllable_feature.append(count_syllables(word))
    return syllable_feature

def _get_wordnet_features(words):   
    """Get the number of synsets of each word."""
    synonym_feature = []
    for word in words:
        synonym_feature.append(len(wn.synsets(word)))
    return synonym_feature

def _get_all_features(words, counts):
    """Get all features for the given words."""
    length_feature = _get_word_length_features(words)
    frequency_feature = _get_word_frequency_features(words, counts)
    syllable_feature = _get_syllable_count_features(words)
    synonym_feature = _get_wordnet_features(words)
    feat_array = []
    for i in range(len(length_feature)):
        feat_array.append([length_feature[i], frequency_feature[i], syllable_feature[i], synonym_feature[i]])
    return np.array(feat_array)

def _init_classifier_and_results(train_feat_normalized, train_labels, dev_feat_normalized, dev_labels, clf):
    clf.fit(train_feat_normalized, train_labels)

    train_pred = clf.predict(train_feat_normalized)
    dev_pred = clf.predict(dev_feat_normalized)

    print(f"\n=============== Classifier: {clf.__class__.__name__}================")
    print("\nTraining Set Metrics:")
    evaluate(train_pred, train_labels)
    print("\nDevelopment Set Metrics:")
    evaluate(dev_pred, dev_labels)

def my_classifier(training_file, development_file, counts):
    '''Train a classifier using your own features. Print out evaluation results on the training and development data.'''
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)
    train_feat = _get_all_features(train_words, counts)
    train_labels = np.array(train_labels)

    train_mean = train_feat.mean(axis=0)
    train_std = train_feat.std(axis=0)
    train_feat_normalized = (train_feat - train_mean) / train_std

    dev_feat = _get_all_features(dev_words, counts)
    dev_labels = np.array(dev_labels)

    dev_feat_normalized = (dev_feat - train_mean) / train_std

    clf = RandomForestClassifier(random_state=20)
    _init_classifier_and_results(train_feat_normalized, train_labels, dev_feat_normalized, dev_labels, clf)
    clf = DecisionTreeClassifier(random_state=20)
    _init_classifier_and_results(train_feat_normalized, train_labels, dev_feat_normalized, dev_labels, clf)
    clf = SVM(random_state=20)
    _init_classifier_and_results(train_feat_normalized, train_labels, dev_feat_normalized, dev_labels, clf)


def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    print("-----------")
    my_classifier(training_file, development_file, counts)

def predict_test_labels(training_file, development_file, test_file, counts):
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    train_words = train_words + dev_words
    train_labels = train_labels + dev_labels

    test_words, _ = load_file(test_file)

    train_feat = _get_all_features(train_words, counts)
    train_labels = np.array(train_labels)

    train_mean = train_feat.mean(axis=0)
    train_std = train_feat.std(axis=0)
    train_feat_normalized = (train_feat - train_mean) / train_std

    test_feat = _get_all_features(test_words, counts)
    test_feat_normalized = (test_feat - train_mean) / train_std

    clf = SVM(random_state=20)
    clf.fit(train_feat_normalized, train_labels)

    test_pred = clf.predict(test_feat_normalized)

    with open('test_labels.txt', 'w') as f:
        for label in test_pred:
            f.write(str(label) + '\n')
    print("Predictions Done!\nsaved to 'test_labels.txt'")

def error_analysis(training_file, development_file, test_file, counts):
    train_words, train_labels = load_file(training_file)
    dev_words, dev_labels = load_file(development_file)

    train_words = train_words + dev_words
    train_labels = train_labels + dev_labels

    test_words, _ = load_file(test_file)

    train_feat = _get_all_features(train_words, counts)
    train_labels = np.array(train_labels)

    train_mean = train_feat.mean(axis=0)
    train_std = train_feat.std(axis=0)
    train_feat_normalized = (train_feat - train_mean) / train_std

    test_feat = _get_all_features(test_words, counts)
    test_feat_normalized = (test_feat - train_mean) / train_std

    clf = SVM(random_state=20)
    clf.fit(train_feat_normalized, train_labels)

    test_pred = clf.predict(test_feat_normalized)

    for i in range(len(test_words)):
        print(f"{test_words[i]}: {test_pred[i]}")
    

if __name__ == "__main__":
    training_file = "/var/csc483/complex_words_training.txt"
    development_file = "/var/csc483/data/complex_words_development.txt"
    test_file = "/var/csc483/data/complex_words_test_unlabeled.txt"

    print("Loading ngram counts ...")
    ngram_counts_file = "/var/csc483/ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)

    ## YOUR CODE HERE
    # Train your best classifier, predict labels for the test dataset and write
    # the predicted labels to the text file 'test_labels.txt', with ONE LABEL
    # PER LINE
    predict_test_labels(training_file, development_file, test_file, counts)

