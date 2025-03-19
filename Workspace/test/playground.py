import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from complex_word_classification import all_complex, word_length_threshold, word_frequency_threshold, load_ngram_counts, naive_bayes, logistic_regression, my_classifier, classifiers, baselines, predict_test_labels, error_analysis

if __name__ == '__main__':
    # Define paths relative to project root
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    training_file = os.path.join(base_path, 'data', 'complex_words_training.txt')
    development_file = os.path.join(base_path, 'data', 'complex_words_development.txt')
    test_file = os.path.join(base_path, 'data', 'complex_words_test_unlabeled.txt')
    ngram_file = os.path.join(base_path,'Ngram Counts.txt.gz')

    # Verify files exist
    for f in [training_file, development_file, test_file, ngram_file]:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            sys.exit(1)

    # Load ngram counts and run classifier
    print("Loading ngram counts ...")
    counts = load_ngram_counts(ngram_file)

    # all_complex(training_file)
    word_length_threshold(training_file, development_file)
    # word_frequency_threshold(training_file, development_file, counts)

    # baselines(training_file, development_file, counts)
    # classifiers(training_file, development_file, counts)

    # print("\n========== Classifiers ===========\n")

    # print("Naive Bayes")
    # print("-----------")
    # naive_bayes(training_file, development_file, counts)

    # print("\nLogistic Regression")
    # print("-----------")
    # logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    my_classifier(training_file, development_file, counts)
    
    # predict_test_labels(training_file, development_file, test_file, counts)
    # error_analysis(training_file, development_file,test_file, counts)