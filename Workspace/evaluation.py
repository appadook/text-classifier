"""Evaluation Metrics

Author: Kristina Striegnitz and Kurtik Appadoo

I affirm that I will carry out my academic endeavors with full academic honesty, and I rely on my fellow students to do the same.

Complete this file for part 1 of the project.
"""

def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    accuracy = correct / len(y_pred)
    return accuracy

def get_precision(y_pred, y_true, label=1):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    true_positive = 0
    total = 0
    for i in range(len(y_true)):
        if y_pred[i] == label and y_true[i] == label:
            true_positive += 1
        if y_pred[i] == label:
            total += 1
    precision = true_positive / total if total > 0 else 0
    return precision

def get_recall(y_pred, y_true, label=1):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    true_positive = 0
    actual_positive = 0
    for i in range(len(y_true)):
        if y_true[i] == label:
            actual_positive += 1
            if y_pred[i] == label:
                true_positive += 1
    if actual_positive == 0:
        return 0
    recall = true_positive / actual_positive
    return recall


def get_fscore(y_pred, y_true, label=1):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    precision = get_precision(y_pred, y_true, label)
    recall = get_recall(y_pred, y_true, label)
    if precision + recall == 0:
        return 0
    fscore = 2 * precision * recall / (precision + recall)
    return fscore


def evaluate(y_pred, y_true, label=1):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    accuracy = get_accuracy(y_pred, y_true)
    precision = get_precision(y_pred, y_true, label)
    recall = get_recall(y_pred, y_true, label)
    fscore = get_fscore(y_pred, y_true, label)
    
    print(f"Accuracy: {accuracy * 100:.0f}%")
    print(f"Precision: {precision * 100:.0f}%")
    print(f"Recall: {recall * 100:.0f}%")
    print(f"F-score: {fscore * 100:.0f}%")

