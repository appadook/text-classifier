# Text Classifier

A comprehensive text classification toolkit for natural language processing tasks, specifically focused on:
1. Movie review sentiment classification
2. Complex word identification

## Project Overview

This project implements various text classification techniques using machine learning algorithms to solve two main tasks:

1. **Movie Review Classification**: Classifies movie reviews as positive or negative using scikit-learn's Naive Bayes.
2. **Complex Word Identification**: Identifies words that might be difficult for non-native speakers to understand.

## Features

### Evaluation Metrics
- Accuracy calculation
- Precision calculation
- Recall calculation
- F-score calculation

### Baseline Models for Complex Word Classification
- All-complex baseline (labels everything as complex)
- Word length threshold classifier
- Word frequency threshold classifier

### Advanced Classification Models
- Naive Bayes classifier
- Logistic Regression classifier
- Custom classifiers:
  - Random Forest
  - Decision Tree
  - Support Vector Machine (SVM)
  
### Feature Engineering
- Word length features
- Word frequency features (using Google NGram counts)
- Syllable count features
- WordNet-based synonym features

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd text-classifier
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Set up NLTK data:
```bash
python setup.py
```

## Usage

### Movie Review Classification

```python
python Workspace/movie_review_classification.py
```

This script:
- Loads movie reviews from NLTK corpus
- Creates feature representations for each review
- Trains a Gaussian Naive Bayes classifier
- Evaluates performance on the development set

### Complex Word Classification

```python
python Workspace/complex_word_classification.py
```

This script:
- Tests various baseline models (word length, frequency)
- Implements and evaluates Naive Bayes and Logistic Regression classifiers
- Tests custom classifiers with enhanced features
- Generates predictions for the test dataset

### Running Tests

```python
python Workspace/test/test_evaluation.py
```

## Project Structure

- `Workspace/`
  - `evaluation.py`: Implementation of evaluation metrics
  - `complex_word_classification.py`: Complex word identification algorithms
  - `movie_review_classification.py`: Movie review sentiment classification
  - `syllables.py`: Utility for counting syllables
  - `test/`: Unit tests for the project

## Data

The project uses:
- NLTK movie review corpus
- Custom complex word datasets (training, development, and test)
- Google NGram counts for word frequency features

## License

[Your license information here]

## Author

- Kurtik Appadoo