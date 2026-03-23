# Email Spam Detection using Naive Bayes

## Overview
This project implements a custom Naive Bayes classifier from scratch in Python to categorize emails as either "spam" or "ham". It provides a foundational look at natural language processing and probabilistic machine learning by calculating the probability of an email belonging to a specific class based on the occurrence of predefined keywords.

## How It Works
The classification process is broken down into three main steps: preprocessing, training, and prediction.

* **Text Preprocessing (`text_to_vector`):** The input email text is converted to lowercase, and all punctuation is removed. The text is then transformed into a binary feature vector of length 10, indicating the presence (1) or absence (0) of specific vocabulary words.
* **Training the Model (`train_naive_bayes`):** The algorithm calculates the class priors (the overall probability of an email being spam or ham in the dataset). It then calculates the conditional probabilities for each word given the class. To prevent zero-probability errors during multiplication, the model utilizes Laplace smoothing by adding 1 to the numerator and 2 to the denominator.
* **Making Predictions (`predict`):** To classify a new email, the model determines `P(spam|email)` and `P(ham|email)`. It achieves this by multiplying the class prior by the conditional probabilities of the vocabulary words (using `1 - conditional_probability` if the word is absent). The class with the highest resulting score is chosen as the prediction.

## Features (Vocabulary)
Instead of processing every word in the English language, this lightweight model uses a targeted vocabulary of 10 common keywords often associated with spam or business correspondence. 
The vocabulary consists of: `["free", "win", "cash", "prize", "offer", "meeting", "project", "report", "account", "lunch"]`.

## Dataset
For demonstration purposes, the notebook relies on a small, hardcoded dataset:
* **Training Data:** 12 sample emails evenly split between spam and ham categories.
* **Test Data:** 6 unseen sample emails used to evaluate the model's predictions.

## Usage
To use this project:
1. Open `spam_detection.ipynb` in a Jupyter Notebook environment.
2. Run the cells sequentially.
3. The main program will execute, displaying the learned vocabulary, priors, and conditional probabilities, followed by the predicted labels and probability scores for the test emails.
