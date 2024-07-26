# Fake-news-detection-using-python
# Fake News Detection

This project focuses on detecting fake news using a Passive-Aggressive Classifier and TfidfVectorizer. The aim is to build a machine learning model that can accurately classify news articles as real or fake based on their content.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Fake news has become a major issue in recent times, spreading misinformation and causing various negative impacts. This project aims to tackle this problem by using machine learning techniques to detect and classify fake news articles. We use the TfidfVectorizer to convert text data into numerical data and a Passive-Aggressive Classifier to make predictions.

## Dataset

The dataset used for this project contains news articles labeled as real or fake. It includes features such as the title, text, subject, and date of the articles.
The dataset can be downloaded from [here](https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)

## Installation

To run this project, you need to have Python installed along with some necessary libraries. Follow the steps below to set up the environment:

1. Clone the repository:
    ```sh
    git clone https://github.com/nakshatraaditya/fake-news-detection.git
    cd fake-news-detection
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the data:
    ```python
    # Load the data
    import pandas as pd

    df = pd.read_csv('path/to/your/dataset.csv')

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    ```

2. Vectorize the text data using TfidfVectorizer:
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    ```

3. Train the Passive-Aggressive Classifier:
    ```python
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.metrics import accuracy_score

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # Predict on the test set
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')
    ```

4. Evaluate the model:
    ```python
    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    ```

## Results

The model achieves an accuracy of approximately **92.66%** on the test data. Further evaluation metrics such as precision, recall, and F1-score are provided in the classification report.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/yourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/yourFeature`)
5. Open a pull request

