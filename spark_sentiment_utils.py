"""
File Name: spark_sentiment_utils.py
Description: This file contains utility functions for data processing and model evaluation in the Spark-Sentiment-Analysis project.
Author: Kevin Mastascusa
Date: 2023-06-19
"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def preprocess_data(data):
    """
    Preprocesses the input data by applying tokenization, stop word removal, and word embedding generation using Word2Vec.
    Returns the transformed data.
    """
    tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    word2vec = Word2Vec(vectorSize=100, inputCol="filtered_words", outputCol="features")
    
    pipeline_stages = [tokenizer, stopwords_remover, word2vec]
    pipeline = Pipeline(stages=pipeline_stages)
    transformed_data = pipeline.fit(data).transform(data)
    
    return transformed_data

def train_model(train_data):
    """
    Trains a logistic regression model on the given training data.
    Returns the trained model.
    """
    lr = LogisticRegression(labelCol="label", featuresCol="features")
    lr_model = lr.fit(train_data)
    
    return lr_model

def evaluate_model(model, test_data):
    """
    Evaluates the trained model on the given test data and returns the AUC.
    """
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    auc = evaluator.evaluate(predictions)
    
    return auc

# Additional tests

def test_preprocess_data():
    """
    Test the preprocess_data function.
    """
    # Add test cases here
    pass

def test_train_model():
    """
    Test the train_model function.
    """
    # Add test cases here
    pass

def test_evaluate_model():
    """
    Test the evaluate_model function.
    """
    # Add test cases here
    pass

if __name__ == "__main__":
    # Run the tests
    test_preprocess_data()
    test_train_model()
    test_evaluate_model()
