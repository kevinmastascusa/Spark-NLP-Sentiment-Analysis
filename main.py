"""
Project Title: Spark-Sentiment-Analysis
Description: A sentiment analysis project using Apache Spark for large-scale text data processing.
Author: Kevin Mastascusa
Date: 2023-06-19
"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder.appName("Spark-NLP-Sentiment-Analysis").getOrCreate()

# Load the sentiment dataset as a DataFrame
data = spark.read.option("header", "true").csv("sentiment_dataset.csv")

# Data Preparation
# Tokenize text into individual words
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
# Remove common stopwords from words
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
# Vectorize the filtered words into a feature vector
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
# Apply IDF transformation to the feature vector
idf = IDF(inputCol="raw_features", outputCol="features")
# Prepare the label column for binary classification
label_indexer = StringIndexer(inputCol="sentiment", outputCol="label")
# Define the pipeline stages
pipeline_stages = [tokenizer, stopwords_remover, vectorizer, idf, label_indexer]

# Create a pipeline
pipeline = Pipeline(stages=pipeline_stages)

# Fit the pipeline to the data
pipeline_model = pipeline.fit(data)

# Transform the data
transformed_data = pipeline_model.transform(data)

# Split the data into training and testing sets
train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=42)

# Train a Logistic Regression model
lr = LogisticRegression(labelCol="label", featuresCol="features")
lr_model = lr.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)

# Print the Area Under the ROC curve (AUC)
print("Area Under the ROC Curve (AUC):", auc)

# To-Do List:
# 1. Experiment with different feature extraction techniques, such as word embeddings or TF-IDF with n-grams.
# 2. Explore other classification algorithms in Spark MLlib, such as Random Forest or Gradient-Boosted Trees.
# 3. Fine-tune the hyperparameters of the chosen algorithm to optimize performance.
# 4. Investigate additional evaluation metrics, such as precision, recall, or F1 score.
# 5. Visualize the results, such as generating ROC curves or confusion matrices, to gain insights into model performance.
# 6. Handle unbalanced datasets using techniques like oversampling or undersampling.
# 7. Incorporate cross-validation for robust model evaluation and parameter selection.
# 8. Document and organize your code, including providing clear explanations of the NLP techniques used.
# 9. Showcase scalability by processing larger datasets or using Spark clusters.
# 10. Stay updated with the latest advancements in Apache Spark and NLP techniques.

