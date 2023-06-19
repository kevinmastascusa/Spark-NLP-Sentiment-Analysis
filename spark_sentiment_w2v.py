#spark_sentiment_w2v.py

"""
Project Title: Spark-Sentiment-Analysis
Description: A sentiment analysis project using Apache Spark for large-scale text data processing.
Author: Kevin Mastascusa
Date: 2023-06-19
"""

# Import the required libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder.appName("Spark-NLP-Sentiment-Analysis").getOrCreate()

"""
SparkSession is the entry point to interact with Apache Spark and provides a way to create and configure
Spark functionality. It allows you to create DataFrames, perform distributed computations, and execute
machine learning tasks on large-scale data using Spark's distributed computing capabilities.
"""

# Load the sentiment dataset as a DataFrame
data = spark.read.option("header", "true").csv("sentiment_dataset.csv")

"""
DataFrame is a distributed collection of data organized into named columns. It represents a structured data
set and can be manipulated using Spark's DataFrame API. In this project, the sentiment dataset is loaded into
a DataFrame, where each row represents a sample and each column represents a feature or label.
"""

# Data Preparation
# Tokenize text into individual words using RegexTokenizer
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")

"""
RegexTokenizer is a Transformer that splits text into individual words based on a specified regular expression
pattern. In this project, it is used to tokenize the text data from the 'text' column and output the tokenized
words into a new column called 'words'.
"""

# Remove common stopwords from the words using StopWordsRemover
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

"""
StopWordsRemover is a Transformer that removes common stopwords (e.g., 'and', 'the', 'is') from a given set of
words. It helps eliminate words that do not contribute much to the sentiment analysis task. The output of the
RegexTokenizer is fed into the StopWordsRemover, and the filtered words are stored in a new column called
'filtered_words'.
"""

# Generate word embeddings using Word2Vec to convert filtered words into fixed-sized feature vectors
word2vec = Word2Vec(vectorSize=100, inputCol="filtered_words", outputCol="features")

"""
Word2Vec is an Estimator that maps words from a given corpus into fixed-sized vectors. It captures semantic
similarities between words and represents them as dense numerical vectors. In this project, it is used to generate
word embeddings from the filtered words in the 'filtered_words' column and store the resulting feature vectors
in a new column called 'features'.
"""

# Prepare the label column for binary classification using StringIndexer
label_indexer = StringIndexer(inputCol="sentiment", outputCol="label")

"""
StringIndexer is an Estimator that encodes string labels into numerical values. It assigns a unique numerical index
to each distinct label in a given column. In this project, it is used to convert the sentiment labels in the 'sentiment'
column into numerical labels and store them in a new column called 'label'.
"""

# Define the pipeline stages
pipeline_stages = [tokenizer, stopwords_remover, word2vec, label_indexer]

"""
A pipeline is a sequence of stages, where each stage represents a transformation or an estimator. It allows you to
organize and chain multiple data processing steps together. In this project, the pipeline stages include tokenization,
stopwords removal, word embedding generation, and label indexing. These stages will be executed in order to transform
the input data and train the sentiment analysis model.
"""

# Create a pipeline with the defined stages
pipeline = Pipeline(stages=pipeline_stages)

# Fit the pipeline to the data and transform the data
transformed_data = pipeline.fit(data).transform(data)

"""
Fitting the pipeline to the data involves training the pipeline by executing each stage on the input DataFrame and
creating a fitted pipeline model. Then, transforming the data applies the fitted pipeline model to the input DataFrame,
performing all the defined transformations and generating the transformed DataFrame.
"""

# Split the transformed data into training and testing sets
train_data, test_data = transformed_data.randomSplit([0.8, 0.2], seed=42)

"""
Splitting the data is a common practice in machine learning to separate the dataset into training and testing subsets.
The transformed data is split into training and testing sets, with 80% for training and 20% for testing. The seed is set
for reproducibility.
"""

# Train a Logistic Regression model
lr = LogisticRegression(labelCol="label", featuresCol="features")
lr_model = lr.fit(train_data)

"""
LogisticRegression is a classification algorithm that models the relationship between the features and the target
variable using logistic functions. In this project, a logistic regression model is trained using the labeled training
data, where 'labelCol' specifies the column with the target variable and 'featuresCol' specifies the column with the
input features.
"""

# Make predictions on the test data
predictions = lr_model.transform(test_data)

"""
Making predictions involves applying the trained model to the test data to obtain predicted labels for the samples.
The 'transform' method of the logistic regression model is used to apply the model to the test data and generate
predictions.
"""

# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
auc = evaluator.evaluate(predictions)

"""
BinaryClassificationEvaluator is an evaluator that measures the performance of a binary classification model. It
computes the area under the receiver operating characteristic (ROC) curve (AUC) as a metric to evaluate the model's
ability to distinguish between positive and negative samples. In this project, the evaluator is applied to the
predictions DataFrame to calculate the AUC.
"""

# Print the Area Under the ROC curve (AUC)
print("Area Under the ROC Curve (AUC):", auc)

# To-Do List:
# 1. Explore other advanced feature extraction techniques like TF-IDF with n-grams or Doc2Vec.
# 2. Experiment with different classification algorithms in Spark MLlib.
# 3. Fine-tune hyperparameters of the chosen algorithm for optimal performance.
# 4. Investigate additional evaluation metrics such as precision, recall, or F1 score.
# 5. Visualize the results, such as generating ROC curves or confusion matrices.
# 6. Handle unbalanced datasets using techniques like oversampling or undersampling.
# 7. Incorporate cross-validation for robust model evaluation and parameter selection.
# 8. Document and organize your code to improve readability and maintainability.
# 9. Explore techniques for model deployment and serving in a production environment.
# 10. Stay updated with the latest advancements in Apache Spark and NLP techniques.
