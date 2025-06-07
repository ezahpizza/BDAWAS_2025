#1. Linear Regression
# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
# Create Spark session
spark = SparkSession.builder \
 .appName("Linear Regression Example") \
 .getOrCreate()
# Create sample data
data = [
 (1.5, 3.0, 2.5, 10.0),
 (2.0, 4.1, 3.0, 15.0),
 (1.8, 3.3, 2.8, 12.0),
 (3.0, 6.0, 4.5, 25.0),
 (2.5, 5.0, 3.5, 20.0),
 (3.5, 7.2, 5.0, 30.0),
 (1.2, 2.5, 2.0, 8.0),
 (2.8, 5.8, 3.8, 22.0)
]
# Define schema and create DataFrame
columns = ["feature1", "feature2", "feature3", "target_column"]
data_df = spark.createDataFrame(data, columns)
# Show data preview
data_df.show()
# Prepare data for Linear Regression
feature_cols = ['feature1', 'feature2', 'feature3']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# Transform the dataset to include feature vector
assembled_data = assembler.transform(data_df)
# Select features and target column
final_data = assembled_data.select("features", "target_column")
# Split data into training and testing sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
# Create and train Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="target_column")
lr_model = lr.fit(train_data)
# Evaluate the model
test_results = lr_model.evaluate(test_data)
print(f"R2: {test_results.r2}")
print(f"RMSE: {test_results.rootMeanSquaredError}")
print(f"MAE: {test_results.meanAbsoluteError}")
# Make predictions on test data
predictions = lr_model.transform(test_data)
predictions.select("features", "target_column", "prediction").show()


+--------+--------+--------+-------------+
#2. Logicstic Regression
# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Create Spark session
spark = SparkSession.builder.appName("Logistic Regression Example").getOrCreate()
# Create sample data for classification
data = [
 (1.5, 3.0, 2.5, 0),
 (2.0, 4.1, 3.0, 1),
 (1.8, 3.3, 2.8, 0),
 (3.0, 6.0, 4.5, 1),
 (2.5, 5.0, 3.5, 1),
 (3.5, 7.2, 5.0, 1),
 (1.2, 2.5, 2.0, 0),
 (2.8, 5.8, 3.8, 1)
]
# Define schema and create DataFrame
columns = ["feature1", "feature2", "feature3", "label"]
data_df = spark.createDataFrame(data, columns)
# Prepare data for Logistic Regression
feature_cols = ['feature1', 'feature2', 'feature3']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(data_df).select("features", "label")
# Split data into training and testing sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
# Create and train Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)
# Make predictions on test data
predictions = lr_model.transform(test_data)
predictions.select("features", "label", "prediction").show()
# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Logistic Regression Accuracy: {accuracy}")


+--------+--------+--------+-------------+
#3. Decision Tree Classifier
# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Create Spark session
spark = SparkSession.builder \
 .appName("Decision Tree Classifier Example") \
 .getOrCreate()
# Create sample data for classification
data = [
 (1.5, 3.0, 2.5, 0),
 (2.0, 4.1, 3.0, 1),
 (1.8, 3.3, 2.8, 0),
 (3.0, 6.0, 4.5, 1),
 (2.5, 5.0, 3.5, 1),
 (3.5, 7.2, 5.0, 1),
 (1.2, 2.5, 2.0, 0),
 (2.8, 5.8, 3.8, 1)
]
# Define schema and create DataFrame
columns = ["feature1", "feature2", "feature3", "label"]
data_df = spark.createDataFrame(data, columns)
# Show data preview
data_df.show()
# Prepare data for Decision Tree Classifier
feature_cols = ['feature1', 'feature2', 'feature3']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# Transform the dataset to include feature vector
assembled_data = assembler.transform(data_df)
# Select features and target column
final_data = assembled_data.select("features", "label")
# Split data into training and testing sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
# Create and train Decision Tree Classifier
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_model = dt.fit(train_data)
# Make predictions on test data
predictions = dt_model.transform(test_data)
# Show prediction results
predictions.select("features", "label", "prediction").show()
# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
 labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")


+-------------+-----+----------+
#4. Random Forest Classification
# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Create Spark session
spark = SparkSession.builder.appName("Random Forest Classifier").getOrCreate()
# Create sample data
data = [
 (1.5, 3.0, 2.5, 0),
 (2.0, 4.1, 3.0, 1),
 (1.8, 3.3, 2.8, 0),
 (3.0, 6.0, 4.5, 1),
 (2.5, 5.0, 3.5, 1),
 (3.5, 7.2, 5.0, 1),
 (1.2, 2.5, 2.0, 0),
 (2.8, 5.8, 3.8, 1)
]
# Define schema and create DataFrame
columns = ["feature1", "feature2", "feature3", "label"]
data_df = spark.createDataFrame(data, columns)
# Prepare data for Random Forest Classifier
feature_cols = ['feature1', 'feature2', 'feature3']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(data_df).select("features", "label")
# Split data into training and testing sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
# Create and train Random Forest model
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
rf_model = rf.fit(train_data)
# Make predictions on test data
predictions = rf_model.transform(test_data)
predictions.select("features", "label", "prediction").show()
# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Random Forest Accuracy: {accuracy}")


+-------------+-----+----------+
#5. Naive Bayes Classification
# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Create Spark session
spark = SparkSession.builder.appName("Naive Bayes Classifier").getOrCreate()
# Create sample data
data = [
 (1.5, 3.0, 2.5, 0),
 (2.0, 4.1, 3.0, 1),
 (1.8, 3.3, 2.8, 0),
 (3.0, 6.0, 4.5, 1),
 (2.5, 5.0, 3.5, 1),
 (3.5, 7.2, 5.0, 1),
 (1.2, 2.5, 2.0, 0),
 (2.8, 5.8, 3.8, 1)
]
# Define schema and create DataFrame
columns = ["feature1", "feature2", "feature3", "label"]
data_df = spark.createDataFrame(data, columns)
# Prepare data for Naive Bayes Classifier
feature_cols = ['feature1', 'feature2', 'feature3']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(data_df).select("features", "label")
# Split data into training and testing sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
# Create and train Naive Bayes model
nb = NaiveBayes(featuresCol="features", labelCol="label")
nb_model = nb.fit(train_data)
# Make predictions on test data
predictions = nb_model.transform(test_data)
predictions.select("features", "label", "prediction").show()
# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Naive Bayes Accuracy: {accuracy}")


+-------------+-----+----------+
# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Create Spark session
spark = SparkSession.builder.appName("SVM Classifier").getOrCreate()
# Create sample data
data = [
 (1.5, 3.0, 2.5, 0),
 (2.0, 4.1, 3.0, 1),
 (1.8, 3.3, 2.8, 0),
 (3.0, 6.0, 4.5, 1),
 (2.5, 5.0, 3.5, 1),
 (3.5, 7.2, 5.0, 1),
 (1.2, 2.5, 2.0, 0),
 (2.8, 5.8, 3.8, 1)
]
# Define schema and create DataFrame
columns = ["feature1", "feature2", "feature3", "label"]
data_df = spark.createDataFrame(data, columns)
# Prepare data for SVM Classifier
feature_cols = ['feature1', 'feature2', 'feature3']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_data = assembler.transform(data_df).select("features", "label")
# Split data into training and testing sets
train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)
# Create and train SVM model
svm = LinearSVC(featuresCol="features", labelCol="label")
svm_model = svm.fit(train_data)
# Make predictions on test data
predictions = svm_model.transform(test_data)
predictions.select("features", "label", "prediction").show()
# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"SVM Accuracy: {accuracy}")
