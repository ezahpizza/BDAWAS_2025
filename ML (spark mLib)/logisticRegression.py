from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Create Spark session
spark = SparkSession.builder \
    .appName("Logistic Regression with Pipeline") \
    .getOrCreate()

# Sample data
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
columns = ["feature1", "feature2", "feature3", "label"]
data_df = spark.createDataFrame(data, columns)

# Feature assembler
feature_cols = ['feature1', 'feature2', 'feature3']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Split data
train_data, test_data = data_df.randomSplit([0.8, 0.2], seed=42)

# Fit pipeline
pipeline_model = pipeline.fit(train_data)

# Make predictions
predictions = pipeline_model.transform(test_data)

# Show predictions
predictions.select("features", "label", "prediction", "probability").show()

# Evaluate predictions
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Logistic Regression Accuracy: {accuracy}")
