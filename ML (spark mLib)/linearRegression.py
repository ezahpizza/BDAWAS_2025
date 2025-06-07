from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder \
    .appName("Linear Regression with Pipeline") \
    .getOrCreate()

# Sample data
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
columns = ["feature1", "feature2", "feature3", "target_column"]
data_df = spark.createDataFrame(data, columns)

# Define feature assembler
feature_cols = ['feature1', 'feature2', 'feature3']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Define linear regression model
lr = LinearRegression(featuresCol="features", labelCol="target_column")

# Create a pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Split data into training and testing sets
train_data, test_data = data_df.randomSplit([0.8, 0.2], seed=42)

# Train the pipeline model
pipeline_model = pipeline.fit(train_data)

# Make predictions
predictions = pipeline_model.transform(test_data)

# Evaluate the model
lr_model = pipeline_model.stages[-1]  # Extract the LinearRegressionModel from pipeline
evaluation = lr_model.evaluate(predictions)

# Print evaluation metrics
print(f"R2: {evaluation.r2}")
print(f"RMSE: {evaluation.rootMeanSquaredError}")
print(f"MAE: {evaluation.meanAbsoluteError}")

# Show predictions
predictions.select("features", "target_column", "prediction").show()
