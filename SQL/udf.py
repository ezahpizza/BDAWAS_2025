from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import LongType
# Create Spark session
spark = SparkSession.builder.appName("UDF Example").getOrCreate()
# Define a Python function to cube a number
def cubed(s):
 return s * s * s
# Register UDF in Spark SQL
spark.udf.register("cubed", cubed, LongType())
# Create a DataFrame with values from 1 to 8
df = spark.range(1, 9)
# Register the DataFrame as a temporary SQL view
df.createOrReplaceTempView("udf_test")
# Use SQL query to apply the UDF
df_result = spark.sql("SELECT id, cubed(id) AS id_cubed FROM udf_test")
# Show the output
df_result.show()

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, when
from pyspark.sql.types import IntegerType, StringType
# Create a Spark session
spark = SparkSession.builder.appName("Null Checking Example").getOrCreate()
# Sample DataFrame with NULL values
data = [(1, "apple"), (2, None), (3, "banana"), (4, None)]
df = spark.createDataFrame(data, ["id", "s"])
# Define a UDF to check for nulls before applying string length
@udf(IntegerType())
def safe_strlen(s):
 return len(s) if s is not None else None
# Using SQL-style null check with CASE WHEN
df = df.withColumn("length", when(df.s.isNotNull(), safe_strlen(df.s)))
# Show the output
df.show()

2. Pandas UDFs
import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType
# Create a Spark session
spark = SparkSession.builder.appName("Pandas UDF Example").getOrCreate()
# Define a Pandas UDF (Vectorized UDF)
@pandas_udf(LongType())
def cubed(a: pd.Series) -> pd.Series:
 return a * a * a
# Create a Spark DataFrame
df = spark.range(1, 4)
# Apply the Pandas UDF
df_result = df.select("id", cubed(col("id")).alias("id_cubed"))
# Show the output
df_result.show()

3. Creating and Querying Tables in Spark SQL
spark.sql("CREATE TABLE IF NOT EXISTS people (name STRING, age INT)")
spark.sql("INSERT INTO people VALUES ('Michael', NULL)")
spark.sql("INSERT INTO people VALUES ('Andy', 30)")
spark.sql("INSERT INTO people VALUES ('Samantha', 19)")
spark.sql("SHOW TABLES").show()
spark.sql("SELECT * FROM people WHERE age < 20").show()
spark.sql("SELECT name FROM people WHERE age IS NULL").show()
