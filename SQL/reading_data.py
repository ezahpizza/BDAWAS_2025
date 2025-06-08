DataFrameReader
1. Reading a Parquet File

df = spark.read.format("parquet").load("path/to/parquet/file")
________________________________________
2. Reading a CSV File with Options
df = spark.read.format("csv") \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("mode", "PERMISSIVE") \
    .load("path/to/csv/file")
________________________________________
3. Reading a JSON File
df = spark.read.format("json").load("path/to/json/file")
df.show()
________________________________________
4. Reading with a Defined Schema

df = spark.read.format("csv") \
    .schema(schema) \
    .option("header", "true") \
    .load("path/to/csv/file")
________________________________________
8. Using load()
df = spark.read.format("json").load("path/to/json/file")

DataFrameWriter
9. Using format()
df.write.format("json").save("output/json_data")
________________________________________
10. Using option()
df.write.format("csv") \
    .option("header", "true") \
    .option("mode", "overwrite") \
    .save("output/csv_data")
________________________________________
11. Using bucketBy()
df.write.format("parquet") \
    .bucketBy(4, "Age") \
    .mode("overwrite") \
    .save("output/bucketed_data")
________________________________________
12. Using save()
df.write.format("parquet").save("output/parquet_data")
________________________________________
13. Using saveAsTable()
df.write.format("parquet") \
    .mode("overwrite") \
    .saveAsTable("people_table")