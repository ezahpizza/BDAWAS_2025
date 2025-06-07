# Define schema programmatically
schema = StructType([
    StructField("author", StringType(), False),
    StructField("title", StringType(), False),
    StructField("pages", IntegerType(), False)
])

# Define schema using DDL
schema = "author STRING, title STRING, pages INT"

1. Projections and Filters
few_fire_df = fire_df.select("IncidentNumber", "AvailableDtTm", "CallType") \
                      .where(col("CallType") != "Medical Incident")
few_fire_df.show(5, truncate=False)

2. Counting Distinct Values
fire_df.select("CallType") \
       .where(col("CallType").isNotNull()) \
       .agg(countDistinct("CallType").alias("DistinctCallTypes")) \
       .show()

3. Renaming and Modifying Columns
new_fire_df = fire_df.withColumnRenamed("Delay", "ResponseDelayedinMins")
new_fire_df.select("ResponseDelayedinMins").where(col("ResponseDelayedinMins") > 5).show(5)

4. Date and Time Transformations
fire_ts_df = new_fire_df.withColumn("IncidentDate", to_timestamp(col("CallDate"), "MM/dd/yyyy")) \
                        .drop("CallDate") \
                        .withColumn("OnWatchDate", to_timestamp(col("WatchDate"), "MM/dd/yyyy")) \
                        .drop("WatchDate") \
                        .withColumn("AvailableDtTS", to_timestamp(col("AvailableDtTm"), "MM/dd/yyyy hh:mm:ss a")) \
                        .drop("AvailableDtTm")
fire_ts_df.select("IncidentDate", "OnWatchDate", "AvailableDtTS").show(5, truncate=False)

5. Aggregations
fire_ts_df.select("CallType") \
          .where(col("CallType").isNotNull()) \
          .groupBy("CallType") \
          .count() \
          .orderBy("count", ascending=False) \
          .show(10, truncate=False)


