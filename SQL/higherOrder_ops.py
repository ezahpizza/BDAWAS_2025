1. Filter Operation
# Filter rows where Value > 300 and Active is True
filtered_df = df.filter((col("Value") > 300) & (col("Active") == True))

2. Select Operation
selected_df = df.select("Value", "Title")

3. Map Equivalent using selectExpr
mapped_df = df.selectExpr("Value * 2 as ValueDoubled", "Title")

4. GroupBy Operation
grouped_df = df.groupBy("Active").agg(count("Title").alias("TitleCount"))

7. First Operation
Concept
The first() method returns the first row of the DataFrame.
Code:
# Get the first row
first_row = df.first()
print(first_row)


+--------------+--------------------------+--------------------------+-----
SPARK SQL

1. #Create a temporary view for SQL queries
df.createOrReplaceTempView("us_delay_flights_tbl")
#Select flights with distance > 1000 miles
spark.sql("""
 SELECT distance, origin, destination
 FROM us_delay_flights_tbl
 WHERE distance > 1000
 ORDER BY distance DESC
""").show(10)


2. #Flights from SFO to ORD with delays > 120 minutes
        spark.sql("""
        SELECT date, delay, origin, destination
        FROM us_delay_flights_tbl
        WHERE delay > 120 AND origin = 'SFO' AND destination = 'ORD'
        ORDER BY delay DESC
        """).show(10)

W
3. #Classify flights based on delay durations
spark.sql("""
            SELECT delay, origin, destination,
            CASE
            WHEN delay > 360 THEN 'Very Long Delays'
            WHEN delay > 120 AND delay <= 360 THEN 'Long Delays'
            WHEN delay > 60 AND delay <= 120 THEN 'Short Delays'
            WHEN delay > 0 AND delay <= 60 THEN 'Tolerable Delays'
            WHEN delay = 0 THEN 'No Delays'
            ELSE 'Early'
            END AS Flight_Delays
            FROM us_delay_flights_tbl
            ORDER BY origin, delay DESC
            """).show(10)
