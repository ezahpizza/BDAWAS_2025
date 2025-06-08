1. filter()
# Filter rows where Value > 300 and Active is True
filtered_df = df.filter((col("Value") > 300) & (col("Active") == True))

spark.sql("""
        SELECT celsius,
        filter(celsius, t -> t > 38) AS high
        FROM tC
        """).show(truncate=False)

2. select()
selected_df = df.select("Value", "Title")

3. Map  + Expr
mapped_df = df.selectExpr("Value * 2 as ValueDoubled", "Title")

4. groupBy()
grouped_df = df.groupBy("Active").agg(count("Title").alias("TitleCount"))

5. first()
first_row = df.first()
print(first_row)

6. transform()
spark.sql("""
        SELECT celsius,
        transform(celsius, t -> ((t * 9) DIV 5) + 32) AS fahrenheit
        FROM tC
        """).show(truncate=False) 

7. exists() 
spark.sql("""
        SELECT celsius,
        exists(celsius, t -> t = 38) AS threshold
        FROM tC
        """).show(truncate=False) 

8. reduce() 
spark.sql("""
        SELECT celsius,
        reduce(
        celsius,
        0,
        (acc, t) -> acc + t,
        acc -> (acc DIV size(celsius) * 9 DIV 5) + 32
        ) AS avgFahrenheit
        FROM tC
        """).show(truncate=False) 

9. array_zip() 
data = [([1, 2], [2, 3], [3, 4])]
df = spark.createDataFrame(data, ["array1", "array2", "array3"])
df.createOrReplaceTempView("arrays")
spark.sql("""
        SELECT arrays_zip(array1, array2, array3) AS zipped
        FROM arrays
        """).show(truncate=False)

10.  array_union() 
spark.sql("""
        SELECT array_union(array(1, 2, 3), array(3, 4, 5)) AS union_array
        """).show(truncate=False)

11. . array_except() 
spark.sql("""
        SELECT array_except(array(1, 2, 3), array(1, 3, 5)) AS except_array
        """).show(truncate=False) 

12. sequence() 
spark.sql("""
        SELECT sequence(1, 5) AS sequence_array
        """).show(truncate=False)

13. flatten() 
spark.sql("""
        SELECT flatten(array(array(1, 2), array(3, 4))) AS flat_array
        """).show(truncate=False) 

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
