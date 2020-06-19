from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from functools import reduce
import math


def preprocess_data(df):
    """
    Function that performs multiple steps of preprocessing for the 
    CO2-emissions data.
    """

    # Drop the first two columns, since they are not needed
    df = df.drop("Series Name").drop("Series Code")

    # Rename the columns corresponding to the years by removing
    # the part in the square brackets. E.g., "2004 [YR2004]" --> "2004"
    old_column_names = df.schema.names
    new_column_names = df.schema.names[:2] + [col.split(" ")[0] for col in df.schema.names[2:]]
    df = reduce(lambda data, idx: data.withColumnRenamed(old_column_names[idx], new_column_names[idx]), range(len(old_column_names)), df)

    # Null values are represented by ".."
    # 1. drop all rows for which there is NO DATA at all, i.e., all values are NaN
    # Of course, we could also replace them by the mean or median of all other countries 
    # per column, but I don't think it makes a lot of sense, since the variations are high
    df = df.na.drop("all", subset=[col for col in df.schema.names[2:]])

    # 2. Using the fill function to impute missing values
    year_cols = df.columns[2:]
    df = df.withColumn('mean', (sum([df[i] for i in year_cols]) / len(year_cols))).na.fill('mean', subset=df.columns[:-1]).select(df.columns[:2] + year_cols)

    return df

def main():
    print("HI")
    # set configuration
    

    # 1. import the data
    co2_data = sqlContext.read.option("inferSchema", "true").option("header", "true").option("nanValue", "..").csv("data/co2_emissions_2004-2014.csv")
    co2_data = preprocess_data(co2_data)
    #co2_data.write.option("header", "true").csv("test")
    #co2_data.show(5)

if __name__ == "__main__":
    conf = SparkConf().setAppName("ExamAdmission")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)

    main()
