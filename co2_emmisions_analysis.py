from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from functools import reduce
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import geoplot
import geopandas
import requests
import matplotlib.pyplot as plt

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

    # 2. Using the fill function to impute missing values with the mean of the country
    year_cols = df.columns[2:]
    df = df.withColumn('mean', (sum([df[i] for i in year_cols]) / len(year_cols))).na.fill('mean', subset=df.columns[:-1]).select(df.columns[:2] + year_cols)

    # drop all years but 2004 and 2014, because they are not needed for the analysis
    df = df.select(df.columns[:2] + ["2004", "2014"])

    return df

def compute_emission_change(row):
    """
    For each country (i.e., for each row) compute the change between each year  
    """
    change = (row["2014"] - row["2004"]) / row["2004"]
    return change

def compute_k_means(df, k):
    va = VectorAssembler().setInputCols(["change"]).setOutputCol("features")

    new_df = va.setHandleInvalid("skip").transform(df)
    kmeans = KMeans().setK(k).setSeed(1)
    
    model = kmeans.fit(new_df)


    # add the cluster predictions to the data frame
    transformed = model.transform(new_df)
    return transformed

def plot_clusters(df, out_path):
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # geoplot needs a GeoDataFrame (pandas), so convert the spark dataframe to pandas
    kms_df = df.toPandas()
    
    # merge world and df on Country Code and iso_a3 to get the geometry 
    join_df = geopandas.GeoDataFrame(kms_df.merge(world, left_on="Country Code", right_on="iso_a3", how="left"), geometry="geometry")
    join_df = join_df[join_df.geometry != None]

    p = geoplot.choropleth(join_df, hue="prediction", cmap="Greens", legend=True)
    plt.title("Countries clustered by co2-emissions")
    plt.savefig(out_path)

def main():
    # 1. import the data
    co2_data = sqlContext.read.option("inferSchema", "true").option("header", "true").option("nanValue", "..").csv("data/co2_emissions_2004-2014.csv")
    co2_data = preprocess_data(co2_data)
    #co2_data.write.option("header", "true").csv("test")
    #co2_data.show(5)

    # compute change
    compute_change_udf = F.udf(compute_emission_change, FloatType())
    co2_data = co2_data.withColumn("change", compute_change_udf(F.struct([co2_data[x] for x in co2_data.columns])))
    
    co2_data = compute_k_means(co2_data, k=7)

    plot_clusters(co2_data, "co2_emissions.png")

if __name__ == "__main__":
    # set configuration and context
    conf = SparkConf().setAppName("ExamAdmission")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)

    main()
