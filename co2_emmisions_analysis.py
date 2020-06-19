from pyspark import SparkConf, SparkContext

def main():
    conf = SparkConf().SetAppName("ExamAdmission")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")