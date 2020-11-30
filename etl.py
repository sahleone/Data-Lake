import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format,monotonically_increasing_id


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    '''Creates PySpark Session '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """ Processes song data and outputs song and artist tables in a spcified location
    
    Keyword arguments:
    spark -- spark session
    input_data -- start of file path for input data
    output_data -- start of file path for output data
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, "song_data/*/*/*/*.json")
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select('song_id', 'title', 'artist_id', 'year', 'duration').dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year", "artist_id").parquet(path = output_data + "/songs/songs.parquet",\
                                                               mode = "overwrite")

    # extract columns to create artists table
    artists_table = df.select(
        df.artist_id,
        df.artist_name,
        df.artist_location,
        df.artist_latitude,
        df.artist_longitude
    ).dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.parquet(path = output_data + "artists/artists.parquet", mode ='overwrite')


def process_log_data(spark, input_data, output_data):
    
    """Processes log data and outputs users, time and songplays
    tables in a spcified location
    
    Keyword arguments:
    spark -- spark session
    input_data -- start of file path for input data
    output_data -- start of file path for output data
    """
    
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/2018/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.select('userId', 'firstName', 'lastName', 'gender', 'level').dropDuplicates()
    users_table = users_table.withColumnRenamed('userId', 'user_id') \
                    .withColumnRenamed('firstName', 'first_name') \
                    .withColumnRenamed('lastName', 'last_name') 
    
    # write users table to parquet files
    users_table.write.parquet(output_data + 'users/users.parquet',mode = 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x))))
    df = df.withColumn('datetime', get_datetime(df.timestamp))
    
    # extract columns to create time table
    time_table = df.select('datetime') \
        .withColumn('hour', hour('datetime')) \
        .withColumn('day', dayofmonth('datetime')) \
        .withColumn('week', weekofyear('datetime')) \
        .withColumn('month', month('datetime')) \
        .withColumn('year', year('datetime')) \
        .withColumn('weekday', dayofweek('datetime')) \
        .dropDuplicates() 
    time_table = time_table.withColumnRenamed('start_time', col('datetime'))
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(output_data + "time/time.parquet",mode = 'overwrite')


    # read in song data to use for songplays table
    song_df = spark.read.json(os.path.join(input_data, "song_data/A/A/A/*.json"))

    # extract columns from joined song and log datasets to create songplays table 
    joint_table = df.join(song_df, (df.song == song_df.title) &\
                              (df.artist == song_df.artist_name) &\
                              (df.length == song_df.duration), 'left_outer') 
    songplays_table  = joint_table.select(
                            col("timestamp").alias("start_time"),
                            col("userId").alias("user_id"),
                            df.level,
                            song_df.song_id,
                            song_df.artist_id,
                            col("sessionId").alias("session_id"),
                            df.location,
                            col("userAgent").alias("user_agent"),
                            year('datetime').alias('year'),
                            month('datetime').alias('month')
                        ).withColumn("songplay_id", monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').parquet(output_data + "songplays/songplays.parquet",mode = 'overwrite')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = ""
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
