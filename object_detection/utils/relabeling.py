import boto3
import pandas as pd
import ast
import os
import shutil


def get_false_positives(rds_client: boto3.client, secret_arn: str, resource_arn: str) -> pd.DataFrame:

    query = """
            SELECT dashboard_video.video_hash_value, dashboard_video.comment, classification_type, seconds_from_start_video
            FROM dashboard_video
            LEFT JOIN dashboard_classification 
            ON dashboard_video.video_hash_value = dashboard_classification.video
            LEFT JOIN dashboard_run
            ON dashboard_video.run = dashboard_run.run_hash
            WHERE boolean_active = 0 AND boolean_initial_classification = 0 AND dashboard_run.location = "TBoer"
            ORDER BY dashboard_video.video_hash_value, boolean_active
            """

    print("Running query on RDS database to retrieve false positives")

    response = rds_client.execute_statement(
        secretArn=secret_arn,
        resourceArn=resource_arn,
        database="application",
        formatRecordsAs='JSON',
        sql=query
    )

    list_of_dicts = ast.literal_eval(response['formattedRecords'])
    df = pd.DataFrame(list_of_dicts)
    df['comment'] = df['comment'].replace('General Processor Generated', '')
    df['type'] = 'FP'

    return df


def get_false_negatives(rds_client: boto3.client, secret_arn: str, resource_arn: str) -> pd.DataFrame:

    query = """
            SELECT dashboard_video.video_hash_value, dashboard_video.comment, classification_type, seconds_from_start_video
            FROM dashboard_video
            LEFT JOIN dashboard_classification 
            ON dashboard_video.video_hash_value = dashboard_classification.video
            LEFT JOIN dashboard_run
            ON dashboard_video.run = dashboard_run.run_hash
            WHERE dashboard_classification.author != "Autodetect" AND dashboard_run.location = "TBoer"
            ORDER BY dashboard_video.video_hash_value, boolean_active
            """

    print("Running query on RDS database to retrieve false negatives")

    response = rds_client.execute_statement(
        secretArn=secret_arn,
        resourceArn=resource_arn,
        database="application",
        formatRecordsAs='JSON',
        sql=query
    )

    list_of_dicts = ast.literal_eval(response['formattedRecords'])
    df = pd.DataFrame(list_of_dicts)
    df['comment'] = df['comment'].replace('General Processor Generated', '')
    df['type'] = 'FN'

    return df


def download_relabeling_videos_from_s3(bucket, location, df_relabeling, videos_folder):

    # Create temp directory when it does not exist
    if not os.path.exists(videos_folder):
        os.makedirs(videos_folder)
    else:
        for file in os.scandir(videos_folder):
            os.remove(file.path)

    # Download all unique videos
    for video_hash in df_relabeling['video_hash_value'].unique():
        bucket.download_file(f'{location}/{video_hash}.mp4', f'{videos_folder}/{video_hash}.mp4')
