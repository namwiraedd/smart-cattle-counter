import boto3
import os

def upload_to_s3(file_path):
    s3 = boto3.client("s3")
    bucket = os.getenv("S3_BUCKET", "cattle-counter-reports")
    s3.upload_file(file_path, bucket, file_path)
    return f"https://{bucket}.s3.amazonaws.com/{file_path}"
