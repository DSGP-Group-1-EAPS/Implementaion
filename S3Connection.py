import boto3

# Global variables to store access keys and region
global_access_key_id = None
global_secret_access_key = None
global_region_name = None


def access_iam_role(access_key_id, secret_access_key, region_name):
    global global_access_key_id, global_secret_access_key, global_region_name
    global_access_key_id = access_key_id
    global_secret_access_key = secret_access_key
    global_region_name = region_name

    session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region_name
    )
    return session


def get_resource(session, service):
    return session.resource(service)


def get_bucket(session, bucket_name):
    return session.Bucket(bucket_name)


def get_model(bucket_name, model_path, local_model_file_path):
    s3_client = boto3.client('s3', aws_access_key_id=global_access_key_id, aws_secret_access_key=global_secret_access_key)
    s3_client.download_file(bucket_name, model_path, local_model_file_path)
    # Add other S3-related functions as needed
