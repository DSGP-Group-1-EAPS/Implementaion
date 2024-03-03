import boto3


def access_iam_role(aws_access_key_id, aws_secret_access_key, region_name):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    return session


def get_resource(session, service):
    return session.resource(service)


def get_bucket(session, bucket_name):
    return session.Bucket(bucket_name)