import boto3
import requests
import os
import json
import time
import ray
from requests import ConnectionError, HTTPError
from botocore.exceptions import ClientError


def walk(root_dir):
    flac_files = {}
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(root_dir):
        path = root.split(os.sep)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            _, ext = os.path.splitext(file)
            if ext == '.flac':
                flac_files[file] = os.path.join(root, file)
    return flac_files


def poll_sqs():
    sqs_client = create_sqs_client()
    queue_url = "https://sqs.us-east-1.amazonaws.com/549836289368/speech2text-queue"
    response = sqs_client.receive_message(
        QueueUrl=queue_url)
    # Create a new message
    msg = {}
    for message in response.get("Messages", []):
        message_body = message["Body"]
        receipt_handle = message['ReceiptHandle']
        msg = json.loads(message_body)
        sqs_client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
    return msg

def create_role(vault_server_addr: str, vault_role_name: str, aws_role_name: str, token: str):
    url = vault_server_addr + '/v1/aws/roles/' + vault_role_name

    try:
        resp = requests.request('POST', url, headers={"X-Vault-Token": token}, data= \
            {"role_arns": aws_role_name,
             "credential_type": "assumed_role"},
                                timeout=2)
    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
        return
    except ConnectionError:
        print('We failed to reach a server.')
        return
    return resp


def read_sts_creds(vault_server_addr: str, client_token: str, vault_role_name: str):
    # First get client authorization token

    url = vault_server_addr + '/v1/aws/sts/' + vault_role_name
    try:
        resp = requests.request('POST', url, headers={"X-Vault-Token": client_token}, timeout=1)
        # Can also use this:
        # resp = requests.request('GET', url, headers={"Authorization": "Bearer " + client_token}, timeout=1)
    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
        return
    except ConnectionError:
        print('We failed to reach a server.')
        return
    if resp.ok:
        return resp.json()


def create_s3_client():
    vault_server_addr = os.getenv('VAULT_ADDR')
    vault_root_token = os.getenv('VAULT_ROOT_TOKEN')
    vault_role_name = os.getenv('VAULT_S3_ROLE_NAME')
    aws_role_arn = os.getenv('AWS_ROLE_ARN')
    region = 'us-east-1'
    # This is a one time operation performed during onboarding
    # res = create_role(vault_server_addr, vault_role_name, aws_role_arn, vault_root_token)
    sts_creds = read_sts_creds(vault_server_addr, vault_root_token, vault_role_name)
    s3_client = boto3.client(
        's3',
        region_name=region,
        aws_access_key_id=sts_creds["data"]["access_key"],
        aws_secret_access_key=sts_creds["data"]["secret_key"],
        aws_session_token=sts_creds["data"]["security_token"]
    )
    return s3_client


def create_sqs_client():
    vault_server_addr = os.getenv('VAULT_ADDR')
    vault_root_token = os.getenv('VAULT_ROOT_TOKEN')
    vault_role_name = os.getenv('VAULT_SQS_ROLE_NAME')
    aws_role_arn = os.getenv('AWS_ROLE_ARN')
    region = 'us-east-1'
    # This is a one time operation performed during onboarding
    # res = create_role(vault_server_addr, vault_role_name, aws_role_arn, vault_root_token)
    sts_creds = read_sts_creds(vault_server_addr, vault_root_token, vault_role_name)
    sqs_client = boto3.client(
        'sqs',
        region_name=region,
        aws_access_key_id=sts_creds["data"]["access_key"],
        aws_secret_access_key=sts_creds["data"]["secret_key"],
        aws_session_token=sts_creds["data"]["security_token"]
    )
    return sqs_client


def boto_response_ok(resp):
    return resp["ResponseMetadata"]["HTTPStatusCode"] == 200


def upload_to_s3(file_name, bucket, object_name, logger=None):
    s3_client = create_s3_client()
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        if logger:
            logger.error(e)
        return False
    return True



def exit_spin_loop(queues, pools, logging):
    # Logic: for each queue and corresponding actor pool, wait for the queue to empty out, i.e., consumers
    # consuming from that queue to pull tasks from that queue. Then, put a sentinal message on the queue
    # which knocks the process function on the actor out of the spin loop. Do this until there are no more
    # actors in a busy state. This can be determined by calling has_next on the actor pool.
    for queue, pool in zip(queues, pools):
        while not queue.empty():
            time.sleep(0.3)

        queue.put({None: None})

        # we submitted tasks to the speech2text and ner actors. Those are in the submitted state.
        # get number of actors in submitted state
        num_pending_actors = len(pool._future_to_actor)
        for idx in range(0, num_pending_actors):
            try:
                res = pool.get_next_unordered(timeout=1)
            except TimeoutError:
                logging.info("this actor likely either died or couldn't be scheduled")

            queue.put({None: None})
            time.sleep(0.1)
         # If all went well, there should be no pending_actors at this point. If there are, just kill them
        num_pending_actors = len(pool._future_to_actor)
        if num_pending_actors == 0:
            logging.info("successfully exited spin loop for all actors")
        else:
            for future_ref in pool._future_to_actor:
                actor_ref = pool._future_to_actor.get(future_ref)
                logging.info("actor {0} either dead or not scheduled. Killing it".format(actor_ref[1]))
                ray.kill(actor_ref[1])


def shutdown_actors(queues, pools, logging):
    logging.info("shutting down actors")
    for queue, pool in zip(queues, pools):
        # now pop all idle workers in the pool kill them. Also kill the queue
        while pool.has_free():
            ref = pool.pop_idle()
            ray.kill(ref)
        queue.shutdown(force=True)
