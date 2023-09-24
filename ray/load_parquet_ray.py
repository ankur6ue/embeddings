import time
import ray
import os
from ray.util.queue import Queue, Empty
from dotenv import dotenv_values
import boto3
from utils import create_s3_client
import pandas as pd
import io
# Define the square task.


@ray.remote(num_cpus=1)
def read_from_s3_bucket(bucket_name):
    session = boto3.session.Session(profile_name=profile_name)
    s3_client = session.client('s3')
    bucket_name = "ankur-aws-s3-test-bucket"


# @ray.remote(resources={"low_load": 1})
@ray.remote(resources={"high_load": 1})
def run_workflow():
    num_parquet_files = 10
    num_tasks = 5
    queue1 = Queue(maxsize=25, actor_options={"name": "data-producer-queue"})
    queue2 = Queue(maxsize=25, actor_options={"name": "data-consumer-queue"})
    for index in range(1, num_parquet_files + 1):
        queue1.put(index)
    futures = [read_from_s3.remote([queue1, queue2]) for i in range(num_tasks)]
    start = time.time()
    num_items = 0
    while num_items < num_parquet_files:
        index = queue2.get()
        print('processed file# {0}'.format(index))
        num_items = num_items + 1
    print("time to process {0} files: {1} seconds".format(num_parquet_files, time.time() - start))

@ray.remote
def enqueue():
    pass


@ray.remote(num_cpus=1, resources={"high_load": 1})
def read_from_s3(args):
    in_queue = args[0]
    out_queue = args[1]
    while 1:
        try:
            index = in_queue.get(block=False)
            print('retrieved index {0}'.format(index))
            if index is None:
                return -1

            s3_client = create_s3_client()
            bucket_name = "ankur-aws-s3-test-bucket"
            # obj = s3_client.list_objects(Bucket=bucket_name, Prefix='data.parquet')
            parquet_file = 'data.parquet{0}.gzip'.format(index)
            obj = s3_client.get_object(Bucket=bucket_name, Key=parquet_file)
            df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
            out_queue.put(index)
        except Empty:
            return - 1


def main():
    config = dotenv_values("config.env")
    for k, v in config.items():
        os.environ[k] = v
    # s3_client = create_s3_client()
    file_path = 'data.parquet1.gzip'
    # df = read_from_s3(s3_client, file_path)
    ray_cluster_ip = "ray://3.85.8.31:31448"
    working_dir = "/home/ankur/dev/apps/ML/learn/embeddings/ray/"
    # can also specify python packages to be installed, at the cluster level, or at the actor/task level
    # runtime_env = {"pip": ["torch"]}
    ray.init(address=ray_cluster_ip, runtime_env={"working_dir": working_dir,
             "env_vars":config})

    ray.get(run_workflow.remote())


if __name__ == "__main__":
    main()