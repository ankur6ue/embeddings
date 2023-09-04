
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from multiprocessing import Queue, Pool, freeze_support
import json
from sentence_transformers import SentenceTransformer
import math
import torch
import os
import time
import boto3
import io

model = None

def gpu_id():
    for i in range(2):
        yield i


def init(model_name, queue):
    global model
    process_num = queue.get()
    model = load_model(model_name, process_num)


def load_model(model_name, process_num):
    # sentence = ['This framework generates embeddings for each input sentence']
    # embedding = model.encode(sentence)
    print('loading model on gpu {0}'.format(process_num))
    model = SentenceTransformer(model_name)
    is_cuda = torch.cuda.is_available()
    return model.cuda(process_num)


def process(batch):
    # concatenate the review text into a list of strings to pass to the sentence transformer model
    # print('processing batch on process {0}'.format(os.getpid()))
    global model
    # print('processing batch on gpu {0}'.format(model.device.index))
    review_batch = [r['text'] for r in batch]
    # convert batch to dataframe
    df = pd.DataFrame.from_records(batch)
    # the device=model.device.index is important! It tells sentencetransformer to use the GPU on which
    # the model already lives, otherwise the sentencetransformer will copy the model params to GPU 0!
    embeddings = model.encode(review_batch, device=model.device.index)
    df['embeddings'] = embeddings.tolist()
    return df


def write_to_s3(profile_name, parquet_file, data_buffer):
    session = boto3.session.Session(profile_name='ankur-dev')
    s3_client = session.client('s3')
    bucket_name = "ankur-aws-s3-test-bucket"
    s3_client.put_object(Bucket=bucket_name, Key=parquet_file, Body=data_buffer.getvalue())


def read_from_s3(profile_name, parquet_file):
    session = boto3.session.Session(profile_name=profile_name)
    s3_client = session.client('s3')
    bucket_name = "ankur-aws-s3-test-bucket"
    obj = s3_client.get_object(Bucket=bucket_name, Key=parquet_file)
    df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    return df


if __name__ == '__main__':
    review_json_path = '/home/ankur/dev/apps/ML/data/yelp/yelp_academic_dataset_review.json'
    num_gpus = torch.cuda.device_count()
    batch_size_per_gpu = 1280
    profile_name = 'ankur-dev'
    batch_size = num_gpus * batch_size_per_gpu
    file_path = 'data.parquet.gzip'
    # f1 = pd.read_hdf('data.h5')
    # model_name = 'sentence-transformers/sentence-t5-base'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    queue = Queue()
    for index in range(num_gpus):
        queue.put(index)

    df = pd.DataFrame()
    start = time.time()
    # we need to inform each process which gpu id to use. There is no way that I know that allows for passing
    # different arguments to init function. So, we put the gpu indices on a multi-process safe queue, and pass
    # the queue to the arguments of the init function. Each init callback will pull a number out of the queue, which
    # serves are the gpu id. Because the queue is multiprocess safe, this is safe to do. Kind of a hacky solution,
    # but it works..
    with Pool(num_gpus, initializer=init,
              initargs=(model_name, queue)) as pool:
        batch = []
        line_count = 0
        total_line_count = 0
        num_batches = 0
        file_index = 0
        batches_accum = math.floor(1e5/batch_size)
        with open(review_json_path, 'r') as f:

            for line in f:
                line_count = line_count + 1
                total_line_count = total_line_count + 1
                batch.append(json.loads(line))
                if line_count >= batch_size:
                    num_batches = num_batches + 1
                    res = pd.concat(pool.starmap(process, [[batch[0:batch_size_per_gpu]], [batch[batch_size_per_gpu:]]]),
                                    ignore_index=True)
                    df = pd.concat([df, res], ignore_index=True)
                    batch = []
                    line_count = 0
                    print('processed {0} lines in {1} sec'.format(total_line_count, (time.time() - start)))
                    if num_batches > batches_accum:
                        file_index = file_index + 1
                        parquet_file = 'data.parquet{0}.gzip'.format(file_index)
                        # df.to_parquet(parquet_file, compression='gzip')

                        # for writing to in-memory io buffer.. this is actually slower than writing/reading
                        # to/from disk
                        print('writing dataframe to in-memory buffer..')
                        out_buffer = io.BytesIO()
                        df.to_parquet(out_buffer, compression='gzip')
                        print('writing to s3: {0}..'.format(parquet_file))
                        write_to_s3(profile_name, parquet_file, out_buffer)
                        # df_read_back = read_from_s3(profile_name, parquet_file)
                        # verify the read back dataframe matches what is written
                        # to compare if the original and written df's are the same
                        # re_read = pd.read_parquet(parquet_file)
                        # are_same = df.compare(re_read)
                        # reset the dataframe
                        df = pd.DataFrame()
                        num_batches = 0



# re_read = pd.read_parquet('data.parquet.gzip')