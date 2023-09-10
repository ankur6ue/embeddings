
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from multiprocessing import Queue, Pool, freeze_support
from queue import Empty
from multiprocessing import Manager, set_start_method
import json
from sentence_transformers import SentenceTransformer
from itertools import repeat
import math
import torch
import os
import time
import boto3
import io
import random
import torch.nn.functional as F

model = None
bucket_name = "ankur-aws-s3-test-bucket"
queue_in = Queue()
queue_out = Queue()

def gpu_id():
    for i in range(2):
        yield i


def init(model_name, queue):
    global model
    process_num = queue.get()
    model = load_model(model_name, process_num)


def shuffle_words(sentence):
        words = sentence.split()
        foo = list(words)
        random.shuffle(foo)
        return ' '.join(foo)


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
    s3_client.put_object(Bucket=bucket_name, Key=parquet_file, Body=data_buffer.getvalue())


def read_from_s3(profile_name, base_parque_file):
    # read from queue
    print('in read_from_s3')
    global queue_in
    global queue_out
    while 1:
        try:
            file_index = queue_in.get(timeout=1)
            if file_index is None:
                print('exiting process {0}'.format(os.getpid()))
                return
            print('retrieved index {0}'.format(file_index))
            try:
                session = boto3.session.Session(profile_name=profile_name)
                s3_client = session.client('s3')
                parquet_file = 'data.parquet{0}.gzip'.format(file_index)
                obj = s3_client.get_object(Bucket=bucket_name, Key=parquet_file)
                df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
                queue_out.put(df)
            except Exception as e:
                print('boto call raised exception')
                return
        except Empty: # no more elements to retrieve, return from process
            print('exiting process {0}'.format(os.getpid()))
            return



if __name__ == '__main__':
    # set the fork start method
    # set_start_method('fork')
    review_json_path = '/home/ankur/dev/apps/ML/data/yelp/yelp_academic_dataset_review.json'
    num_gpus = torch.cuda.device_count()
    profile_name = 'ankur-dev'
    base_parque_file = 'data.parquet.gzip'
    # f1 = pd.read_hdf('data.h5')
    # model_name = 'sentence-transformers/sentence-t5-base'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name).cuda(0)
    query_text = "The best thing they had that actually tasted great was the rice pudding. The sauces and rice were watery and sloppy looking. I went for the buffet and had my hopes up high, but as they say your more likely to become disappointed when you have high expectations. I'm definitely not taking my hungry stomach back here. The only positive thing was that the service was great and the waiters were attentive. According to a food snob who only eats things that look and taste top notch, the food quality was horrible and options were limited."
    queries = []
    queries.append(query_text)
    # generate 25 queries from shuffled text
    num_queries = 10
    for i in range(0,num_queries):
        queries.append(shuffle_words(query_text))

    query_embeddings = model.encode(queries)
    # use second GPU.. because there is more memory there as model is on the first gpu
    queries_pt = torch.from_numpy(query_embeddings).cuda(1)

    num_proc = 5
    queue_in = Queue()
    queue_out = Queue()
    num_parquet_files = 10
    df_cosine_sim = [pd.DataFrame() for i in range(0, num_queries)]
    # query = df['embeddings'][0]
    # query_pt = torch.from_numpy(np.array(query, dtype=np.float32)).cuda(0)


    for index in range(1, num_parquet_files+1):
        queue_in.put(index)

    start = time.time()
    with Pool(num_proc) as pool:
        pool.starmap_async(read_from_s3, zip(repeat(profile_name, num_proc), repeat(base_parque_file, num_proc)))
        # we can either wait for all tasks to finish before we run cosine_similarity for each dataframe, or
        # we can do it in async. Doing it async is more efficient, but runs into timing issues, because there is
        # no good way (I know) to know if a process is finished pulling data from S3..
        # however doing it here results in join never exiting because of a weird issue with putting large
        # objects in a multi-process queue..
        # see:
        # https://stackoverflow.com/questions/11854519/python-multiprocessing-some-functions-do-not-return-when-they-are-complete-que

        # pool.close()
        # pool.join()
        while 1:
            try:
                df = queue_out.get(timeout=30)
                print('got dataframe from output queue..')
                vectors = np.array(df['embeddings'].to_list(), dtype=np.float32)
                vectors_pt = torch.from_numpy(vectors).cuda(1)
                cosine_sim = F.cosine_similarity(queries_pt.unsqueeze(1), vectors_pt, dim=-1).cpu().numpy()
                for i in range(0, num_queries):
                    # pick top 100 cosine distances for each query
                    df_ = pd.DataFrame({'review_id': df['review_id'], 'cosine_dist': cosine_sim[i,:]})
                    df_ = df_.sort_values('cosine_dist', ascending=False).head(100)
                    df_cosine_sim[i] = pd.concat([df_cosine_sim[i], df_], ignore_index=True)
            except Empty:
                print('done')
                break

    print("time to process {0} files: {1} seconds".format(num_parquet_files, time.time() - start))
    pool.close()
    pool.join()

    # re_read = pd.read_parquet('data.parquet.gzip')