#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: scuisqianfeng
@license: Apache Licence 
@file: format_code.py 
@time: 2021/11/18
@contact: scuisqianfeng@gmail.com
@site:  
@software: PyCharm 
"""


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while tokill() == False:
        for batch, (batch_images, batch_labels) \
                in enumerate(dataset_generator):
            # We fill the queue with new fetched batch until we reach the max       size.
            batches_queue.put((batch, (batch_images, batch_labels)) \
                              , block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue):
    """Thread worker for transferring pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        batch, (batch_images, batch_labels) = batches_queue.get(block=True)
        batch_images_np = np.transpose(batch_images, (0, 3, 1, 2))
        batch_images = torch.from_numpy(batch_images_np)
        batch_labels = torch.from_numpy(batch_labels)

        batch_images = Variable(batch_images).cuda()
        batch_labels = Variable(batch_labels).cuda()
        cuda_batches_queue.put((batch, (batch_images, batch_labels)), block=True)
        if tokill() == True:
            return


if __name__ == '__main__':
    import time
    import Thread
    import sys
    from Queue import Empty, Full, Queue

    num_epoches = 1000
    # model is some Pytorch CNN model
    model.cuda()
    model.train()
    batches_per_epoch = 64
    # Training set list suppose to be a list of full-paths for all
    # the training images.
    training_set_list = None
    # Our train batches queue can hold at max 12 batches at any given time.
    # Once the queue is filled the queue is locked.
    train_batches_queue = Queue(maxsize=12)
    # Our numpy batches cuda transferer queue.
    # Once the queue is filled the queue is locked
    # We set maxsize to 3 due to GPU memory size limitations
    cuda_batches_queue = Queue(maxsize=3)

    training_set_generator = InputGen(training_set_list, batches_per_epoch)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)
    preprocess_workers = 4

    # We launch 4 threads to do load && pre-process the input images
    for _ in range(preprocess_workers):
        t = Thread(target=threaded_batches_feeder, \
                   args=(train_thread_killer, train_batches_queue, training_set_generator))
        t.start()
    cuda_transfers_thread_killer = thread_killer()
    cuda_transfers_thread_killer.set_tokill(False)
    cudathread = Thread(target=threaded_cuda_batches, \
                        args=(cuda_transfers_thread_killer, cuda_batches_queue, train_batches_queue))
    cudathread.start()

    # We let queue to get filled before we start the training
    time.sleep(8)
    for epoch in range(num_epoches):
        for batch in range(batches_per_epoch):
            # We fetch a GPU batch in 0's due to the queue mechanism
            _, (batch_images, batch_labels) = cuda_batches_queue.get(block=True)

            # train batch is the method for your training step.
            # no need to pin_memory due to diminished cuda transfers using queues.
            loss, accuracy = train_batch(batch_images, batch_labels)

    train_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread shutdown
            train_batches_queue.get(block=True, timeout=1)
            cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass
    print("Training done")
