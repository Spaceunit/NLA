from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import math
import expression
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib
import expression

import multiprocessing
import time


class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            print('%s: %s' % (proc_name, next_task))
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        time.sleep(0.1)  # pretend to take some time to do the work
        return '%s * %s = %s' % (self.a, self.b, self.a * self.b)

    def __str__(self):
        return '%s * %s' % (self.a, self.b)

    def getVx(self, param):
        vx = None
        try:
            vx = self.expression_P.execute_l(param) / math.sqrt(math.pow(self.expression_P.execute_l(param), 2)
                                                                + math.pow(self.expression_Q.execute_l(param), 2))
        except ZeroDivisionError:
            vx = float('Inf')
        return vx

    def getVy(self, param):
        vy = None
        try:
            vy = self.expression_Q.execute_l(param) / math.sqrt(math.pow(self.expression_P.execute_l(param), 2)
                                                                + math.pow(self.expression_Q.execute_l(param), 2))
        except ZeroDivisionError:
            vy = float('Inf')
        return vy

    def count(self, param):
        k = 1
        while k < task['n']:
            xk[0] += task['alpha'][0] * m_getVx(xk, P=P, Q=Q)
            xk[1] += task['alpha'][1] * m_getVy(xk, P=P, Q=Q)
            task['result'].append(xk.copy())
            k += 1


if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks, results) for i in range(num_consumers)]
    for w in consumers:
        w.start()

    # Enqueue jobs
    num_jobs = 10
    for i in range(num_jobs):
        tasks.put(Task(i, i))

    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # Start printing results
    while num_jobs:
        result = results.get()
        print('Result:', result)
        num_jobs -= 1
