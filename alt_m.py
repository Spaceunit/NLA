import multiprocessing
import time
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double
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

class CountProcess(multiprocessing.Process):

    def __init__(self, tasks_to_accomplish, tasks_that_are_done):
        multiprocessing.Process.__init__(self)
        self.task_queue = tasks_to_accomplish
        self.result_queue = tasks_that_are_done
        self.exit = multiprocessing.Event()
        self.alpha = [0.01, 0.01]
        self.P = expression.Expression("P", "x2-x1**2")
        self.Q = expression.Expression("Q", "x2**2-2*x2-2*x1-x1**2")
        self.n = 100

    @staticmethod
    def get_vx(param, P, Q):
        vx = None
        try:
            vx = P.execute_l(param) / math.sqrt(
                math.pow(P.execute_l(param), 2) + math.pow(Q.execute_l(param), 2))
        except ZeroDivisionError:
            vx = float('Inf')
        return vx

    @staticmethod
    def get_vy(param, P, Q):
        vy = None
        try:
            vy = Q.execute_l(param) / math.sqrt(
                math.pow(P.execute_l(param), 2) + math.pow(Q.execute_l(param), 2))
        except ZeroDivisionError:
            vy = float('Inf')
        return vy

    @staticmethod
    def deepcopy(x):
        xn = [[] for _ in x]
        for i in range(len(x)):
            for j in range(len(x[i])):
                xn[i].append(x[i][j])
        return xn

    def run(self):
        while not self.exit.is_set():
            try:
                '''
                    try to get task from the queue. get_nowait() function will 
                    raise queue.Empty exception if the queue is empty. 
                    queue(False) function would do the same task also.
                '''
                task = self.task_queue.get_nowait()
                # print(task)
                print(task['name'])

                k = 1
                xk = task['point'].copy()
                self.alpha = task['alpha'].copy()
                self.n = task['n']
                point_move = [xk.copy()]

                while k < self.n:
                    xk[0] += self.alpha[0] * self.get_vx(xk, self.P, self.Q)
                    xk[1] += self.alpha[1] * self.get_vy(xk, self.P, self.Q)
                    point_move.append(xk.copy())
                    k += 1
                self.result_queue.put(self.deepcopy(point_move))
            except queue.Empty:
                self.shutdown()

        print("You exited!")

    def shutdown(self):
        print("Shutdown initiated")
        self.exit.set()


def printresult_g(result):
    vr = []
    for i in range(len(result)):
        vr.append([[], []])
        for j in range(len(result[i])):
            vr[-1][0].append(result[i][j][0])
            vr[-1][1].append(result[i][j][1])
        plt.plot(vr[-1][0], vr[-1][1], 'b-',)

    print("Make plot")
    # print(vr[0])
    plt.show()


if __name__ == "__main__":
    c_range = [-2, 2, -2, 2]
    start_point = [[x, y] for x in range(c_range[0], c_range[1]) for y in
                   range(c_range[2], c_range[3])]
    points_count = len(start_point)
    number_of_task = points_count
    number_of_processes = multiprocessing.cpu_count() - 2
    # tasks_to_accomplish = multiprocessing.Queue()
    # tasks_that_are_done = multiprocessing.Queue()
    tasks_to_accomplish = multiprocessing.Manager().Queue()
    tasks_that_are_done = multiprocessing.Manager().Queue()
    processes = []
    result = []
    n = 1000000
    alpha = [0.00001, 0.00001]

    for i in range(number_of_task):
        # tasks_to_accomplish.put("Task no " + str(i))
        tasks_to_accomplish.put(
            {'name': 'point #' + str(i), 'point': start_point[i].copy(), 'n': n, 'alpha': alpha.copy(),
             'result': [start_point[i].copy()], 'time': 0.0})

    # creating processes
    for w in range(number_of_processes):
        processes.append(CountProcess(tasks_to_accomplish, tasks_that_are_done))

    for w in range(number_of_processes):
        processes[w].start()

    for p in range(number_of_processes):
        processes[p].join()

    # while not tasks_that_are_done.empty():
    #     print(tasks_that_are_done.get())

    # print the output
    print("Tssks that done are empty? -", tasks_that_are_done.empty())
    while not tasks_that_are_done.empty():
        r = tasks_that_are_done.get()
        result.append(r)

    printresult_g(result)

    #process = CountProcess(tasks_to_accomplish, tasks_that_are_done)
    #process.start()
    #print("Waiting for a while")
    #time.sleep(3)
    #process.shutdown()
    #time.sleep(3)
    #print("Child process state: %d" % process.is_alive())