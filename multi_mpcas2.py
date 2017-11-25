from multiprocessing import Lock, Process, Queue, current_process, JoinableQueue
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

def do_job(tasks_to_accomplish, tasks_that_are_done):
    while not tasks_to_accomplish.empty():
        try:
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            return True
        #print(task)
        print(task['name'])
        P = expression.Expression("P", "x2-x1**2")
        Q = expression.Expression("Q", "x2**2-2*x2-2*x1-x1**2")

        class Point(Structure):
            _fields_ = [('x', c_double), ('y', c_double)]
        def m_getVx(param, P, Q):
            vx = None
            try:
                vx = P.execute_l(param) / math.sqrt(
                math.pow(P.execute_l(param), 2) + math.pow(Q.execute_l(param), 2))
            except ZeroDivisionError:
                vx = float('Inf')
            return vx

        def m_getVy(param, P, Q):
            vy = None
            try:
                vy = Q.execute_l(param) / math.sqrt(
                math.pow(P.execute_l(param), 2) + math.pow(Q.execute_l(param), 2))
            except ZeroDivisionError:
                vy = float('Inf')
            return vy

        start = timer()
        xk = task['point'].copy()
        k = 1
        while k < task['n']:
            xk[0] += task['alpha'][0] * m_getVx(xk, P=P, Q=Q)
            xk[1] += task['alpha'][1] * m_getVy(xk, P=P, Q=Q)
            task['result'].append(xk.copy())
            k += 1
        dt = timer() - start
        task['time'] = dt
        tasks_that_are_done.put({'name': task['name'], 'point': task['point'].copy(), 'n': task['n'], 'alpha': task['alpha'].copy(),
                 'result': [task['result'][i].copy() for i in range(len(task['result']))], 'time': task['time']})
        # tasks_that_are_done.close()
    return True


def main():

    def printresult_g(result):
        vr = []
        for i in range(len(result)):
            vr.append([[], []])
            for j in range(len(result[i])):
                vr[-1][0].append(result[i][j][0])
                vr[-1][1].append(result[i][j][1])
            plt.plot(vr[-1][0], vr[-1][1], 'b-',)
        print(vr[0])
        plt.show()

    c_range = [-2, 2, -2, 2]
    start_point = [[x, y] for x in range(c_range[0], c_range[1]) for y in
                   range(c_range[2], c_range[3])]
    points_count = len(start_point)

    n = 100
    alpha = [0.00001, 0.00001]

    result = []

    number_of_task = points_count
    number_of_processes = 4
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    for i in range(number_of_task):
        # tasks_to_accomplish.put("Task no " + str(i))
        tasks_to_accomplish.put({'name': 'point #' + str(i), 'point': start_point[i].copy(), 'n': n, 'alpha': alpha.copy(), 'result': [start_point[i].copy()], 'time': 0.0})

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done,))
        processes.append(p)
        p.start()

    # completing process
    while not tasks_to_accomplish.empty():
        # print("Tssks are empty? -", tasks_to_accomplish.empty())
        pass

    # print("Tssks are empty? -", tasks_to_accomplish.empty())
    # while not tasks_that_are_done.empty():
    #     r = tasks_that_are_done.get()
    #     print(r['name'] + ' is done in ' + '{: f}'.format(r['time']))
    #     result.append(r['result'])
    # printresult_g(result)

    for p in processes:
       p.join()

    # print the output
    print("Tssks that done are empty? -", tasks_that_are_done.empty())
    while not tasks_that_are_done.empty():
        r = tasks_that_are_done.get()
        print(r['name'] + ' is done in ' + '{: f}'.format(r['time']))
        print(len(r['result']))
        result.append(r['result'])

    printresult_g(result)

    return True


if __name__ == '__main__':
    main()