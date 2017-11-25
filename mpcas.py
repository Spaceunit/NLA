import math
import matrix
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import mlab
import matplotlib.cm as mplcm
from matplotlib import colors

import queue

from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import axes3d
import multiprocessing
import random
from numba import autojit
from timeit import default_timer as timer

import expression
# plt.cm.s

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

class MPCAS:
    def __init__(self):
        self.commands = {
            "commands": {
                "none": 0,
                "exit": 1,
                "test": 2,
                "clear": 3,
                "help": 4,
                "new": 5,
                "show slist": 6,
                "show scount": 7,
                "acc": 8,
                "mk": 9,
                "start": 10,
                "show result": 11,
                "image 1": 12,
                "start m": 13,
            },
            "description": {
                "none": "do nothing",
                "exit": "exit from module",
                "test": "do test stuff",
                "clear": "clear something",
                "help": "display helpfull information",
                "new": "enter new raw data",
                "show slist": "show raw data",
                "show scount": "show raw data",
                "show acc": "show accuracy",
                "acc": "set accuracy",
                "mk": "set default raw data",
                "start": "start calculation process",
                "start m": "start calculation process with multi p",
                "show result": "show result",
                "image 1": "show 2D visualization",
            }
        }
        self.result = {"point": []}
        self.expression_P = expression.Expression("P", "x2-x1**2")
        self.expression_Q = expression.Expression("Q", "x2**2-2*x2-2*x1-x1**2")
        self.condition = expression.Expression("No name", "x < 5")
        self.start_point = [0.0, 0.0]

        self.tasks_to_accomplish = multiprocessing.Manager().Queue()
        self.tasks_that_are_done = multiprocessing.Manager().Queue()

        self.processes = []


        self.accuracy = 3
        self.alpha = [10.0 ** (-self.accuracy), 10.0 ** (-self.accuracy)]

        self.n = 1000000
        self.points_count = 10
        self.start_point = [[x, y] for x in range(self.points_count + 1) for y in range(self.points_count + 1)]

        self.epsilon = [1, 1]
        self.makedefault()



    def showCommands(self):
        print('')
        print("Commands...")
        print("---")
        for item in self.commands["commands"]:
            print(str(item) + ":")
            print("Number: " + str(self.commands["commands"][item]))
            print("Description: " + str(self.commands["description"][item]))
            print("---")

    def enterCommand(self):
        command = "0"
        print('')
        print("Enter command (help for Q&A)")
        while (command not in self.commands):
            command = input("->")
            if (command not in self.commands["commands"]):
                print("There is no such command")
            else:
                return self.commands["commands"][command]

    def showHelp(self):
        print('')
        print("Help v0.002")
        self.showCommands()

    def make_range_utits(self):
        self.start_point = [[x, y] for x in range(self.c_range[0], self.c_range[1]) for y in
                            range(self.c_range[2], self.c_range[3])]

    def make_range_interval(self):
        self.start_point

    def makedefault(self):

        self.accuracy = 2
        self.epsilon[0] = 10.0 ** (-self.accuracy)
        self.start_point = [0.2, 0.1]
        self.points_count = 40
        #grd = [cord for cord]
        self.c_range = [-20, 20, -20, 20]
        # self.start_point = [[x, y] for x in range(self.c_range[0], self.c_range[1]) for y in range(self.c_range[2], self.c_range[3])]
        self.make_range_units()


        self.points_count = len(self.start_point)
        self.result = {'point': []}
        print(self.result['point'])
        print(self.start_point)
        for i in range(len(self.start_point)):
            self.result['point'].append([self.start_point[i].copy()])
            # print(self.result['point'])
        # print(self.result['point'])
        self.epsilon[1] = self.epsilon[0]
        self.alpha = self.epsilon.copy()
        #self.result.append(self.start_point.copy())
        #self.n = 1000000
        self.n = 10000

    def importparam(self, accuracy):
        # self.accuracy = accuracy
        pass

    def setaccuracy(self):
        task = 0
        print('')
        print("Enter accuracy:")
        while task != 1:
            self.accuracy = int(input("-> "))
            print("Input is correct? (enter - yes/n - no)")
            command = input("-> ")
            if command != "n":
                task = 1
            else:
                if self.accuracy < 0:
                    print("Please enter positive number!")
                    task = 0
        self.epsilon[0] = 10 ** (-self.accuracy)
        self.epsilon[1] = self.epsilon[0]

    def inputnewdata(self):
        self.expression.input_expr()
        self.expression.input_range()
        pass

    def dostaff(self):
        task = 0
        while task != 1:
            print('')
            print("Modeling of phase curves of autonomous systems.")
            print('')
            task = self.enterCommand()
            if task == 2:
                self.print_boundary()
                pass
            elif task == 3:
                pass
            elif task == 4:
                self.showHelp()
            elif task == 5:
                self.inputnewdata()
            elif task == 6:
                self.print_raw_data()
            elif task == 8:
                self.setaccuracy()
            elif task == 9:
                self.makedefault()
            elif task == 10:
                self.resolve()
            elif task == 11:
                self.printresult()
            elif task == 12:
                self.printresult_g()
            elif task == 13:
                self.resolve_m()
        pass

    def print_raw_data(self):
        self.expression_P.show_expr()
        self.expression_Q.show_expr()
        pass

    @autojit
    def resolve(self):
        self.makedefault()
        #self.result['point'] = [[]] * (self.points_count ** 2)
        # xk = self.start_point.copy()
        # print(xk)
        # self.start_point = [[x, y] for x in range(self.points_count+1) for y in range(self.points_count+1)]
        start = timer()
        k = 1
        i = 0
        while i < self.points_count:
            xk = self.result['point'][i][0].copy()
            # print(xk)
            # for j in range(len(self.result['point'])):
                # print(len(self.result['point'][j]))
            k = 1
            while k < self.n:
                xk[0] += self.alpha[0] * self.getVx(xk)
                xk[1] += self.alpha[1] * self.getVy(xk)
                self.collect_data(i, xk)
                k += 1
            i += 1
        dt = timer() - start
        print("Was counted in {: f} s".format(dt))
        # print(xk)
        # print(len(self.result['point'][0]))
        # print(len(self.result['point'][1]))
        # print(len(self.result['point'][2]))
        # print(len(self.result['point'][3]))
        # self.printresult()

    def resolve_m(self):
        self.makedefault()
        xk = self.start_point.copy()
        print(xk)

        points_count = len(self.start_point)
        number_of_task = points_count
        number_of_processes = multiprocessing.cpu_count()
        for i in range(number_of_task):
            # tasks_to_accomplish.put("Task no " + str(i))
            self.tasks_to_accomplish.put(
                {'name': 'point #' + str(i), 'point': self.start_point[i].copy(), 'n': self.n, 'alpha': self.alpha.copy(),
                 'result': [self.start_point[i].copy()], 'time': 0.0})

        # creating processes
        for w in range(number_of_processes):
            self.processes.append(CountProcess(self.tasks_to_accomplish, self.tasks_that_are_done))

        for w in range(number_of_processes):
            self.processes[w].start()

        for p in range(number_of_processes):
            self.processes[p].join()

        # while not self.tasks_that_are_done.empty():
        #     print(self.tasks_that_are_done.get())

        # print the output
        print("Tssks that done are empty? -", self.tasks_that_are_done.empty())
        while not self.tasks_that_are_done.empty():
            r = self.tasks_that_are_done.get()
            self.result['point'].append(r)


    def resolve_worker(self, P, Q, xk, alpha, i):
        print("Begin worker, #", i)
        print("Start point:")
        print(xk)
        k = 1
        result = [xk.copy()]
        while k < self.n:
            xk[0] += alpha[0] * m_getVx(xk, P=P, Q=Q)
            xk[1] += alpha[1] * m_getVy(xk, P=P, Q=Q)
            result.append(xk.copy())
            k += 1
        return result

    def m_getVx(self, param, P, Q):
        vx = None
        try:
            vx = P.execute_l(param) / math.sqrt(math.pow(P.execute_l(param), 2) + math.pow(Q.execute_l(param), 2))
        except ZeroDivisionError:
            vx = float('Inf')
        return vx

    def m_getVy(self, param, P, Q):
        vy = None
        try:
            vx = Q.execute_l(param) / math.sqrt(math.pow(P.execute_l(param), 2) + math.pow(Q.execute_l(param), 2))
        except ZeroDivisionError:
            vy = float('Inf')
        return vy

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

    def get_alpha(self, param):
        return self.alpha

    @staticmethod
    def arguments_list(x_w, flag):
        i = 0
        replace_array = []
        while i < len(x_w):
            if i != flag:
                replace_array.append(x_w[i])
            else:
                replace_array.append(None)
            i += 1
        return replace_array

    @staticmethod
    def lambda_arguments_list(x_w, s, flag):
        i = 0
        replace_array = []
        while i < len(x_w):
            replace_array.append("(" + str(x_w[i]) + "+" + str(s.vector[i]) + "*x" + ")")
            i += 1
        return replace_array

    def par_sort(self, x, f, cycling):
        f_temp = f.copy()
        x_temp = self.deepcopy(x)
        cycling_temp = cycling.copy()
        index = [i for i in range(len(x))]
        f.sort()
        for i in range(len(x)):
            x[i] = x_temp[f_temp.index(f[i])]
            cycling[i] = cycling_temp[f_temp.index(f[i])]
            f_temp[f_temp.index(f[i])] = None

    @staticmethod
    def compare0(x1, x2):
        ansver = False
        for i in range(len(x1)):
            for j in range(len(x1[0])):
                if x1[i][j] in x2:
                    ansver = True
        return ansver

    @staticmethod
    def compare(x1, x2):
        ansver = False
        for i in range(len(x1)):
            if x1[i] in x2:
                ansver = True
        return ansver

    def halting_check(self, f_arr, center):
        r = True
        f_center = self.expression.execute_l(center)
        if math.sqrt(math.pow(sum([item - f_center for item in f_arr]), 2.0) / float(len(f_arr))) <= self.epsilon[0]:
            r = False
            print("Halting check! - True")
        self.result["f_call"] += 1
        return r

    @staticmethod
    def norm(v):
        return math.sqrt(sum([math.pow(item, 2) for item in v]))

    @staticmethod
    def dif(v1, v2):
        i = 0
        r = []
        while i < len(v1):
            r.append(v1[i] - v2[i])
            i += 1
        return r

    @staticmethod
    def dif_part(v1, v2, part):
        r = v1.copy()
        r[part] -= v2[part]
        return r

    @staticmethod
    def sum(v1, v2):
        i = 0
        r = []
        while i < len(v1):
            r.append(v1[i] + v2[i])
            i += 1
        return r

    @staticmethod
    def sum_part(v1, v2, part):
        r = v1.copy()
        r[part] += v2[part]
        return r

    @staticmethod
    def mul(v1, c):
        r = v1.copy()
        for i in range(len(r)):
            r[i] *= c
        return r

    @staticmethod
    def deepcopy(x):
        xn = [[]for _ in x]
        for i in range(len(x)):
            for j in range(len(x[i])):
                xn[i].append(x[i][j])
        return xn

    def collect_data(self, p_num, x_w):
        self.result["point"][p_num].append(x_w.copy())
        pass

    def printresult_g0(self):
        NUM_COLORS = 1163
        # cNorm = matplotlib.colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
        # scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        colors_list = matplotlib.colors.get_named_colors_mapping()
        colors_list = [colors_list[item] for item in colors_list]

        # fig = plt.figure()
        #ax = fig.add_subplot(111)
        # old way:
        # ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
        # new way:


        verts = [[]] * (self.points_count)
        path = []
        print(len(verts))
        print(len(self.result['point'][0]))
        print(len(self.result['point'][1]))
        print(len(self.result['point'][2]))
        print(len(self.result['point'][3]))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        for i in range(len(verts)):
            for j in range(len(self.result['point'][i])):
                verts[i].append((self.result['point'][i][j][0], self.result['point'][i][j][1]))
            path.append(Path(verts[i]))
        #colors = 100 * np.random.rand(len(verts))
        # colors = matplotlib.colors.to_rgb(colors)
        # random.randint(0, NUM_COLORS)
        for i in range(len(verts)):
            patch = patches.PathPatch(path[i], facecolor='none', lw=1)
            ax.add_patch(patch)
            # xs, ys = zip(*verts[i])
            # ax.plot(xs, ys, '-', lw=1, color=colors_list[random.randint(0, NUM_COLORS)], ms=10)
            pass
        print(verts)
        # p = PatchCollection(m_patches, cmap=matplotlib.cm.jet, alpha=0.4, lw=1.0)

        # colors = 100 * np.random.rand(len(m_patches))
        # p.set_array(np.array(colors))

        # ax.add_collection(p)
        # ax.set_prop_cycle('color', plt.cm.spectral(np.linspace(0, 1, 30)))
        # xs, ys = zip(*verts[0])
        # ax.plot(xs, ys, '-', lw=1, color='red', ms=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Modeling of phase curves of autonomous systems.')
        plt.grid(True)
        plt.show()

    def printresult_3d_0(self):
        pass

    def printresult_3d(self):
        pass

    def print_boundary(self):
        pass

    def printresult_g(self):
        vr = []
        #fig = plt.figure()
        for i in range(len(self.result['point'])):
            vr.append([[], []])
            for j in range(len(self.result['point'][i])):
                vr[-1][0].append(self.result["point"][i][j][0])
                vr[-1][1].append(self.result["point"][i][j][1])
            #ax = fig.add_subplot(111)
            #ax.plot(vr[-1][0], vr[-1][1], '-', lw=1, color='red', ms=10)
            plt.plot(vr[-1][0], vr[-1][1], 'b-',)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        print(vr[0])
        # ax.plot(vr[0][0], vr[0][1], '-', lw=1, color='red', ms=10)
        plt.show()



    def print_boundary_1(self):
        pass

    def printresult(self):
        for j in range(len(self.result['point'])):
            print("----------------------------------------")
            for i in range(len(self.result['point'][j])):
                print("point #" + str(j) + " cord #" + str(i))
                print("x:", self.result['point'][j][i])
                print("----------------------------------------")
        pass

    def printresult_m(self, result: type([])):
        pass


def resolve_worker(self, P, Q, xk, alpha, i):
    print("Begin worker, #", i)
    print("Start point:")
    print(xk)
    k = 1
    result = [xk.copy()]
    while k < self.n:
        xk[0] += alpha[0] * m_getVx(xk, P=P, Q=Q)
        xk[1] += alpha[1] * m_getVy(xk, P=P, Q=Q)
        result.append(xk.copy())
        k += 1
    return result


def m_getVx(param, P, Q):
    vx = None
    try:
        vx = P.execute_l(param) / math.sqrt(math.pow(P.execute_l(param), 2) + math.pow(Q.execute_l(param), 2))
    except ZeroDivisionError:
        vx = float('Inf')
    return vx


def m_getVy(param, P, Q):
    vy = None
    try:
        vx = Q.execute_l(param) / math.sqrt(math.pow(P.execute_l(param), 2) + math.pow(Q.execute_l(param), 2))
    except ZeroDivisionError:
        vy = float('Inf')
    return vy

