import math
import matrix
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import mlab

from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import axes3d

import expression


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
                "show result": "show result",
                "image 1": "show 2D visualization",
            }
        }
        self.expression_P = expression.Expression("P", "x2-x**2")
        self.expression_Q = expression.Expression("Q", "x2**2-2*x2-2x1-x**2")
        self.condition = expression.Expression("No name", "x < 5")
        self.accuracy = 3
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

    def makedefault(self):
        # self.accuracy = 5
        self.epsilon[0] = 10.0 ** (-self.accuracy)
        self.epsilon[1] = self.epsilon[0]

    def importparam(self, accuracy):
        # self.accuracy = accuracy
        pass

    def setaccuracy(self):
        task = 0
        print('')
        print("Enter accuracy:")
        while (task != 1):
            self.accuracy = int(input("-> "))
            print("Input is correct? (enter - yes/n - no)")
            command = input("-> ")
            if (command != "n"):
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
                self.printresult()
        pass

    def print_raw_data(self):
        self.expression_P.show_expr()
        self.expression_Q.show_expr()
        pass

    def resolve(self):
        self.makedefault()

    def get_d_lambda(self, x_w, s):
        try:
            d_lambda = self.norm(x_w) / self.norm(s.vector)
        except ZeroDivisionError:
            d_lambda = float('Inf')
        return d_lambda

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

    def collect_data(self):
        pass

    def printresult_g(self):
        pass

    def printresult_3d_0(self):
        pass

    def printresult_3d(self):
        pass

    def print_boundary(self):
        pass

    def print_boundary_1(self):
        pass

    def printresult(self):
        pass

    def printresult_m(self, result: type([])):
        pass