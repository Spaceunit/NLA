import openpyxl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab


import matrix

import mpcas


class Work:
    def __init__(self):
        self.accuracy = 3
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
                "mpcas": 10,
            },
            "description": {
                "none": "do nothing",
                "exit": "exit from module",
                "test": "do test stuff",
                "clear": "clear something",
                "help": "display helpfull information",
                "new": "enter new raw data",
                "show slist": "show raw data",
                "show scount": "show something",
                "acc": "set accuracy",
                "mk": "set default raw data",
                "mpcas": "Modeling of phase curves of autonomous systems",
            }
        }
        pass

    def enterCommand(self):
        command = "0"
        print('')
        print("Enter command (help for Q&A)")
        while command not in self.commands["commands"]:
            command = input("->")
            if command not in self.commands["commands"]:
                print("There is no such command")
            else:
                return self.commands["commands"][command]

    def showCommands(self):
        print('')
        print("Commands...")
        print("---")
        for item in self.commands["commands"]:
            print(str(item) + ":")
            print("Number: " + str(self.commands["commands"][item]))
            print("Description: " + str(self.commands["description"][item]))
            print("---")

    def showHelp(self):
        print('')
        print("Help v0.002")
        print("Author of this program: Oleksiy Polshchak")
        self.showCommands()

    def dostaff(self):
        task = 0
        while (task != 1):
            print('')
            print("Fundamentals of nonlinear analysis v1 2017")
            print('')
            task = self.enterCommand()
            if task == 2:
                self.dostaff()
            elif task == 3:
                pass
            elif task == 4:
                self.showHelp()
            elif task == 5:
                self.inputnewdata()
                pass
            elif task == 6:
                self.a.showmatrix()
                pass
            elif task == 8:
                self.setaccuracy()
                pass
            elif task == 9:
                self.makedafault()
            elif task == 10:
                Task = mpcas.MPCAS()
                Task.importparam(self.accuracy)
                Task.dostaff()
                pass
        pass


    def inputnewdata(self):
        task = 0
        self.a = matrix.Matrix([], "Initial matrix")
        while task != 1:
            print('')
            print("Enter matrix dimension:")
            while task != 1:
                num = int(input("-> "))
                print("Input is correct? (enter - yes/n - no)")
                command = input("-> ")
                if command != "n":
                    self.a = self.inputmatrix(num)
                    task = 1
            task = 0
            self.a.rename("Initial matrix")
            self.a.showmatrix()
            print("Matrix is correct? (enter - yes/n - no)")
            command = input("-> ")
            if command != "n":
                task = 1

    def inputmatrix(self, num):
        print('')
        i = 0
        task = 0
        nm = matrix.Matrix([], "new matrix")
        while i < num:
            print("Enter matrix row (use spaces)")
            print("Row ", i + 1)
            while task != 1:
                row = list(map(float, input("-> ").split()))
                print("Input is correct? (enter - yes/n - no)")
                command = input("-> ")
                if command != "n" and len(row) == num:
                    task = 1
                    nm.appendnrow(row)
                elif len(row) != num:
                    print('')
                    print("Incorrect input: count of items.")
            task = 0
            i += 1
        return nm

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
        pass

    def makedafault(self):
        self.accuracy = 3


if __name__ == '__main__':
    Some = Work()
    Some.dostaff()
