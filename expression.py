class Expression:
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression
        self.range = None
        self.local_min = None
        self.local_max = None
        self.parameters = {}

    def copy(self):
        copy_ex = Expression(self.name, self.expression)
        if self.range != None:
            copy_ex.range = self.range.copy()
        else:
            copy_ex.range = self.range
        copy_ex.local_min = self.local_min
        copy_ex.local_max = self.local_max
        copy_ex.parameters.update(self.parameters)
        return copy_ex

    def input_range(self):
        ans = False
        command = ""
        print('')
        while not ans:
            self.range = list(map(float, input("Input range -> ").split()))
            print("Input is correct?[Y]-> ")
            command = input("-> ")
            if (command.lower() != "y" or command.lower() != "yes") and len(self.range) != 2:
                print("Try again and input \"yes\" or \"y\"")
                ans = True
            else:
                ans = True

    def input_expr(self):
        ans = False
        command = ""
        print('')
        while not ans:
            self.expression = str(input("enter expression -> "))
            print("Input is correct?[Y]-> ")
            command = input("-> ")
            if command.lower() != "y" or command.lower() != "yes":
                print("Try again and input \"yes\" or \"y\"")
                ans = True
            else:
                ans = True

    def show_expr(self):
        print("Expression name:", self.name)
        print("Expression:", self.expression)
        print("Range:", self.range)
        print("Local minimum:", self.local_min)
        print("Local maximum:", self.local_max)
        print("Parameters:", self.parameters)
        pass

    def replace_arg(self, r_array):
        # change x1 x2 x3 ...
        i = 0
        while i < len(r_array):
            if r_array[i] != None:
                #self.expression.replace('x'+str(i+1), '{'+ str(i)+'}')
                self.expression = self.expression.replace("x" + str(i + 1), '('+str(r_array[i])+')')
                #print('x' + str(i + 1), r_array[i])
            i += 1
        #print(len(r_array))
        i = 0
        if len(r_array) == 2 or len(r_array) == 3:
            while i < len(r_array):
                if r_array[i] == None:
                    self.expression = self.expression.replace("x" + str(i + 1), "x")
                i += 1
        #print(self.expression)
        pass

    def execute(self, x):
        try:
            result = eval(self.expression)
        except ZeroDivisionError:
            result = eval(self.parameters["ZDE_expression"])
        return result

    def execute_d(self, x):
        try:
            result = eval(self.expression, x)
        except ZeroDivisionError:
            result = eval(self.parameters["ZDE_expression"], x)
        except OverflowError:
            result = float('Inf')
        return result

    def execute_l(self, x):
        x_dict = {}
        for i in range(len(x)):
            x_dict["x" + str(i + 1)] = x[i]
        return self.execute_d(x_dict)

    def diff_derivative(self, x, h):
        try:
            result = 0.5 * (self.execute(x + h) - self.execute(x - h)) / h
        except ZeroDivisionError:
            result = 0.5 * (self.execute(x + h) - self.execute(x - h)) / float('Inf')
        return result

    def diff_derivative_pi2_l(self, x, h, j):
        x_p = [x[0], x[1]]
        x_m = [x[0], x[1]]
        x_p[j] += h
        x_m[j] -= h
        try:
            result = 0.5 * (self.execute_l(x_p) - self.execute_l(x_m)) / h
        except ZeroDivisionError:
            result = 0.5 * (self.execute_l(x_p) - self.execute_l(x_m)) / float('Inf')
        return result

    def diff2_derivative_pi2_l(self, x, h, i, j):
        x_p = [x[0], x[1]]
        x_m = [x[0], x[1]]
        x_p[i] += h
        x_m[i] -= h
        try:
            result = 0.5 * (self.diff_derivative_pi2_l(x_p, h, j) - self.diff_derivative_pi2_l(x_m, h, j)) / h
        except ZeroDivisionError:
            result = 0.5 * (self.diff_derivative_pi2_l(x_p, h, j) - self.diff_derivative_pi2_l(x_m, h, j)) / float('Inf')
        return result

    def rename(self, name):
        self.name = name
        pass

    def concatexp(self):
        pass