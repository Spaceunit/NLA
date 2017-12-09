import os
from pylab import cm as cm
import matplotlib.pyplot as plt
from timeit import default_timer as timer
cmaps = [m for m in cm.datad if not m.endswith("_r")]
import time
try:
    import pycuda
    import pycuda.driver as drv
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
else:
    GPU_ACCELERATION_AVAILABLE = True

if GPU_ACCELERATION_AVAILABLE:
    import numpy as np
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

def is_gpu_accelerated():
    return GPU_ACCELERATION_AVAILABLE


class DefaultConstants:
    MAX_ITERATIONS = 512
    ESCAPE_RADIUS = 4
    HEIGHT = 8192
    WIDTH = 8192


class GPUObject(object):
    def __init__(self, max_iterations=512, escape_radius=4, power=2):
        self.kernel_headers = self.set_kernel_headers(max_iterations, escape_radius, power)
        self.block_size = (64, 4, 1)

    def generate_kernel_module(self, kernel_code):
        return SourceModule('%s%s' % (self.kernel_headers, kernel_code))

    @staticmethod
    def set_kernel_headers(max_iterations, escape_radius, power):
        return """
        #include <pycuda-complex.hpp>
        #include <math.h>
        typedef pycuda::complex<float> complex;

        #define MAX_ITERATIONS %(MAX_ITERATIONS)s
        #define ESCAPE_RADIUS %(ESCAPE_RADIUS)s
        #define POWER %(POW)s

    """ % ({'MAX_ITERATIONS': max_iterations,
            'ESCAPE_RADIUS': escape_radius,
            'POW': power})


class FractalBuilderGPU(GPUObject):
    def __init__(self, max_iterations=512, escape_radius=2, function_name='mandelbrot', power=2):
        GPUObject.__init__(self, max_iterations, escape_radius)
        if not is_gpu_accelerated():
            exit()
        self.get_pixel_iterations = self.get_function(function_name, max_iterations, escape_radius, power)

    def get_function(self, function_name, max_iterations, escape_radius, power):
        self.kernel_headers = self.set_kernel_headers(max_iterations=max_iterations, escape_radius=escape_radius,
                                                      power=power)
        kernel_module = self.generate_kernel_module(self.set_function_name(function_name))
        return kernel_module.get_function('get_pixel_iterations')

    def build_kernel_code(self, function_name, max_iterations, escape_radius, power):
        self.get_pixel_iterations = self.get_function(function_name, max_iterations, escape_radius, power)

    def build(self, width, height, real_axis_range, imag_axis_range, iterations):
        if not is_gpu_accelerated():
            print('No GPU acceleration is available.')
            exit(code=1)

        # iterations = np.empty(width * height, np.int32)
        # iterations_gpu = gpuarray.to_gpu(iterations)

        # z_values = np.empty(width * height, np.float32)
        # z_values_gpu = gpuarray.to_gpu(z_values)

        c_min = complex(real_axis_range[0], imag_axis_range[0])
        c_max = complex(real_axis_range[1], imag_axis_range[1])
        distance = c_max - c_min

        dx, mx = divmod(width, self.block_size[0])
        dy, my = divmod(height, self.block_size[1])
        grid_size = ((dx + (mx > 0)), (dy + (my > 0)))

        start = timer()
        self.get_pixel_iterations(np.int32(width), np.int32(height), np.complex64(c_min), np.complex64(distance),
                                  drv.Out(iterations), block=self.block_size, grid=grid_size)
        dt = timer() - start
        print("Was counted in {: f} s".format(dt))

        return iterations

    @staticmethod
    def set_function_name(new_function_name):
        return """
            __device__ complex get_complex_abs(complex z) {
                return complex(abs(z.real()), abs(z.imag()));
            }

            __device__ int julia(int width, int height, complex c_min, complex distance, int x, int y, complex &z) {
                float fx = x / (float)(width - 1),
                      fy = y / (float)(height - 1);

                z = c_min + complex(fx * distance.real(), fy * distance.imag());
                complex c = complex(-0.11, 0.51);
                //complex c = complex(-0.7, 0.27015);

                int iteration = 0;
                while(iteration < MAX_ITERATIONS && abs(z) < ESCAPE_RADIUS) {
                    //z = get_complex_abs(z) * get_complex_abs(z) + c;
                    z = pow(z, POWER) + c;
                    iteration++;
                }

                return iteration;
            }

            __device__ int mandelbrot(int width, int height, complex c_min, complex distance, int x, int y, complex &z) {
                float fx = x / (float)(width - 1),
                      fy = y / (float)(height - 1);

                complex c = c_min + complex(fx * distance.real(), fy * distance.imag());
                z = c;

                int iteration = 0;
                while(iteration < MAX_ITERATIONS && abs(z) < ESCAPE_RADIUS) {
                    //z = pow(get_complex_abs(z), POWER) + c;
                    z = pow(z, POWER) + c;
                    iteration++;
                }

                return iteration;
            }

            __device__ int fire_ship(int width, int height, complex c_min, complex distance, int x, int y, complex &z) {
                float fx = x / (float)(width - 1),
                      fy = y / (float)(height - 1);

                complex c = c_min + complex(fx * distance.real(), fy * distance.imag());
                z = c;

                int iteration = 0;
                while(iteration < MAX_ITERATIONS && abs(z) < ESCAPE_RADIUS) {
                    z = pow(get_complex_abs(z), POWER) + c;
                    iteration++;
                }

                return iteration;
            }

            __global__ void get_pixel_iterations(int width, int height, complex c_min, complex distance,
                                                 int *iterations) {
                int x = threadIdx.x + blockDim.x * blockIdx.x;
                int y = threadIdx.y + blockDim.y * blockIdx.y;
                int threadId = y * width + x;

                if (x < width && y < height) {
                    complex z;
                    iterations[threadId] = %(SET_FUNCTION)s(width, height, c_min, distance, x, y, z);
                }
            }
            """ % ({'SET_FUNCTION': new_function_name})


class Mandelbrot(FractalBuilderGPU):
    def __init__(self):
        FractalBuilderGPU.__init__(self)
        self.commands = {
            "commands": {
                "none": 0,
                "exit": 1,
                "test": 2,
                "clear": 3,
                "help": 4,
                "iter": 5,
                "show slist": 6,
                "radius": 7,
                "range": 8,
                "mk": 9,
                "start": 10,
                "show result": 11,
                "image": 12,
                "image file": 13,
                "name": 14,
                "res": 15,
                "path": 16,
            },
            "description": {
                "none": "do nothing",
                "exit": "exit from module",
                "test": "do test stuff",
                "clear": "clear something",
                "help": "display helpfull information",
                "iter": "enter count of iterations",
                "show slist": "show raw data",
                "radius": "enter escape radius",
                "range": "enter range of calculations",
                "mk": "set default raw data",
                "start": "start calculation process",
                "start m": "start calculation process with multi p",
                "show result": "show result",
                "image": "show 2D visualization",
                "image file": "save bitmap into file",
                "name": "enter name of function",
                "path": "enter file path",
                "res": "set resolution of image"
            }
        }
        self.result = {"point": []}
        self.start_point = [0.0, 0.0]
        self.escape_radius = 2.0
        self.power = 2
        self.real_axis_range = [-2., 2.]
        self.imag_axis_range = [-2., 2.]
        self.niter = 1000
        self.width = 8192
        self.height = 8192
        self.function_name = 'mandelbrot'
        self.file_path = 'Z:\\NLA\\fig_test_0.png'
        self.build_kernel_code('mandelbrot', self.niter, self.escape_radius, self.power)
        print(self.kernel_headers)
        self.matrix = np.empty(self.height * self.width, np.int32)
        self.function_names = ['mandelbrot', 'julia', 'fire_ship']
        self.make_default()

    def show_commands(self):
        print('')
        print("Commands...")
        print("---")
        for item in self.commands["commands"]:
            print(str(item) + ":")
            print("Number: " + str(self.commands["commands"][item]))
            print("Description: " + str(self.commands["description"][item]))
            print("---")

    def enter_command(self):
        command = "0"
        print('')
        print("Enter command (help for Q&A)")
        while (command not in self.commands):
            command = input("->")
            if (command not in self.commands["commands"]):
                print("There is no such command")
            else:
                return self.commands["commands"][command]

    def show_help(self):
        print('')
        print("Help v0.002")
        self.show_commands()

    def make_default(self):
        self.start_point = [-0.5, 0.0]
        self.c = [-0.11, 0.51]
        self.power = 2
        self.escape_radius = 2.0
        self.niter = 1000
        self.width = 8192
        self.height = 8192
        self.matrix.fill(0)
        self.function_name = 'mandelbrot'

    def importparam(self, accuracy):
        # self.accuracy = accuracy
        pass

    def set_range(self):
        print('')
        i = 0
        task = 0
        nm = []
        num = 2
        while i < num:
            print("Enter matrix row (use spaces)")
            print("Row ", i + 1)
            while task != 1:
                row = list(map(float, input("-> ").split()))
                print("Input is correct? (enter - yes/n - no)")
                command = input("-> ")
                if command != "n" and len(row) == 2:
                    task = 1
                    nm.append(row)
                elif len(row) != num:
                    print('')
                    print("Incorrect input: count of items.")
            task = 0
            i += 1
        self.real_axis_range = nm[0]
        self.imag_axis_range = nm[1]

    def input_function_name(self):
        task = 0
        print('')
        print("Enter function name:")
        print("Available functions", self.function_names)
        while task != 1:
            f_name = input("-> ")
            print("Input is correct? (enter - yes/n - no)")
            command = input("-> ")
            if command != "n":
                if f_name not in self.function_names:
                    print("Please enter present function name!")
                    task = 0
                else:
                    self.function_name = f_name
                    self.build_kernel_code(function_name=self.function_name, max_iterations=self.niter,
                                           escape_radius=self.escape_radius, power=self.power)
                    task = 1

    def set_radius(self):
        task = 0
        print('')
        print("Enter escape radius:")
        while task != 1:
            radius = float(input("-> "))
            print("Input is correct? (enter - yes/n - no)")
            command = input("-> ")
            if command != "n":
                self.escape_radius = radius
                self.build_kernel_code(function_name=self.function_name, max_iterations=self.niter,
                                       escape_radius=self.escape_radius, power=self.power)
                task = 1
            else:
                if radius < 0:
                    print("Please enter positive number!")
                    task = 0

    def set_count_of_iterations(self):
        task = 0
        print('')
        print("Enter count of iterations:")
        while task != 1:
            n = int(input("-> "))
            print("Input is correct? (enter - yes/n - no)")
            command = input("-> ")
            if command != "n":
                self.niter = n
                self.build_kernel_code(function_name=self.function_name, max_iterations=self.niter,
                                       escape_radius=self.escape_radius, power=self.power)
                task = 1
            else:
                if n < 0:
                    print("Please enter positive number!")
                    task = 0

    def set_resolution(self):
        task = 0
        print('')
        print("Enter count of pixels by axis:")
        while task != 1:
            n = int(input("-> "))
            print("Input is correct? (enter - yes/n - no)")
            command = input("-> ")
            if command != "n":
                if n < 1:
                    print("Please enter positive number!")
                    task = 0
                else:
                    self.width = n
                    self.height = n
                    task = 1

    def amount_of_images(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.axis('off')
        print(cmaps)
        print(cmaps[9])
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.axis('off')
        print(cmaps)
        print(cmaps[9])
        n = [10, 100, 250, 500, 750, 1000]
        p = [2, 3, 4, 5]
        file_path = 'E:\\TEMP\\NLA\\%(NAME)s_set_of_power_%(POW)s_in_%(ITER)s_iterations.png'
        for name in self.function_names:
            self.function_name = name
            for j in p:
                self.power = j
                for item in n:
                    self.niter = item
                    self.build_kernel_code(self.function_name, self.niter, self.escape_radius, self.power)
                    self.resolve()
                    M = self.matrix
                    # time.sleep(50)
                    M = M.reshape(self.height, self.width)
                    start = timer()
                    plt.imsave(file_path % ({'POW': j, 'NAME': name, 'ITER': item}), M,
                               cmap='PuBu')

                    print(file_path % ({'POW': j, 'NAME': name, 'ITER': item}), " - is done")
                    dt = timer() - start
                    print("Was rendered and saved in {: f} s".format(dt))

    def do_staff(self):
        task = 0
        while task != 1:
            print('')
            print("Modeling of phase curves of autonomous systems.")
            print('')
            task = self.enter_command()
            if task == 2:
                self.amount_of_images()
            elif task == 3:
                pass
            elif task == 4:
                self.show_help()
            elif task == 5:
                self.set_count_of_iterations()
            elif task == 6:
                self.print_raw_data()
            elif task == 7:
                self.set_radius()
            elif task == 8:
                self.set_range()
            elif task == 9:
                self.make_default()
            elif task == 10:
                self.resolve()
            elif task == 11:
                self.print_result()
            elif task == 12:
                self.print_result_g()
            elif task == 13:
                self.print_result_g_file()
                pass
            elif task == 14:
                self.input_function_name()
            elif task == 15:
                self.set_resolution()
            elif task == 16:
                pass
            elif task == 17:
                pass
            elif task == 18:
                self.set_count_of_iterations()
            elif task == 19:
                pass
            elif task == 20:
                pass
            elif task == 21:
                pass
            elif task == 22:
                pass
        pass

    def print_raw_data(self):
        print("Start point(s)")
        print(self.start_point)
        pass

    def resolve(self):
        self.matrix = self.build(height=self.height, width=self.width,
                                 real_axis_range=self.real_axis_range,
                                 imag_axis_range=self.imag_axis_range, iterations=self.matrix)

    def print_result_g(self):
        print(self.matrix)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        print(cmaps)
        print(cmaps[9])
        plt.imshow(self.matrix.reshape(self.height, self.width),
                   cmap=cmaps[cmaps.index('afmhot')],
                   aspect='auto')
        plt.show()

    def print_result_g_file(self):
        # plt.imsave(self.image, 'Z:\\NLA\\fig_test.png')
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        print(cmaps)
        print(cmaps[9])
        plt.imsave(self.file_path, self.matrix.get().reshape(self.height, self.width),
                   cmap=cmaps[cmaps.index('afmhot')],
                   origin='lower')
        # plt.savefig('Z:\\NLA\\fig_test_0.png', bbox_inches='tight')
        # self.image.save('Z:\\NLA\\fig_test_0.png')

    def print_result(self):
        for j in range(len(self.result['point'])):
            print("----------------------------------------")
            for i in range(len(self.result['point'][j])):
                print("point #" + str(j) + " cord #" + str(i))
                print("x:", self.result['point'][j][i])
                print("----------------------------------------")
        pass


if __name__ == '__main__':
    Some = Mandelbrot()
    Some.do_staff()
