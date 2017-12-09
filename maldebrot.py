from PIL import Image
import os
import json
import logging
try:
    import pycuda
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
else:
    GPU_ACCELERATION_AVAILABLE = True

if GPU_ACCELERATION_AVAILABLE:
    import numpy as np
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
from pylab import cm as cm
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import matplotlib.cm as mcm

import math
import matrix
import types
import matplotlib.lines as lns
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import mlab
import matplotlib.cm as mplcm
from matplotlib import colors

from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import random
from timeit import default_timer as timer

cmaps = [m for m in cm.datad if not m.endswith("_r")]

LOGGER_FORMAT = '[%(asctime)-15s] [%(process)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=LOGGER_FORMAT)
LOGGER = logging.getLogger('mandelbrot_visualisation')


def is_gpu_accelerated():
    return GPU_ACCELERATION_AVAILABLE


class DefaultConstants:
    MAX_ITERATIONS = 1000
    ESCAPE_RADIUS = 4
    COLOR_DENSITY = int(10 * (MAX_ITERATIONS / 512))
    LOG_ESCAPE_RADIUS = math.log(ESCAPE_RADIUS, 2)
    # RESOURCES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    COLOR_SCHEME_FILE = os.path.join('color_scheme.json')

    if not os.path.exists(COLOR_SCHEME_FILE):
        COLOR_SCHEME = []
    else:
        with open(COLOR_SCHEME_FILE) as color_scheme_file:
            COLOR_SCHEME = json.load(color_scheme_file)
    TOTAL_COLORS = len(COLOR_SCHEME)
    HEIGHT = 8192
    WIDTH = 8192


class GPUObject(object):
    def __init__(self, logger, max_iterations=DefaultConstants.MAX_ITERATIONS,
                 escape_radius=DefaultConstants.ESCAPE_RADIUS):
        self._logger = logger
        self._kernel_headers = """
        #include <pycuda-complex.hpp>
        #include <math.h>
        typedef pycuda::complex<float> complex;

        #define MAX_ITERATIONS %(MAX_ITERATIONS)s
        #define ESCAPE_RADIUS %(ESCAPE_RADIUS)s

    """ % ({'MAX_ITERATIONS': max_iterations,
            'ESCAPE_RADIUS': escape_radius})

        self._block_size = (64, 4, 1)

    def _generate_kernel_module(self, kernel_code):
        return SourceModule('%s%s' % (self._kernel_headers, kernel_code))


GENERATING_KERNEL_CODE = """
    
    __device__ complex get_complex_abs(complex z) {
        return complex(abs(z.real()), abs(z.imag()));
    }
    
    __device__ int g_j_get_pixel_iterations(int width, int height, complex cmin, complex dc, int x, int y, complex & z) {
        float fx = x / (float)(width - 1),
              fy = y / (float)(height - 1);

        // complex c = cmin + complex(fx * dc.real(), fy * dc.imag());
        z = cmin + complex(fx * dc.real(), fy * dc.imag());
        complex c = complex(-0.11, 0.51);
        //complex c = complex(-0.7, 0.27015);
        //z = c;

        int iteration = 0;
        while(iteration < MAX_ITERATIONS && abs(z) < ESCAPE_RADIUS) {
            //z = get_complex_abs(z) * get_complex_abs(z) + c;
            z = z*z + c;
            iteration++;
        }

        return iteration;
    }
    
    __device__ int g_m_get_pixel_iterations(int width, int height, complex cmin, complex dc, int x, int y, complex & z) {
        float fx = x / (float)(width - 1),
              fy = y / (float)(height - 1);

        complex c = cmin + complex(fx * dc.real(), fy * dc.imag());
        //complex c = complex(-0.11, 0.51);
        z = c;

        int iteration = 0;
        while(iteration < MAX_ITERATIONS && abs(z) < ESCAPE_RADIUS) {
            //z = get_complex_abs(z) * get_complex_abs(z) + c;
            z = z*z + c;
            iteration++;
        }

        return iteration;
    }

    __global__ void get_pixel_iterations(int * iterations, float * z_values, int width, int height, complex cmin,
                                         complex dc) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;

        if (x < width && y < height) {
            complex cz;
            iterations[y * width + x] = g_j_get_pixel_iterations(width, height, cmin, dc, x, y, cz);
            z_values[y * width + x] = abs(cz);
        }
    }
"""

RENDERING_KERNEL_CODE = """
    #define LOG_ESCAPE_RADIUS %(LOG_ESCAPE_RADIUS)s
    #define COLOR_DENSITY %(COLOR_DENSITY)s
    #define TOTAL_COLORS %(TOTAL_COLORS)s

    __device__ int g_get_pixel_color(int iteration_count, float z_value, float dc, int * color_scheme) {
        float log_z = log2(z_value);
        float hue = iteration_count + 1 - abs(log2(log_z / LOG_ESCAPE_RADIUS));

        float color_density = COLOR_DENSITY;
        if (dc < 0.01)
        {
            color_density /= 4;
        }
        else if (dc < 0.04)
        {
            color_density /= 3;
        }
        else if (dc < 0.2)
        {
            color_density /= 2;
        }

        int color_index = int(color_density * hue);
        if (color_index >= TOTAL_COLORS) {
            color_index = TOTAL_COLORS - 1;
        }

        return color_scheme[color_index];
    }

    __global__ void get_pixel_color(int * colors, int * color_scheme, int * iterations, float * z_values,
                                    int width, int height, float dc) {
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;

        if (x < width && y < height) {
            int iteration_count = iterations[y * width + x];
            if (iteration_count == MAX_ITERATIONS) {
                colors[y * width + x] = 0;
            } else {
                float z_value = z_values[y * width + x];

                colors[y * width + x] = g_get_pixel_color(
                    iteration_count, z_value, dc, color_scheme);
            }
        }
    }
""" % ({'LOG_ESCAPE_RADIUS': int(DefaultConstants.LOG_ESCAPE_RADIUS),
        'COLOR_DENSITY': DefaultConstants.COLOR_DENSITY,
        'TOTAL_COLORS': DefaultConstants.TOTAL_COLORS})


class MandelbrotGeneratorGPU(GPUObject):

    def __init__(self, logger, max_iterations=DefaultConstants.MAX_ITERATIONS,
                 escape_radius=DefaultConstants.ESCAPE_RADIUS):
        GPUObject.__init__(self, logger, max_iterations, escape_radius)
        if not is_gpu_accelerated():
            return

        kernel_module = self._generate_kernel_module(GENERATING_KERNEL_CODE)
        self._get_pixel_iterations = kernel_module.get_function('get_pixel_iterations')

    def generate(self, width, height, real_axis_range, imag_axis_range):
        if not is_gpu_accelerated():
            self._logger.error('No GPU acceleration is available.')
            return

        # iterations = np.empty(width * height, np.int32)
        iterations = np.zeros((width, height)).astype(np.int32)
        iterations_gpu = gpuarray.to_gpu(iterations)

        z_values = np.empty(width * height, np.float32)
        z_values_gpu = gpuarray.to_gpu(z_values)

        cmin = complex(real_axis_range[0], imag_axis_range[0])
        cmax = complex(real_axis_range[1], imag_axis_range[1])
        dc = cmax - cmin

        dx, mx = divmod(width, self._block_size[0])
        dy, my = divmod(height, self._block_size[1])
        grid_size = ((dx + (mx > 0)), (dy + (my > 0)))

        self._get_pixel_iterations(
            iterations_gpu, z_values_gpu,
            np.int32(width), np.int32(height),
            np.complex64(cmin), np.complex64(dc),
            block=self._block_size, grid=grid_size)

        return [iterations_gpu, z_values_gpu, abs(dc)]


class MandelbrotRendererGPU(GPUObject):

    def __init__(self, logger, max_iterations=DefaultConstants.MAX_ITERATIONS,
                 escape_radius=DefaultConstants.ESCAPE_RADIUS, color_scheme=DefaultConstants.COLOR_SCHEME):
        GPUObject.__init__(self, logger, max_iterations, escape_radius)
        if not is_gpu_accelerated():
            return

        kernel_module = self._generate_kernel_module(RENDERING_KERNEL_CODE)
        self._get_pixel_color = kernel_module.get_function('get_pixel_color')

        color_scheme = np.asarray(color_scheme, np.int32)
        self._color_scheme_gpu = gpuarray.to_gpu(color_scheme)

    def render(self, width, height, results):
        if not is_gpu_accelerated():
            self._logger.error('No GPU acceleration is available.')
            return

        image = Image.new('RGB', (width, height))

        iterations_gpu, z_values_gpu, dc = results

        colors = np.empty(width * height, np.int32)
        colors_gpu = gpuarray.to_gpu(colors)

        dx, mx = divmod(width, self._block_size[0])
        dy, my = divmod(height, self._block_size[1])
        grid_size = ((dx + (mx > 0)), (dy + (my > 0)))

        self._get_pixel_color(
            colors_gpu, self._color_scheme_gpu,
            iterations_gpu, z_values_gpu,
            np.int32(width), np.int32(height), np.float32(dc),
            block=self._block_size, grid=grid_size)

        colors = colors_gpu.get()

        # This is really slow, must be optimized
        image.putdata(colors.tolist())

        return image


class Mandelbrot(MandelbrotGeneratorGPU, MandelbrotRendererGPU):
    def __init__(self, logger):
        MandelbrotGeneratorGPU.__init__(self, logger)
        MandelbrotRendererGPU.__init__(self, logger)
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
                "image": 12,
                "image file": 13,
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
                "image": "show 2D visualization",
                "image file": "save bitmap into file",
            }
        }
        self.result = {"point": []}
        self.start_point = [0.0, 0.0]
        self.escape_radius = 2.0
        self.log_escape_radius = math.log(self.escape_radius, 2)
        self.niter = 512
        self.make_default()
        self.image = None
        self.m = None

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
        self.result = {"point": []}
        self.start_point = [-0.5, 0.0]
        self.escape_radius = 2.0
        self.niter = 512

    def importparam(self, accuracy):
        # self.accuracy = accuracy
        pass

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
                task = 1
            else:
                if n < 0:
                    print("Please enter positive number!")
                    task = 0

    def do_staff(self):
        task = 0
        while task != 1:
            print('')
            print("Modeling of phase curves of autonomous systems.")
            print('')
            task = self.enter_command()
            if task == 2:
                pass
            elif task == 3:
                pass
            elif task == 4:
                self.show_help()
            elif task == 5:
                pass
            elif task == 6:
                self.print_raw_data()
            elif task == 8:
                pass
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
                pass
            elif task == 15:
                pass
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
        res = MandelbrotGeneratorGPU.generate(self, height=DefaultConstants.HEIGHT, width=DefaultConstants.WIDTH, real_axis_range=(-2, 1),
                                              imag_axis_range=(-1.5, 1.5))

        self.m = res[0].get()
        self.image = MandelbrotRendererGPU.render(self, height=DefaultConstants.HEIGHT, width=DefaultConstants.WIDTH, results=res)
        pass

    def collect_data(self, p_num, x_w):
        self.result["point"][p_num].append(x_w.copy())
        pass

    def print_result_g0(self):
        plt.imshow(self.image)
        plt.show()

    def print_result_g(self):
        print(self.m)
        M = self.m.reshape(DefaultConstants.HEIGHT, DefaultConstants.WIDTH)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        print(M)
        print(cmaps)
        print(cmaps[9])
        plt.imshow(M, cmap=cmaps[cmaps.index('PuBuGn')], origin='lower',
                   aspect='auto')

        plt.show()

    def print_result_g_file(self):
        # plt.imsave(self.image, 'Z:\\NLA\\fig_test.png')
        self.image.save('Z:\\NLA\\fig_test_0.png')

    def print_result(self):
        for j in range(len(self.result['point'])):
            print("----------------------------------------")
            for i in range(len(self.result['point'][j])):
                print("point #" + str(j) + " cord #" + str(i))
                print("x:", self.result['point'][j][i])
                print("----------------------------------------")
        pass


if __name__ == '__main__':
    Some = Mandelbrot(LOGGER)
    Some.do_staff()
