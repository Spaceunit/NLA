import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from pylab import cm as cm
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from pycuda.compiler import SourceModule

# a = np.random.randn(4, 4)
# a = a.astype(np.float32)
# a_gpu = drv.mem_alloc(a.nbytes)
# drv.memcpy_htod(a_gpu, a)

global n, n_block, n_grid, x0, y0, side, loops, M, power, my_obj

loops = 400
n = 800
n_block = 16
n_grid = int(n/16)
n = n_block * n_grid
#start point
x0 = -0.5
y0 = 0.0
side = 10.0
i_cmap = 49
power = 2
fig = plt.figure(figsize=(10, 10),)
# fig = plt.figure()
fig.suptitle('Mandelbrot Set by GPU (PyCUDA)')
ax = fig.add_subplot(111)
ax.axis('off')
cmaps = [m for m in cm.datad if not m.endswith("_r")]

mod = SourceModule("""
    #include <stdio.h>
    #include <pycuda-complex.hpp>
    #include <math.h>
    typedef   pycuda::complex<double> pyComplex;
__device__ float norma(pyComplex z){
    return norm(z);
}
__global__ void mandelbrot(double x0, double y0,double side, int L,int power,int *M)
{
    int n_x = blockDim.x*gridDim.x;
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int idy = threadIdx.y + blockDim.y*blockIdx.y;
    int threadId = idy*n_x+idx;
    float delta = side/n_x;
    // pyComplex c( x0 - side/2. + delta * idx, y0 - side/2. + delta * idy);
    pyComplex c(-0.7, 0.27015);
    pyComplex z( x0 - side/2. + delta * idx, y0 - side/2. + delta * idy);
    int h = 0;
    float R = 2.0;
    while( h<L && norma(z)<R){
        z=pow(z,power)+c;
        h+=1;
    }
    M[threadId]=h;
}

""")

M = np.zeros((n, n)).astype(np.int32)
mandel = mod.get_function("mandelbrot")
mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
       block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))

print(M)


def zoom_on_square(eclick, erelease):
    """eclick and erelease are the press and release events"""
    global n, side, x0, y0, my_obj, M, power
    x1, y1 = min(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata)
    x2, y2 = max(eclick.xdata, erelease.xdata), max(eclick.ydata, erelease.ydata)
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    print(' Nx=%d, Ny=%d, x0=%f, y0=%f' % (x1, y1, x0, y0))
    print(' Nx=%d, Ny=%d, x0=%f, y0=%f' % (x2, y2, x0, y0))
    x_1 = x0+side*(x1 - n / 2.) / n
    y_1 = y0+side*(y1 - n / 2.) / n
    x_2 = x0+side*(x2 - n / 2.) / n
    y_2 = y0+side*(y2 - n / 2.) / n
    x0 = (x_2 + x_1) / 2.
    y0 = (y_2 + y_1) / 2.

    # Average of the 2 rectangle sides
    side = side * (x2 - x1 + y2 - y1) / n / 2
    mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
           block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
    my_obj = plt.imshow(M, origin='lower', cmap=cmaps[i_cmap], aspect='equal',)
    my_obj.set_data(M)
    ax.add_patch(Rectangle((1 - .1, 1 - .1), 0.2, 0.2, alpha=1, facecolor='none', fill=None, ))
    ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
    plt.draw()

def zoom_out_square(eclick, erelease):
    """eclick and erelease are the press and release events"""
    global n, side, x0, y0, my_obj, M, power
    x1, y1 = min(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata)
    x2, y2 = max(eclick.xdata, erelease.xdata), max(eclick.ydata, erelease.ydata)
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    print(' Nx=%d, Ny=%d, x0=%f, y0=%f' % (x1, y1, x0, y0))
    print(' Nx=%d, Ny=%d, x0=%f, y0=%f' % (x2, y2, x0, y0))
    x_1 = x0+side*(x1 - n / 2.) / n
    y_1 = y0+side*(y1 - n / 2.) / n
    x_2 = x0+side*(x2 - n / 2.) / n
    y_2 = y0+side*(y2 - n / 2.) / n
    x0 = (x_2 + x_1) / 2.
    y0 = (y_2 + y_1) / 2.

    # Average of the 2 rectangle sides
    side = side * (x2 - x1 + y2 - y1) / n / 2
    mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
           block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
    my_obj = plt.imshow(M, origin='lower', cmap=cmaps[i_cmap], aspect='equal',)
    my_obj.set_data(M)
    ax.add_patch(Rectangle((1 - .1, 1 - .1), 0.2, 0.2, alpha=1, facecolor='none', fill=None, ))
    ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
    plt.draw()


def key_selector(event):
    global n, side, x0, y0, my_obj, M, power, loops, i_cmap, n_grid
    print(' Key pressed.')

    # Increase max number of iterations
    if event.key == u'up':
        loops = int(loops*1.2)
        print("Maximum number of iterations changed to %d" % loops)
        mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
               block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
        my_obj = plt.imshow(M, cmap=cmaps[i_cmap], origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
        plt.draw()

    # Decrease max number of iterations
    if event.key == u'down':
        loops = int(loops/1.2)
        print("Maximum number of iterations changed to %d" % loops)
        mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
               block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
        my_obj = plt.imshow(M, cmap=cmaps[i_cmap], origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
        plt.draw()

    # Increase  number of pixels
    if event.key == u'right':
        n = int(n*1.2)
        n_grid = int(n / 16.)
        n = n_block * n_grid
        M = np.zeros((n, n)).astype(np.int32)
        print("Number of pixels per dimension changed to %d" % n)
        mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
               block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
        my_obj = plt.imshow(M, cmap=cmaps[i_cmap], origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
        plt.draw()
    # Decrease  number of pixels
    if event.key == u'left':
        n = int(n/1.2)
        n_grid = int(n/16.)
        n = n_block*n_grid
        M = np.zeros((n, n)).astype(np.int32)
        print("Number of pixels per dimension changed to %d" % n)
        mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
               block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
        my_obj = plt.imshow(M, cmap=cmaps[i_cmap], origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
        plt.draw()

    # Decrease  number of pixels
    if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        power = int(event.key)
        if 10 > power > 0:
            print("Power index set to %d" % power)
            i_cmap = 49
            side = 3.0
            x0 -= .5
            y0 = 0.
            loops = 200
            mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
                   block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
            my_obj = plt.imshow(M, cmap=cmaps[i_cmap], origin='lower')
            ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
            plt.draw()


# don't use middle button
key_selector.RS = RectangleSelector(ax, zoom_on_square, drawtype='box', useblit=True, button=[1, 3], minspanx=5,
                                    minspany=5,
                                    spancoords='pixels')
#                                   interactive=False)


def zoom_on_point(event):
    global n, side, x0, y0, my_obj, loops, M, i_cmap, power
    # print(" Button pressed: %d" % (event.button))
    # print(' event.x= %f, event.y= %f '%(event.x,event.y))

    # Zoom on clicked point; new side=10% of old side
    if event.button == 3 and event.inaxes:
        x1, y1 = event.xdata, event.ydata
        x0 = x0+side*(x1-n/2.)/n
        y0 = y0+side*(y1-n/2.)/n
        side = side*.1
        mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
               block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
        my_obj = plt.imshow(M, origin='lower', cmap=cmaps[i_cmap])
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
        plt.draw()

    # Click on left side of image to reset to full fractal
    if not event.inaxes and event.x < .3 * n:
        power = 2
        side = 3.0
        x0 = -.5
        y0 = 0.
        i_cmap = 49
        mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops),
               np.int32(power), drv.Out(M), block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
        my_obj = plt.imshow(M, cmap=cmaps[i_cmap], origin='lower')
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
        plt.draw()
    # Left click on right side of image to set a random colormap
    if event.button == 1 and not event.inaxes and event.x > .7 * n:
        i_cmap_current = i_cmap
        i_cmap = np.random.randint(len(cmaps))
        if i_cmap == i_cmap_current:
            i_cmap -= 1
            if i_cmap < 0:
                i_cmap = len(cmaps) - 1
        # print("color=", i_cmap)
        my_obj = plt.imshow(M, origin='lower', cmap=cmaps[i_cmap])
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
        plt.draw()

    # Right click on right side to set default mapolormap
    if event.button == 3 and not event.inaxes and event.x > .7 * n:
        i_cmap = 49
        my_obj = plt.imshow(M, origin='lower', cmap=cmaps[i_cmap])
        ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
        plt.draw()


fig.canvas.mpl_connect('button_press_event', zoom_on_point)
fig.canvas.mpl_connect('key_press_event', key_selector)
mandel(np.float64(x0), np.float64(y0), np.float64(side), np.int32(loops), np.int32(power), drv.Out(M),
       block=(n_block, n_block, 1), grid=(n_grid, n_grid, 1))
ax.set_title('Side=%.2e, x=%.2e, y=%.2e, %s, Loops=%d' % (side, x0, y0, cmaps[i_cmap], loops))
print(M.max())
plt.imshow(M, origin='lower', cmap=cmaps[i_cmap])
plt.show()

# func = mod.get_function("doublify")
# func(a_gpu, block=(4, 4, 1))
# a_doubled = np.empty_like(a)
# drv.memcpy_dtoh(a_doubled, a_gpu)

# print(a_doubled)
# print(a)
