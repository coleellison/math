from math import *
from PIL import Image
import numpy as np

img_dim = 1000 #png width and height
def slope_matrix(dydx, xdim, ydim, n, scalar = 1):
    """generates a matrix of slopes at equally spaced x and y values

    Parameters
    ----------
    dydx : function
        first order differential equation
    xdim : list
        x window size [min, max], inclusive
    ydim : _type_
        y window size [min, max], inclusive
    n : int
        resolution
    scalar : int, optional
        amount to scale slopes by (for use in slope_field when window dimensions are not equa), by default 1

    Returns
    -------
    2-D array
        matrix of slopes sorted by x and y coordinate to map onto cartesian plane
    """
    xmin, xmax = xdim
    ymin, ymax = ydim
    matrix = [[atan(dydx(x,y) * scalar) for x in np.linspace(xmin, xmax, n)] for y in np.linspace(ymax, ymin, n)]
    return matrix

#adds two vectors component-wise
vector_addition = lambda v1, v2: [v1[i] + v2[i] for i in range(len(v1))]

def slope_rgb(name, dydx, xdim, ydim):
    """visual representation of slope field at high density. blue = negative slope, red = positive slope, black = 0 slope

    Parameters
    ----------
    name : str
        file name
    dydx : function
        first order differential equation
    xdim : list
        x window size [min, max], inclusive
    ydim : list
        y window size [min, max], inclusive
    """
    n = 1000 #resolution
    m = slope_matrix(dydx, xdim, ydim, n)
    for r in range(n):
        for c in range(n):
            color = min(int(255 * (2 / pi) * (m[r][c])), 255) #map slope from radians to color <- [-255, 255]
            if color > 0:
                m[r][c] = [color, 0, 0] #red
            else:
                color = color * -1
                m[r][c] = [0, 0, color] #blue
    image_arr = np.uint8(np.array(m))
    image = Image.fromarray(image_arr, 'RGB')
    image.save(f'{name}.png')

def slope_field(name, dydx, xdim, ydim):
    """generates a slope field to a png

    Parameters
    ----------
    name : str
        file name
    dydx : function
        first order differential equation
    xdim : list
        x window size (min, max), inclusive
    ydim : list
        y window size (min, max), inclusive
    """
    n = 100 #resolution
    scalar = (xdim[1] - xdim[0]) / (ydim[1] - ydim[0]) #scales slopes if x size and y size aren't equal
    m = slope_matrix(dydx, xdim, ydim, n, scalar)
    image_arr = [[False for i in range(1000)] for j in range(1000)] #set everything to False, then paint True-s over it
    for r in range(n):
        for c in range(n):
            theta = m[r][c]
            cy, cx = r * 10 + 5, c * 10 + 5 #centers each slope at the center of its grid coordinate
            image_arr[cy][cx] = True #center
            for i in range(1,5): #extend 4 units along slope in each direction
                dx = round(i * cos(theta))
                dy = round(i * sin(theta))
                image_arr[cy + dy][cx + dx] = True
                image_arr[cy - dy][cx - dx] = True
    image_arr = np.uint8(np.array(image_arr) * 255) #change to pixel luminance
    image = Image.fromarray(image_arr, "L")
    image.save(f'{name}.png')


if __name__ == "__main__":
    #file name
    file_name = "slope_field"

    #dy/dx = f(x,y)
    dydx = lambda x,y: sin(x) + cos(y)

    #x window
    xdim = (-10, 10)

    #y window
    ydim = (-10, 10)

    slope_field(file_name, dydx, xdim, ydim)