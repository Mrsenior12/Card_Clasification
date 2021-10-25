#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division             # Division in Python 2.7
import matplotlib
matplotlib.use('Agg')                       # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import math
from matplotlib import colors

def plot_color_gradients(gradients, names):
    # For pretty latex fonts (commented out, because it does not work on some machines)
    #rc('text', usetex=True) 
    #rc('font', family='serif', serif=['Times'], size=10)
    rc('legend', fontsize=10)

    column_width_pt = 400         # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)


    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('my-gradients.pdf')

def hsv2rgb(h, s, v):
    #TODO
    c = v*s
    x = c*(1-abs((h/60)%2 - 1))
    m = v - c
    r,g,b = {
            0: (c,x,0),
            1: (x,c,0),
            2: (0,c,x),
            3: (0,x,c),
            4: (x,0,c),
            5: (c,0,x),
        }[int(h/60)%6]
    return ((r+m),(g+m),(b+m))

def gradient_rgb_bw(v):
    #TODO
    return (v, v, v)


def gradient_rgb_gbr(v):
    #TODO
    if(v<0.5):
        v1 = v*2
        return(0,(1-v1),v1)
    elif(v>=0.5):
        v1 = (v-0.5)*2
        return(v1,0,(1-v1))

def gradient_rgb_gbr_full(v):
    #TODO
    if v < 0.256:
        v1 = v * 4
        return(0,1,v1)
    elif v < 0.512:
        v1 = (v - 0.256) * 4
        return(0,(1-v1),1)
    elif v < 0.768:
        v1 = (v - 0.512) * 4
        return(v1,0,1)
    else:
        v1 = (v - 0.768) * 4
        return(1,0,(1-v1))

def gradient_rgb_wb_custom(v):
    #TODO
    intervals = 7
    lenght = (math.floor(1024 / intervals)/1000)
    if v < lenght:
        v1 = v * intervals
        return (1, 1-v1, 1)
    elif v < lenght * 2:
        v1 = (v - lenght)*intervals
        return (1-v1,0,1)
    elif v < lenght * 3:
        v1 = (v - lenght * 2)*intervals
        return(0,v1,1)
    elif v < lenght * 4:
        v1 = (v - lenght * 3)*intervals
        return(0,1,1-v1)
    elif v < lenght * 5:
        v1 = (v - lenght * 4)*intervals
        return(v1,1,0)
    elif v < lenght * 6:
        v1 = (v - lenght * 5)*intervals
        return(1,1-v1,0)
    else:
        v1 = (v - lenght * 6)*intervals
        return(1-v1,0,0)
        
n = lambda x: max(0,min(1,x))

def gradient_hsv_bw(v):
    #TODO
    return hsv2rgb(0, 0, v)


def gradient_hsv_gbr(v):
    #TODO
    return hsv2rgb((120+(360-120)*v),1,1)

def gradient_hsv_unknown(v):
    #TODO
    return hsv2rgb(120-120*v,0.5,1)


def gradient_hsv_custom(v):
    #TODO
    return hsv2rgb(360*v,n(1-v*v),1)


if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()

    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])
