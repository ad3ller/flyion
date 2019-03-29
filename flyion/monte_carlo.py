# -*- coding: utf-8 -*-
"""
@author: Adam
"""
from collections.abc import Iterable
import numpy as np
import pandas as pd


def initialize(num, t0=0.0, x0=0.0, v0=0.0, sigma_t=None, sigma_x=None, sigma_v=None):
    """ 3D Gaussian time, position, and velocity distributions.

        Parameters
        ----------
        num :: int
            number of particles
        t0 :: float
            mean initial time
        x0 :: float, or tuple(float, float, float)
            mean initial position 
        v0 :: float, or tuple(float, float, float)
            mean initial velocity 
        sigma_t :: None or float
            time spread 
        sigma_x :: None, float, or tuple(float, float, float)
            position spread 
        sigma_v :: None, float, or tuple(float, float, float)
            velocity spread 
            
        Returns
        -------
        pd.DataFrame(columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

    """
    num = int(num)
    # pandas DataFrame
    columns = ['time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    df = pd.DataFrame(columns=columns, index=np.arange(num), dtype='float64')
    # time
    df.time = t0
    if sigma_t is not None:
        df.time += np.random.randn(num) * sigma_t * 2.0**-0.5
    # position
    if not isinstance(x0, Iterable):
        x0 = (x0, x0, x0)
    if sigma_x is None:
        df.x = x0[0]
        df.y = x0[1]
        df.z = x0[2]
    else:
        if not isinstance(sigma_x, Iterable):
            sigma_x = (sigma_x, sigma_x, sigma_x)
        df.x = x0[0] + np.random.randn(num) * sigma_x[0] * 2.0**-0.5
        df.y = x0[1] + np.random.randn(num) * sigma_x[1] * 2.0**-0.5
        df.z = x0[2] + np.random.randn(num) * sigma_x[2] * 2.0**-0.5
    # velocity
    if not isinstance(v0, Iterable):
        v0 = (v0, v0, v0)
    if sigma_v is None:
        df.vx = v0[0]
        df.vy = v0[1]
        df.vz = v0[2]
    else:
        if not isinstance(sigma_v, Iterable):
            sigma_v = (sigma_v, sigma_v, sigma_v)
        df.vx = v0[0] + np.random.randn(num) * sigma_v[0] * 2.0**-0.5
        df.vy = v0[1] + np.random.randn(num) * sigma_v[1] * 2.0**-0.5
        df.vz = v0[2] + np.random.randn(num) * sigma_v[2] * 2.0**-0.5
    return df