# -*- coding: utf-8 -*-
"""
@author: Adam
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from .trajectory import trajectory, final_position


def fly(fa, vol_t, initial, charge, mass, dt, **kwargs):
    """ Calculate the trajectories of charged particles in a 
    time-varying electric field
    
        Parameters
        ----------
        fa :: FastAdjust
            fast-adjust potential arrays
        vol_t :: func(t),
            returns np.array([v0, v1, ... vn])              (V) 
        initial :: pd.DataFrame
            initial positions and velocities     
        charge :: float64
            particle charge                                 (C)
        mass :: float64
            particle mass                                   (kg)
        dt :: float64
            time step                                       (s)
        max_iterations :: int
            (default: 1 million)
        mode :: str
            'full' (default) or 'final'
        tqdm_kw :: dict
            keyword arguments for tqdm (progress bar)

        Returns
        -------
        mode='full': step-by-step trajectories for all particles ::
            pd.DataFrame(index=['particle', 'time'], columns=['x', 'y', 'z', 'KE', 'PE'])
        
        mode='final': final position for all particles ::
            pd.DataFrame(index=['particle'], columns=['t' 'x', 'y', 'z', 'vx', 'vy', 'vz'])     
    """
    max_iterations = kwargs.get("max_iterations", int(1e6))
    mode = kwargs.get("mode", "full")
    tqdm_kw = kwargs.get("tqdm_kw", {})
    # fly dipoles
    num = len(initial.index)
    result = {}
    for i, row in tqdm(initial.iterrows(), total=len(initial.index), **tqdm_kw):
        t0 = row.time
        x0 = np.array([row.x, row.y, row.z])
        v0 = np.array([row.vx, row.vy, row.vz])
        if fa.electrode_r(x0):
            pass
        elif mode == 'full':
            result[i] = trajectory(fa, vol_t, t0, x0, v0,
                                   charge, mass, dt, max_iterations)
        elif mode == 'final':
            result[i] = final_position(fa, vol_t, t0, x0, v0,
                                       charge, mass, dt, max_iterations, to_series=False)
        else:
            raise ValueError("valid values for arg `mode` : 'full', 'final' ")
    # output
    if mode == "full":
        return pd.concat(result, names=["particle"])
    elif mode == "final":
        df = pd.DataFrame.from_dict(result,
                                    orient='index',
                                    columns=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        df.index.rename('particle')
        return df
    else:
        raise ValueError("valid values for arg `mode` : 'full', 'final' ")
