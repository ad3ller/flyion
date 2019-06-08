# -*- coding: utf-8 -*-
"""
@author: Adam
"""
import numpy as np
import pandas as pd
from scipy.constants import e

def potential_energy(fa, voltages, position, charge):
    """ The potential energy of a charged particle in an electric field.

        Parameters
        ----------
        fa       :: FastAdjust
        voltages :: np.array([v0, v1, ... vn])            (V)
        position :: np.array([x, y, z])                   (m)
        charge   :: float64                               (C)
       
        Returns
        -------
        float64
    """
    return charge * fa.potential_r(position, voltages)

def kinetic_energy(velocity, mass):
    """ The kinetic energy of a particle.

        Parameters
        ----------
        velocity :: np.array([vx, vy, vz])    (m / s)
        mass     :: float64                   (kg) 
        
        Returns
        -------
        float64
    """
    return 0.5 * mass * np.sum(velocity**2.0)

def acceleration(fa, voltages, position, charge_mass_ratio):
    """ Acceleration of an electric charge in an electric field.
    
        Parameters
        ----------
        fa                :: FastAdjust
        voltages          :: np.array([v0, v1, ... vn])            (V)
        position          :: np.array([x, y, z])                   (m)
        charge_mass_ratio :: float64                               (C / kg)

        Returns
        -------
        np.array([ax, ay, az])
    """
    return - charge_mass_ratio * np.array(fa.field_r(position, voltages))

def rk4(fa, voltages, x0, v0, charge_mass_ratio, dt):
    """ Use the fourth order Runge Kutta algorithm to find the position and
    velocity at t = t + dt of a charged particle in an electric field.

        Parameters
        ----------
        fa                :: FastAdjust
        voltages          :: np.array([v0, v1, ... vn])                  (V)
        t                 :: float64                                     (s)
        x0                :: np.array([x, y, z])                         (m)
        v0                :: np.array([vx, vy, vz])                      (m / s)
        charge_mass_ratio :: float64                                     (C / kg)
        dt                :: float64                                     (s)

        Returns
        -------
        position and velocity at t = t + dt ::
            np.array([x, y, z]), np.array([vx, vy, vz])

        http://doswa.com/2009/01/02/fourth-order-runge-kutta-numerical-integration.html
    """
    a0 = acceleration(fa, voltages, x0, charge_mass_ratio)

    x1 = x0 + 0.5 * v0 * dt
    v1 = v0 + 0.5 * a0 * dt
    a1 = acceleration(fa, voltages, x1, charge_mass_ratio)

    x2 = x0 + 0.5 * v1 * dt
    v2 = v0 + 0.5 * a1 * dt
    a2 = acceleration(fa, voltages, x2, charge_mass_ratio)

    x3 = x0 + v2 * dt
    v3 = v0 + a2 * dt
    a3 = acceleration(fa, voltages, x3, charge_mass_ratio)

    x4 = x0 + (dt/6.0)*(v0 + 2*v1 + 2*v2 + v3)
    v4 = v0 + (dt/6.0)*(a0 + 2*a1 + 2*a2 + a3)

    return x4, v4

def trajectory(fa, vol_t, t0, x0, v0, charge, mass, dt, max_iterations=int(1e6)):
    """ Calculate the trajectory and potential and kinetic energy of a charged
    particle in an electric field.
        
        Parameters
        ----------
        fa :: FastAdjust
            fast-adjust potential arrays
        vol_t :: func(t),
            returns np.array([v0, v1, ... vn])              (V)       
        t0 :: float64
            initial time coordinate                         (s)
        x0 :: np.array([x, y, z], dtype=float64)
            initial position vector                         (m)
        v0 :: np.array([vx, vy, vz], dtype=float64)
            initial velocity vector                         (m / s)
        charge :: float64
            particle charge                                 (C m)
        mass :: float64
            particle mass                                   (kg)
        dt :: float64
            time step                                       (s)
        max_iterations :: int
            (default: 1 million)

        Returns
        -------
        step-by-step particle trajectory ::
            pd.DataFrame(index='time', columns=['x', 'y', 'z', 'KE', 'PE'])
    """
    # constants
    charge_mass_ratio = charge / mass
    # initialise
    i = 1
    t = t0
    x = np.array(x0)
    v = np.array(v0)
    voltages = vol_t(t)
    ke = kinetic_energy(v, mass)
    pe = potential_energy(fa, voltages, x, charge)
    result = [[t, *x, ke, pe]]
    # step-by-step trajectory
    while i < max_iterations:
        try:
            x, v = rk4(fa, voltages, x, v, charge_mass_ratio, dt)
            t += dt
            voltages = vol_t(t)
            # energy
            ke = kinetic_energy(v, mass)
            pe = potential_energy(fa, voltages, x, charge)
            # record
            result.append([t, *x, ke, pe])
            # check if particle has hit an electrode
            if fa.electrode_r(x):
                break
            # next step
            i += 1
        except:
            break
    # output
    df = pd.DataFrame(result, columns=['time', 'x', 'y', 'z', 'KE', 'PE']).set_index("time")
    return df

def final_position(fa, vol_t, t0, x0, v0, charge, mass, dt,
                   max_iterations=int(1e6), to_series=True):
    """ Calculate the final position of a particle in an electric field.
        
        Parameters
        ----------
        fa :: FastAdjust
            fast-adjust potential arrays
        vol_t :: func(t),
            returns np.array([v0, v1, ... vn])              (V)       
        t0 :: float64
            initial time coordinate                         (s)
        x0 :: np.array([x, y, z], dtype=float64)
            initial position vector                         (m)
        v0 :: np.array([vx, vy, vz], dtype=float64)
            initial velocity vector                         (m / s)
        charge :: float64
            particle charge                                 (C m)
        mass :: float64
            particle mass                                   (kg)
        dt :: float64
            time step                                       (s)
        max_iterations :: int
            (default: 1 million)

        Returns
        -------
        to_series == True ::
            pd.Series(columns=['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        to_series == False ::
            [t, x, y, z, vx, vy, vz]
    """
    # constants
    charge_mass_ratio = charge / mass
    # initialise
    i = 1
    t = t0
    x = np.array(x0)
    v = np.array(v0)
    voltages = vol_t(t)
    # step-by-step trajectory
    while i < max_iterations:
        try:
            x, v = rk4(fa, voltages, x, v, charge_mass_ratio, dt)
            t += dt
            # check if particle has hit an electrode
            if fa.electrode_r(x):
                break
            # next step
            voltages = vol_t(t)
            i += 1
        except:
            break
    if to_series:
        return pd.Series([t, *x, *v], index=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
    else:
        return [t, *x, *v]
