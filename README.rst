flyion
======

The trajectory of a charged particle in a time-varying electric field

Install
-------

python: 3.7+

requires: scipy, numpy, pandas, matplotlib, tqdm, fastadjust.

.. code-block:: bash

   git clone https://github.com/ad3ller/flyion
   cd ./flyion
   python setup.py install

Basic use
---------

.. code-block:: python

    from scipy.constants import e, m_e
    from fastadjust.io import h5read
    from flyion import initialize, fly

    # A SIMION file with 3 electrodes converted to hdf5 
    fil = os.path.join(r"./", "fast_adjust.h5")
    fa = h5read(fil)

    # many particles, 100 MHz oscillating voltages
    initial = initialize(100, sigma_x=1e-3)
    df = fly(fa, lambda t: [np.sin(t * 1e-8), +200, -300], initial, -e, m_e, dt=5e-10, mode="full")