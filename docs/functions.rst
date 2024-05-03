Functions
=============

``pylion`` defines a number of functions that can be used to configure a simulation.
They all return dictionaries that you can append to a ``Simulation()``.

.. module:: pylion.functions

.. autofunction:: efield
.. autofunction:: placeions
.. autofunction:: createioncloud
.. autofunction:: evolve
.. autofunction:: thermalvelocities
.. autofunction:: minimise
.. autofunction:: ionneutralheating
.. autofunction:: langevinbath
.. autofunction:: lasercool
.. autofunction:: linearpaultrap
.. autofunction:: endcappaultrap
.. autofunction:: endcap_aq
.. autofunction:: endcap_beta
.. autofunction:: endcap_secular
.. autofunction:: timeaverage
.. autofunction:: squaresum
.. autofunction:: dump

Also two helper functions that are not meant to be appended to the simulation:

.. autofunction:: trapaqtovoltage
.. autofunction:: readdump
