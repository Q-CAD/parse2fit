"""This module provides classes to define commonly fitted force field properties, 
including Charge(s), Force(s), Geometry(ies), LatticeVector(s), and Energy.
"""

from abc import ABC, abstractmethod
import numpy as np
from parse2fit.tools.unitconverter import UnitConverter

# Base class for chemical properties
class Property(ABC):
    def __init__(self, value=None, unit=None):
        self.value = value
        self.unit = unit

    def convert_units(self, convert_type, to_unit):
        self.value = UnitConverter.convert(self.value, self.unit, to_unit, convert_type)
        self.unit = to_unit 

    def get_rounded(self, sig_figs):
        return np.round(self.value, sig_figs)

# Convenient class for grouping properties
class PropertyCollection:
    def __init__(self, properties=[]):
        self.properties = properties

    def __repr__(self):
        return f"Collection(properties={self.properties})"

    def add_property(self, prop):
        self.properties.append(prop)

class Energy(Property):
    def __init__(self, value=None, unit=None):
        super().__init__(value, unit)

    def __repr__(self):
        return f"Energy(energy={self.value}, unit={self.unit})"

class Charge(Property):
    def __init__(self, value=None, unit=None, indice=None, specie=None):
        super().__init__(value, unit)
        self.indice = indice
        self.specie = specie

    def __repr__(self):
        return f"Charge(specie={self.specie}, indice={self.indice}, charge={self.value}, unit={self.unit})"

class Force(Property):
    def __init__(self, value=None, unit=None, indice=None, specie=None):
        super().__init__(value, unit)
        self.indice = indice
        self.specie = specie

    def __repr__(self):
        return f"Force(specie={self.specie}, indice={self.indice}, force={self.value}, unit={self.unit})"

class Geometry(Property):
    def __init__(self, value=None, unit=None, indices=None, species=None, geometry_type=None):
        super().__init__(value, unit)
        self.geometry_type = geometry_type
        self.indices = indices
        self.species = species

    def __repr__(self):
        return f"Geometry(species={self.species}, indices={self.indices}, {self.geometry_type}={self.value}, unit={self.unit})"

class LatticeVector(Property):
    def __init__(self, vector=None, unit=None, parameter=None):
        super().__init__(np.linalg.norm(vector), unit)
        self.vector = vector
        self.parameter = parameter

    def __repr__(self):
        return f"LatticeVector(parameter={self.parameter}, vector={self.vector}, magnitude={self.value}, unit={self.unit})"


