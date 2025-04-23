from abc import ABC, abstractmethod
from parse2fit.core.properties import *
from parse2fit.io.xml import XMLParser
from parse2fit.tools.unitconverter import UnitConverter
from pymatgen.command_line.bader_caller import bader_analysis_from_objects
from pymatgen.io.vasp.outputs import Vasprun, Chgcar
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.geometry.analysis import Analysis
from ase.io import read
from pymatgen.core.structure import Structure
from xml.etree.ElementTree import ParseError
import warnings
import numpy as np
import os
import sys

from abc import ABC, abstractmethod
from ase.geometry.analysis import Analysis

class Parser(ABC):
    def __init__(self, directory, **kwargs):
        self.directory = directory
        self.options = kwargs

    @abstractmethod
    def parse_energy(self): pass
    
    @abstractmethod
    def parse_charges(self): pass
    
    @abstractmethod
    def parse_forces(self): pass
    
    @abstractmethod
    def parse_geometries(self, geometry_type): pass
    
    @abstractmethod
    def parse_lattice_vectors(self): pass
    
    @abstractmethod
    def parse_stress_vectors(self): pass
    
    @abstractmethod
    def parse_structure(self): pass
    
    def _get_ase_geometries(self, ASE_atoms, geometry_type, unit):
        if "periodic_geometries" in self.options:
            ASE_atoms.set_pbc(self.options["periodic_geometries"])
        
        analysis = Analysis(ASE_atoms)
        geometry_map = {
            "distances": (analysis.unique_bonds, analysis.get_bond_value, "distance"),
            "angles": (analysis.unique_angles, analysis.get_angle_value, "angle"),
            "dihedrals": (analysis.unique_dihedrals, analysis.get_dihedral_value, "dihedral"),
        }
        
        ase_geometries, value_func, write_type = geometry_map.get(geometry_type, ([], None, None))
        properties = []
        
        for image_ind, image in enumerate(ase_geometries):
            for atom_ind, atom_neighbors in enumerate(image):
                for neighbor_inds, neighbors in enumerate(atom_neighbors):
                    indices = [atom_ind] + [int(n) for n in (neighbors if isinstance(neighbors, tuple) else [neighbors])]
                    species = [ASE_atoms[i].symbol for i in indices]
                    value = value_func(image_ind, tuple(indices))
                    properties.append(Geometry(indices=indices, value=float(value), 
                                               species=species, geometry_type=write_type, unit=unit))
        
        return properties

    def parse_all(self):
        data = {'energy': None, 'structure': None, 'charges': None, 'forces': None, 'stress_vectors': None,
                'lattice_vectors': None, 'distances': None, 'angles': None, 'dihedrals': None}

        if self.options.get("pymatgen_structure") is True:
            data['structure'] = self.parse_structure() # Default if both are specified
        if self.options.get("ase_atoms") is True:
            data['structure'] = self.parse_structure()
        if self.options.get("energy") is True:
            data['energy'] = self.parse_energy()
        if self.options.get("charges") is True:
            data['charges'] = self.parse_charges()
        if self.options.get("forces") is True:
            data['forces'] = self.parse_forces()
        if self.options.get("distances") is True:
            data['distances'] = self.parse_geometries('distances')
        if self.options.get("angles") is True:
            data['angles'] = self.parse_geometries('angles')
        if self.options.get("dihedrals") is True:
            data['dihedrals'] = self.parse_geometries('dihedrals')
        if self.options.get("lattice_vectors") is True:
            data['lattice_vectors'] = self.parse_lattice_vectors()
        if self.options.get("stress_vectors") is True:
            data['stress_vectors'] = self.parse_stress_vectors()

        return data


class RMGParser(Parser):
    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)
        self.rmgrun = XMLParser(os.path.join(self.directory, 'forcefield.xml'))
        self.structure = self.parse_structure()

    def parse_structure(self):
        uc = UnitConverter()
        lattice = self._parse_lattice()
        species = self._parse_species()
        coordinates, position_unit = self._parse_coordinates()

        if position_unit != 'crystal':
            print(f'Units "{position_unit}" for position not currently supported!')
            sys.exit(1)

        return Structure(lattice=lattice, species=species, coords=coordinates, coords_are_cartesian=False)

    def _parse_lattice(self):
        lattice_elements = self.rmgrun.find_by_elements(self.rmgrun.root, ['./structure', './crystal', './varray', './v'], [None, None, {'name': 'basis'}, None])
        lattice = [self.rmgrun.get_formatted_element_text(el, 'array') for el in lattice_elements]

        top_element = self.rmgrun.find_by_elements(self.rmgrun.root, ['./structure', './crystal', './varray'], [None, None, {'name': 'basis'}])[0]
        if self.rmgrun.get_element_attrib(top_element, 'units') == 'bohr':
            lattice = [[float(val) * 0.529177 for val in vec] for vec in lattice]

        return lattice

    def _parse_species(self):
        elements = self.rmgrun.find_by_elements(self.rmgrun.root, ['./atominfo', './array', './set', './rc'], [None, {'name': 'atoms'}, None, None])
        return [self.rmgrun.get_formatted_element_text(self.rmgrun.find_by_elements(el, ['./c'], [None])[0], 'string') for el in elements]

    def _parse_coordinates(self):
        elements = self.rmgrun.find_by_elements(self.rmgrun.root, ['./structure', './varray', './v'], [None, {'name': 'positions'}, None])
        coordinates = [self.rmgrun.get_formatted_element_text(el, 'array') for el in elements]
        unit = self.rmgrun.get_element_attrib(self.rmgrun.find_by_elements(self.rmgrun.root, ['./structure', './varray'], [None, {'name': 'positions'}])[0], 'units')
        return coordinates, unit

    def parse_energy(self):
        el = self.rmgrun.find_by_elements(self.rmgrun.root, ['./energy', './i'], [None, {'name': 'total'}])[0]
        return Energy(value=self.rmgrun.get_formatted_element_text(el, 'float'),
                      unit=self.rmgrun.get_element_attrib(self.rmgrun.root.find('./energy'), 'units'))

    def parse_charges(self):
        elements = self.rmgrun.find_by_elements(self.rmgrun.root, ['./varray', './v'], [{'name': 'voronoi_charge'}, None])
        return PropertyCollection([Charge(value=self.rmgrun.get_formatted_element_text(el), specie=self.structure[i].specie, indice=i, unit='e') for i, el in enumerate(elements)])

    def parse_forces(self):
        top_element = self.rmgrun.find_by_elements(self.rmgrun.root, ['./varray'], [{'name': 'forces'}])[0]
        elements = self.rmgrun.find_by_elements(self.rmgrun.root, ['./varray', './v'], [{'name': 'forces'}, None])
        unit = self.rmgrun.get_element_attrib(top_element, 'units')
        return PropertyCollection([Force(vector=self.rmgrun.get_formatted_element_text(el, 'array'), 
                                         value=float(np.linalg.norm(self.rmgrun.get_formatted_element_text(el, 'array'))),
                                         specie=self.structure[i].specie, indice=i, unit=unit) for i, el in enumerate(elements)])

    def parse_geometries(self, geometry_type):
        unit = 'Angstrom' if geometry_type == 'distances' else 'degrees'
        return PropertyCollection(self._get_ase_geometries(AseAtomsAdaptor().get_atoms(self.structure), geometry_type, unit))

    def parse_lattice_vectors(self):
        atoms = AseAtomsAdaptor().get_atoms(self.structure)
        return PropertyCollection([LatticeVector(value=float(np.linalg.norm(atoms.cell[i])),
                                                 vector=[float(val) for val in atoms.cell[i]],
                                                 parameter=param, unit='Angstrom')
                                   for i, param in enumerate('abc')])

    def parse_stress_vectors(self):
        top_element = self.rmgrun.find_by_elements(self.rmgrun.root, ['./varray'], [{'name': 'stress'}])[0]
        elements = self.rmgrun.find_by_elements(self.rmgrun.root, ['./varray', './v'], [{'name': 'stress'}, None])
        unit = self.rmgrun.get_element_attrib(top_element, 'units')
        return PropertyCollection([StressVector(value=float(np.linalg.norm(self.rmgrun.get_formatted_element_text(el, 'array'))), 
                                                vector=self.rmgrun.get_formatted_element_text(el, 'array'), 
                                                unit=self.rmgrun.get_element_attrib(top_element, 'units'), parameter=p) for el, p in zip(elements, ['x', 'y', 'z'])])


class VaspParser(Parser):
    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)
        self.vasprun = Vasprun(os.path.join(self.directory, 'vasprun.xml'))
        self.structure = self.parse_structure()

    def parse_structure(self):
        return self.vasprun.final_structure

    def parse_energy(self):
        return Energy(value=float(self.vasprun.final_energy), unit='eV')

    def parse_charges(self):
        chgcar, aeccar0, aeccar2, potcar = [os.path.join(self.directory, f) for f in ['CHGCAR', 'AECCAR0', 'AECCAR2', 'POTCAR']]
        if all(map(os.path.exists, [chgcar, aeccar0, aeccar2, potcar])):
            print(f'Performing Bader Analysis for {self.directory}')
            charge_dct = bader_analysis_from_objects(
                chgcar=Chgcar.from_file(chgcar),
                potcar=Potcar.from_file(potcar),
                aeccar0=Chgcar.from_file(aeccar0),
                aeccar2=Chgcar.from_file(aeccar2))
            charges = [Charge(indice=i, value=-transferred, unit='e',
                              specie=str(self.vasprun.ionic_steps[-1]['structure'][i].specie))
                       for i, transferred in enumerate(charge_dct['charge_transfer'])]
        else:
            warnings.warn(f'Missing charge-related files in {self.directory}; no charges parsed')
            charges = []
        return PropertyCollection(charges)

    def parse_forces(self):
        final_forces = self.vasprun.ionic_steps[-1]['forces']
        return PropertyCollection([
            Force(indice=i, vector=list(force), unit='eV/Angstrom',
                  specie=str(self.vasprun.ionic_steps[-1]['structure'][i].specie),
                  value=float(np.linalg.norm(force)))
            for i, force in enumerate(final_forces)
        ])

    def parse_geometries(self, geometry_type):
        unit = 'Angstrom' if geometry_type == 'distances' else 'degrees'
        return PropertyCollection(self._get_ase_geometries(AseAtomsAdaptor().get_atoms(self.structure), geometry_type, unit))

    def parse_lattice_vectors(self):
        atoms = AseAtomsAdaptor().get_atoms(self.structure)
        params = ['a', 'b', 'c']
        return PropertyCollection([
            LatticeVector(value=float(np.linalg.norm(atoms.cell[i])),
                          vector=[float(val) for val in atoms.cell[i]],
                          parameter=params[i], unit='Angstrom')
            for i in range(3)
        ])

    def parse_stress_vectors(self):
        stress = self.vasprun.ionic_steps[-1]['stress']
        params = ['x', 'y', 'z']
        return PropertyCollection([
            StressVector(value=float(np.linalg.norm(stress[i])), vector=[float(j) for j in stress[i]], parameter=params[i], unit='kbar')
            for i in range(3)
        ])


# Factory to create the correct parser
# NOTE: Make changes to this so the calculation type is checked for convergence
class ParserFactory:
    @staticmethod
    def create_parser(directory, **kwargs):
        # Logic to determine the type of DFT run and return the appropriate parser
        # For simplicity, we'll only check electronic convergence
        if os.path.exists(os.path.join(directory, "vasprun.xml")):
            try:
                v_path = os.path.join(directory, "vasprun.xml")
                v = Vasprun(v_path)
            except ParseError:
                #raise ValueError(f"Cannot parse vasprun.xml in {directory}")
                return None
            if v.converged_electronic is True:
                print(f"Electronically converged vasprun.xml in {directory}; parsing")
                return VaspParser(directory, **kwargs)
            else:
                print(f"vasprun.xml in {directory} is not electronically converged; not parsing")
                return None

        if os.path.exists(os.path.join(directory, 'forcefield.xml')): # Subject to name change
            try:
                rmg_path = os.path.join(directory, "forcefield.xml")
                r = XMLParser(rmg_path)
            except ParserError:
                #raise ValueError(f"Cannot parse forcefield.xml in {directory}")
                return None
            converged_electronic_element = r.find_by_elements(r.root, ['converged', 'scf'], [None, None])[0]
            converged_electronic = r.get_formatted_element_text(converged_electronic_element, 'boolean')
            if converged_electronic is True:
                print(f"Electronically converged forcefield.xml in {directory}; parsing")
                return RMGParser(directory, **kwargs)
            else:
                print(f"forcefield.xml in {directory} is not electronically converged; not parsing")
                return None
        return None
