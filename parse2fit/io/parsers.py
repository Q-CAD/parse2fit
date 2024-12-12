from abc import ABC, abstractmethod
from parse2fit.core.properties import *
from parse2fit.io.xml import XMLParser
from pymatgen.command_line.bader_caller import bader_analysis_from_objects
from pymatgen.io.vasp.outputs import Vasprun, Chgcar
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.ase import AseAtomsAdaptor
from ase.geometry.analysis import Analysis
from ase.io import read
from pymatgen.core.structure import Structure
from xml.etree.ElementTree import ParseError
import numpy as np
import os
import sys

# Base class for parsers
class Parser(ABC):
    def __init__(self, directory, **kwargs):
        self.directory = directory
        self.options = kwargs

    @abstractmethod
    def parse_energy(self):
        pass

    @abstractmethod
    def parse_charges(self):
        pass

    @abstractmethod
    def parse_forces(self):
        pass
    
    @abstractmethod
    def parse_geometries(self, geometry_type):
        pass

    def _get_ase_geometries(self, ASE_atoms, geometry_type, unit):
        properties = []
        # Whether to find periodic neighbors or just neighbors within the cell
        if "periodic_geometries" in self.options:
            ASE_atoms.set_pbc(self.options.get('periodic_geometries'))
        
        analysis = Analysis(ASE_atoms)

        if geometry_type == 'distances':
            ase_geometries = analysis.unique_bonds
        elif geometry_type == 'angles':
            ase_geometries = analysis.unique_angles
        elif geometry_type == 'dihedrals':
            ase_geometries = analysis.unique_dihedrals

        for image_ind, image in enumerate(ase_geometries):
            for atom_ind, atom_neighbors in enumerate(image):
                for neighbor_inds, neighbors in enumerate(atom_neighbors):
                    if geometry_type == 'distances':
                        write_type = 'distance'
                        indices = [atom_ind, neighbors]
                        value = analysis.get_bond_value(image_ind, (atom_ind, neighbors))
                        species = [str(ASE_atoms[i].symbol) for i in [atom_ind] + [neighbors]]
                    elif geometry_type == 'angles':
                        write_type = 'angle'
                        indices = [atom_ind] + [neighbor for neighbor in neighbors]
                        value = analysis.get_angle_value(image_ind, (atom_ind,) + neighbors)
                        species = [str(ASE_atoms[i].symbol) for i in [atom_ind] + list(neighbors)]
                    elif geometry_type == 'dihedrals':
                        write_type = 'dihedral'
                        indices = [atom_ind] + [neighbor for neighbor in neighbors]
                        value = analysis.get_dihedral_value(image_ind, (atom_ind,) + neighbors)
                        species = [str(ASE_atoms[i].symbol) for i in [atom_ind] + list(neighbors)]
                    properties.append(Geometry(indices=indices, value=float(value), species=species, 
                                               geometry_type=write_type, unit=unit))
                        
        return properties

    @abstractmethod
    def parse_lattice_vectors(self):
        pass

    @abstractmethod
    def parse_structure(self):
        pass
    
    def parse_all(self):
        data = {'energy': None, 'structure': None, 'charges': None, 'forces': None,
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

        return data

class RMGParser(Parser):
    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)
        self.rmgrun = XMLParser(os.path.join(self.directory, 'forcefield.xml'))
        self.structure = self.parse_structure()

    def parse_structure(self):
        # Returns pymatgen structure object from an RMG .xml file
        # Subject to change with changes in RMG.xml file format
        lattice = []
        lattice_lst = ['./structure', './crystal', './varray', './v']
        lattice_element_lst = self.rmgrun.find_by_elements(self.rmgrun.root, lattice_lst, [None, None, {'name': 'basis'}, None])
        for i, lattice_el in enumerate(lattice_element_lst):
            lattice_vector = self.rmgrun.get_formatted_element_text(lattice_el, 'array')
            lattice.append(lattice_vector)
        top_lattice_element = self.rmgrun.find_by_elements(self.rmgrun.root, lattice_lst[:-1], [None, None, {'name': 'basis'}])[0]
        lattice_unit = self.rmgrun.get_element_attrib(top_lattice_element, 'units')
        if lattice_unit == 'bohr': # Do the conversion here, can change this
            lattice = np.array(lattice) * 0.529177 # Convert Bohr to Angstroms

        species = []
        species_lst = ['./atominfo', './array', './set', './rc']
        species_elements_lst = self.rmgrun.find_by_elements(self.rmgrun.root, species_lst,
                                       [None, {'name': 'atoms'}, None, None])
        for sel in species_elements_lst:
            specie_element = self.rmgrun.find_by_elements(sel, ['./c'], [None])[0]
            specie = self.rmgrun.get_formatted_element_text(specie_element, 'string')
            species.append(specie)

        coordinates = []
        coordinates_lst = ['./structure', './varray', './v']
        coordinates_element_lst = self.rmgrun.find_by_elements(self.rmgrun.root, coordinates_lst, [None, {'name': 'positions'}, None])
        for i, coordinates_el in enumerate(coordinates_element_lst):
            coordinate_vector = self.rmgrun.get_formatted_element_text(coordinates_el, 'array')
            coordinates.append(coordinate_vector)
        
        top_position_element = self.rmgrun.find_by_elements(self.rmgrun.root, coordinates_lst[:-1], [None, {'name': 'positions'}])[0]
        position_unit = self.rmgrun.get_element_attrib(top_position_element, 'units')
        if position_unit == 'crystal':
            structure = Structure(lattice=lattice, species=species, coords=coordinates, coords_are_cartesian=False)
        else:
            print(f'Units "{position_unit}" for position not currently supported!')
            sys.exit(1)
        return structure

    def parse_energy(self):
        energy_lst = ['./energy', './i']
        energy_element_lst = self.rmgrun.find_by_elements(self.rmgrun.root, energy_lst, [None, {'name': 'total'}])
        return Energy(value=self.rmgrun.get_formatted_element_text(energy_element_lst[0], 'float'), 
                unit=self.rmgrun.get_element_attrib(self.rmgrun.root.find('./energy'), 'units'))

    def parse_charges(self):
        charge_lst = ['./varray', './v']
        charges_element_lst = self.rmgrun.find_by_elements(self.rmgrun.root, charge_lst, [{'name': 'voronoi_charge'}, None])
        charges = []
        for i, charges_el in enumerate(charges_element_lst):
            charges.append(Charge(value=self.rmgrun.get_formatted_element_text(charges_el), specie=self.structure[i].specie, indice=i, 
                       unit='e')) # Only supported unit right now is 'e'; can change if code changes 
        #print(f'Charge parsing not supported for RMG DFT in {self.directory}!')
        return PropertyCollection(properties=charges)

    def parse_forces(self):
        force_lst = ['./varray', './v']
        top_force_element = self.rmgrun.find_by_elements(self.rmgrun.root, [force_lst[0]], [{'name': 'forces'}])[0]
        forces_element_lst = self.rmgrun.find_by_elements(self.rmgrun.root, force_lst, [{'name': 'forces'}, None])
        forces = []
        for i, forces_el in enumerate(forces_element_lst):
            forces.append(Force(value=self.rmgrun.get_formatted_element_text(forces_el, 'array'), specie=self.structure[i].specie, indice=i,
                      unit=self.rmgrun.get_element_attrib(top_force_element, 'units')))
        return PropertyCollection(properties=forces)

    def parse_geometries(self, geometry_type):
        atoms = AseAtomsAdaptor().get_atoms(self.structure)
        if geometry_type == 'distances':
            unit = 'Angstrom'
        elif geometry_type == 'angles' or geometry_type == 'dihedrals':
            unit = 'degrees'
        return PropertyCollection(self._get_ase_geometries(atoms, geometry_type, unit))

    def parse_lattice_vectors(self):
        lattice_vectors = []
        atoms = AseAtomsAdaptor().get_atoms(self.structure)
        lattice_vectors.append(LatticeVector(value=float(np.linalg.norm(atoms.cell[0])), vector=list(atoms.cell[0]),
                                             parameter='a', unit='Angstrom'))
        lattice_vectors.append(LatticeVector(value=float(np.linalg.norm(atoms.cell[1])), vector=list(atoms.cell[1]),
                                             parameter='b', unit='Angstrom'))
        lattice_vectors.append(LatticeVector(value=float(np.linalg.norm(atoms.cell[2])), vector=list(atoms.cell[2]),
                                             parameter='c', unit='Angstrom'))
        return PropertyCollection(lattice_vectors)

class VaspParser(Parser):
    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs) 
        self.vasprun = Vasprun(os.path.join(self.directory, 'vasprun.xml'))
        self.structure = self.parse_structure()

    def parse_structure(self):
        #contcar = os.path.join(self.directory, 'CONTCAR')
        return self.vasprun.final_structure

    def parse_energy(self):
        return Energy(value=float(self.vasprun.final_energy), unit='eV')

    def parse_charges(self):
        # Logic to get Bader charges from VASP output files
        charges = []
        chgcar_path = os.path.join(self.directory, 'CHGCAR')
        aeccar0_path = os.path.join(self.directory, 'AECCAR0')
        aeccar2_path = os.path.join(self.directory, 'AECCAR2')
        potcar_path = os.path.join(self.directory, 'POTCAR')

        if os.path.exists(chgcar_path) and os.path.exists(aeccar0_path) and os.path.exists(aeccar2_path) and os.path.exists(potcar_path):
            print(f'Performing Bader Analysis for {self.directory}')
            chgcar = Chgcar.from_file(chgcar_path)
            aeccar0 = Chgcar.from_file(aeccar0_path)
            aeccar2 = Chgcar.from_file(aeccar2_path)
            potcar = Potcar.from_file(potcar_path)
            charge_dct = bader_analysis_from_objects(chgcar=chgcar, potcar=potcar, aeccar0=aeccar0, aeccar2=aeccar2) # Requires bader executable
            for i, transferred in enumerate(charge_dct['charge_transfer']):
                charges.append(Charge(indice=i, value=-1*transferred, unit="e",
                                       specie=str(self.vasprun.ionic_steps[-1]['structure'][i].specie)))
        else: # No CHGCAR to parse
            print(f'No CHGCAR, POTCAR, AECCAR0 or AECCAR2 found in {self.directory}; no charges on structure')
            pass
        return PropertyCollection(charges)

    def parse_forces(self):
        forces = []
        final_forces = self.vasprun.ionic_steps[-1]['forces']
        for i, force_vector in enumerate(final_forces):
            forces.append(Force(indice=i, value=list(force_vector), unit='eV/Angstrom', 
                              specie=str(self.vasprun.ionic_steps[-1]['structure'][i].specie)))
        return PropertyCollection(forces)

    def parse_geometries(self, geometry_type):
        atoms = AseAtomsAdaptor().get_atoms(self.structure)
        if geometry_type == 'distances':
            unit = 'Angstrom'
        elif geometry_type == 'angles' or geometry_type == 'dihedrals':
            unit = 'degrees'
        return PropertyCollection(self._get_ase_geometries(atoms, geometry_type, unit))

    def parse_lattice_vectors(self):
        lattice_vectors = []
        atoms = AseAtomsAdaptor().get_atoms(self.structure)
        lattice_vectors.append(LatticeVector(value=float(np.linalg.norm(atoms.cell[0])), vector=list(atoms.cell[0]), 
                                             parameter='a', unit='Angstrom'))
        lattice_vectors.append(LatticeVector(value=float(np.linalg.norm(atoms.cell[1])), vector=list(atoms.cell[1]), 
                                             parameter='b', unit='Angstrom'))
        lattice_vectors.append(LatticeVector(value=float(np.linalg.norm(atoms.cell[2])), vector=list(atoms.cell[2]), 
                                             parameter='c', unit='Angstrom'))
        return PropertyCollection(lattice_vectors)

# Factory to create the correct parser
class ParserFactory:
    @staticmethod
    def create_parser(directory, **kwargs):
        # Logic to determine the type of DFT run and return the appropriate parser
        # For simplicity, we'll assume it's always a VASP run
        # This could be expanded to handle other DFT codes
        if os.path.exists(os.path.join(directory, "vasprun.xml")):
            try:
                v_path = os.path.join(directory, "vasprun.xml")
                v = Vasprun(v_path)
            except ParseError:
                #raise ValueError(f"Cannot parse vasprun.xml in {directory}")
                return None
            if v.converged is True:
                print(f"Converged vasprun.xml in {directory}; parsing")
                return VaspParser(directory, **kwargs)
            else:
                print(f"vasprun.xml in {directory} is not converged; not parsing")
                return None

        if os.path.exists(os.path.join(directory, 'forcefield.xml')): # Subject to name change
            try:
                rmg_path = os.path.join(directory, "forcefield.xml")
                r = XMLParser(rmg_path)
            except ParserError:
                #raise ValueError(f"Cannot parse forcefield.xml in {directory}")
                return None
            converged_element = r.find_by_elements(r.root, ['converged', 'convergent'], [None, None])[0]
            converged = r.get_formatted_element_text(converged_element, 'boolean')
            if converged is True:
                print(f"Converged forcefield.xml in {directory}; parsing")
                return RMGParser(directory, **kwargs)
            else:
                print(f"forcefield.xml in {directory} is not converged; not parsing")
                return None
        return None
