from abc import ABC, abstractmethod
import os
from parse2fit.core.properties import *
from pymatgen.command_line.bader_caller import bader_analysis_from_path
from pymatgen.io.vasp.outputs import Vasprun
from ase.geometry.analysis import Analysis
from ase.io import read
from pymatgen.core.structure import Structure

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

    @abstractmethod
    def parse_all(self):
        pass

class VaspParser(Parser):
    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs) 
        self.vasprun = Vasprun(os.path.join(self.directory, 'vasprun.xml'))

    def parse_structure(self):
        contcar = os.path.join(self.directory, 'CONTCAR')
        if self.options.get("pymatgen_structure") is True and self.options["pymatgen_structure"] is True:
            return Structure.from_file(contcar)
        elif self.options.get("ase_atoms") is True and self.options["ase_atoms"] is True:
            return read(contcar)

    def parse_energy(self):
        return Energy(value=float(self.vasprun.final_energy), unit='eV')

    def parse_charges(self):
        # Logic to get Bader charges from VASP output files
        charges = []
        chgcar_path = os.path.join(self.directory, 'CHGCAR')
        aeccar0_path = os.path.join(self.directory, 'AECCAR0')
        aeccar2_path = os.path.join(self.directory, 'AECCAR2')

        if os.path.exists(chgcar_path) and os.path.exists(aeccar0_path) and os.path.exists(aeccar2_path):
            print(f'Performing Bader Analysis for {self.directory}')
            charge_dct = bader_analysis_from_path(self.directory) # Requires bader executable
            for i, transferred in enumerate(charge_dct['charge_transfer']):
                charges.append(Charge(indice=i, value=-1*transferred, unit="e",
                                       specie=str(self.vasprun.ionic_steps[-1]['structure'][i].specie)))
        else: # No CHGCAR to parse
            print(f'No CHGCAR, AECCAR0 or AECCAR2 found in {self.directory}; no charges on structure')
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
        atoms = read(os.path.join(self.directory, 'CONTCAR'))
        if geometry_type == 'distances':
            unit = 'Angstrom'
        elif geometry_type == 'angles' or geometry_type == 'dihedrals':
            unit = 'degrees'
        return PropertyCollection(self._get_ase_geometries(atoms, geometry_type, unit))

    def parse_lattice_vectors(self):
        lattice_vectors = []
        contcar_path = os.path.join(self.directory, 'CONTCAR')
        atoms = read(contcar_path)
        lattice_vectors.append(LatticeVector(vector=atoms.cell[0], parameter='a', unit='Angstrom'))
        lattice_vectors.append(LatticeVector(vector=atoms.cell[1], parameter='b', unit='Angstrom'))
        lattice_vectors.append(LatticeVector(vector=atoms.cell[2], parameter='c', unit='Angstrom'))
        return PropertyCollection(lattice_vectors)

    def parse_all(self):
        data = {'energy': None, 'structure': None, 'charges': None, 'forces': None, 
                'lattice_vectors': None, 'distances': None, 'angles': None, 'dihedrals': None}

        if self.options.get("pymatgen_structure") is True and self.options["pymatgen_structure"] is True:
            data['structure'] = self.parse_structure() # Default if both are specified
        if self.options.get("ase_atoms") is True and self.options["ase_atoms"] is True:
            data['structure'] = self.parse_structure()
        if self.options.get("energy") is True and self.options["energy"] is True:
            data['energy'] = self.parse_energy()
        if self.options.get("charges") is True and self.options["charges"] is True:
            data['charges'] = self.parse_charges()
        if self.options.get("forces") is True and self.options["forces"] is True:
            data['forces'] = self.parse_forces()
        if self.options.get("distances") is True and self.options["distances"] is True:
            data['distances'] = self.parse_geometries('distances')
        if self.options.get("angles") is True and self.options["angles"] is True:
            data['angles'] = self.parse_geometries('angles')
        if self.options.get("dihedrals") is True and self.options["dihedrals"] is True:
            data['dihedrals'] = self.parse_geometries('dihedrals')
        if self.options.get("lattice_vectors") is True and self.options["lattice_vectors"] is True:
            data['lattice_vectors'] = self.parse_lattice_vectors()

        return data
    
# Factory to create the correct parser
class ParserFactory:
    @staticmethod
    def create_parser(directory, **kwargs):
        # Logic to determine the type of DFT run and return the appropriate parser
        # For simplicity, we'll assume it's always a VASP run
        # This could be expanded to handle other DFT codes
        if os.path.exists(os.path.join(directory, "vasprun.xml")):
            try:
                return VaspParser(directory, **kwargs)
            except ParseError:
                raise ValueError(f"Issue parsing vasprun.xml in directory: {directory}")
        else:
            raise ValueError(f"Unsupported DFT code for directory: {directory}")

