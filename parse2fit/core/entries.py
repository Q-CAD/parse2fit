from abc import ABC, abstractmethod
from parse2fit.tools.fileconverter import StructuretoString 
from parse2fit.core.properties import Energy, PropertyCollection
from parse2fit.tools.weights import WeightedSampler
from parse2fit.tools.unitconverter import UnitConverter
from pymatgen.core.structure import Structure
from ase import Atoms
import numpy as np
import math
from itertools import product
import os
import json

# Base class for Entries
class Entry(ABC):
    def __init__(self, group_name=None, label=None, structure=None, energy=None, 
            charges=None, forces=None, distances=None, angles=None, 
            dihedrals=None, lattice_vectors=None, stress_vectors=None, **kwargs):
        
        self.group_name = group_name
        self.label = label
        self.structure = structure
        self.energy = energy
        self.charges = charges
        self.forces = forces
        self.distances = distances
        self.angles = angles
        self.dihedrals = dihedrals
        self.lattice_vectors = lattice_vectors
        self.stress_vectors = stress_vectors

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def from_dict(self):
        pass

    def _correct_units(self):
        # Scalar conversions
        if self.energy:
            self.energy.convert_units('energy', self.units_dct['energy'])
        if self.charges:
            for charge in self.charges.properties:
                charge.convert_units('charge', self.units_dct['charge'])
        if self.distances:
            for distance in self.distances.properties:
                distance.convert_units('length', self.units_dct['length'])
        if self.angles:
            for angle in self.angles.properties:
                angle.convert_units('angle', self.units_dct['angle'])
        if self.dihedrals:
            for dihedral in self.dihedrals.properties:
                dihedral.convert_units('angle', self.units_dct['angle'])

        # Vector conversions 
        uc = UnitConverter()
        if self.forces:
            for force in self.forces.properties:
                force.vector = uc.convert(force.vector, force.unit,
                                                   self.units_dct['force'], 'force')
                force.convert_units('force', self.units_dct['force'])
        if self.lattice_vectors:
            for lattice_vector in self.lattice_vectors.properties:
                lattice_vector.vector = uc.convert(lattice_vector.vector, lattice_vector.unit,
                                                   self.units_dct['length'], 'length')
                lattice_vector.convert_units('length', self.units_dct['length'])
        if self.stress_vectors:
            for stress_vector in self.stress_vectors.properties:
                stress_vector.vector = uc.convert(stress_vector.vector, stress_vector.unit,
                                                   self.units_dct['pressure'], 'pressure')
                stress_vector.convert_units('pressure', self.units_dct['pressure'])

class FitSNAPEntry(Entry):
    def __init__(self, group_name=None, label=None, structure=None, energy=None, forces=None,
            lattice_vectors=None, stress_vectors=None, **kwargs):

        super().__init__(group_name=group_name,
                         label=label, 
                         structure=structure, 
                         energy=energy,
                         charges=None, 
                         forces=forces, 
                         distances=None,
                         angles=None, 
                         dihedrals=None, 
                         lattice_vectors=lattice_vectors, 
                         stress_vectors=stress_vectors)
        
        self.units_dct = {'energy': 'eV',
                     'force': 'eV/Angstrom',
                     'length': 'Angstrom',
                     'pressure': 'bar'}

        self._correct_units() # Yield the correct units for FitSNAP

    def __repr__(self):
        return f"FitSNAPEntry(group_name={self.group_name}, label={self.label})"

    def from_dict(self, data):
        supported = ['group_name', 'label', 'structure', 'energy', 
                     'forces', 'lattice_vectors', 'stress_vectors']
        for property_type in supported:
            if property_type in data:
                setattr(self, property_type, data[property_type])
        self._correct_units()
        return self

    def to_dct(self):
        data_dct = {}
        data_dct['Dataset'] = {}
        data_dct['Dataset']['Label'] = self.label
        data_dct['Dataset']['Data'] = [{}]

        # Add the structural information
        data_dct['Dataset']['Data'][0]['NumAtoms'] = len(self.structure)
        data_dct['Dataset']['AtomTypeStyle'] = 'chemicalsymbol'
        data_dct['Dataset']['Data'][0]['AtomTypes'] = [site.species_string for site in self.structure]
        data_dct['Dataset']['PositionsStyle'] = 'angstrom'
        data_dct['Dataset']['Data'][0]['Positions'] = self.structure.frac_coords.tolist()

        # Add remaining information
        if self.energy:
            data_dct['Dataset']['EnergyStyle'] = 'electronvolt'
            data_dct['Dataset']['Data'][0]['Energy'] = self.energy.value
        if self.forces:
            data_dct['Dataset']['ForcesStyle'] = 'electronvoltperangstrom'
            data_dct['Dataset']['Data'][0]['Forces'] = [force.vector for force in self.forces.properties]
        if self.lattice_vectors:
            data_dct['Dataset']['LatticeStyle'] = 'angstrom' 
            data_dct['Dataset']['Data'][0]['Lattice'] = [lattice_vector.vector for lattice_vector in self.lattice_vectors.properties]
        if self.stress_vectors:
            data_dct['Dataset']['StressStyle'] = 'bar'
            data_dct['Dataset']['Data'][0]['Stress'] = [stress_vector.vector for stress_vector in self.stress_vectors.properties]

        return data_dct

    def to_xyz(self):
        xyz_string = f"{len(self.structure)}\n"

        # Lattice vectors
        if self.lattice_vectors:
            lattice = " ".join(str(c) for v in self.lattice_vectors.properties for c in v.vector)
            xyz_string += f'Lattice="{lattice}" '

        # Properties and energy
        xyz_string += 'Properties=species:S:1:pos:R:3'
        if self.forces:
            xyz_string += ':forces:R:3'
        xyz_string += ' '
        if self.energy:
            xyz_string += f'energy={self.energy.value} '

        # Stress vectors
        if self.stress_vectors:
            stress = " ".join(str(c) for v in self.stress_vectors.properties for c in v.vector)
            xyz_string += f'stress="{stress}" '

        xyz_string += '\n'

        # Atomic positions and forces
        for i, site in enumerate(self.structure):
            xyz_string += f"{site.specie} {' '.join(map(str, site.coords))} "
            if self.forces:
                force = next(force.vector for force in self.forces.properties if force.indice == i)
                xyz_string += f"{' '.join(map(str, force))}"
            xyz_string += '\n'

        return xyz_string

    def write_dct(self, write_path):
        """ Writes .json with FitSNAP formatting """
        full_write_path = os.path.join(write_path, self.group_name)
        os.makedirs(full_write_path, exist_ok=True)
        path_name = os.path.join(full_write_path, f'{self.label}.json')
        with open(path_name, 'w') as d:
            json.dump(self.to_dct(), d, indent=4)

    def write_xyz(self, write_path):
        """ Writes .xyz with FitSNAP formatting """
        os.makedirs(write_path, exist_ok=True)
        path_name = os.path.join(write_path, f'{self.group_name}.xyz')
        with open(path_name, 'a') as f:
            f.write(self.to_xyz())


class ReaxEntry(Entry):
    def __init__(self, group_name=None, label=None, structure=None, energy=None, charges=None, forces=None,
            distances=None, angles=None, dihedrals=None, lattice_vectors=None, **kwargs):

        super().__init__(group_name=group_name,
                         label=label, 
                         structure=structure, 
                         energy=energy, 
                         charges=charges, 
                         forces=forces, 
                         distances=distances, 
                         angles=angles, 
                         dihedrals=dihedrals, 
                         lattice_vectors=lattice_vectors, 
                         stress_vectors=None)

        self.write_zero = kwargs.get('write_zero', False)

        self._correct_units() # Yield the correct units for ReaxFF
        
        self.units_dct = {'energy': 'kcal/mol', 'charge': 'e',
                     'force': '(kcal/mol)/Angstrom', 'length': 'Angstrom',
                     'angle': 'degrees'}

    def __repr__(self):
        return f"ReaxEntry(label={self.label})"

    def from_dict(self, data):
        supported = ['structure', 'energy', 'charges', 'forces',
                     'distances', 'angles', 'dihedrals', 'lattice_vectors']
        for property_type in supported:
            if property_type in data:
                setattr(self, property_type, data[property_type])
        self._correct_units()
        return self
       
    def site_counts(self):
        if self.structure:
            site_counts = {}
            for i in range(len(self.structure)):
                if isinstance(self.structure, Structure):
                    specie = str(self.structure[i].specie)
                elif isinstance(self.structure, Atoms):
                    specie = str(self.structure[i].symbol)
                if specie in site_counts:
                    site_counts[specie] += 1
                else:
                    site_counts[specie] = 1
            return site_counts

    def structure_to_string(self, **kwargs):
        ''' Converts the passed structure object into a writable string '''
        return StructuretoString(self.structure, self.label, **kwargs).to_bgf_string()

    def trainsetin_section_header(self, property_type):
        if property_type in ['energy']:
            return "ENERGY\n#Weight Sign Label/Divisor... Relative Energy\n"
        elif property_type in ['charges', 'charge']:
            return "CHARGE\n#Label Weight BGFInd1 Charge\n"
        elif property_type in ['forces', 'force']:
            return "FORCES\n#Label Weight BGFInd1 Fx Fy Fz\n"
        elif property_type in ['distance', 'angle', 'dihedral', 'distances', 'angles', 'dihedrals']:
            return "GEOMETRY\n#Label Weight BGFInd1 BGFInd2 BGFInd3 BGFInd4 Value\n"
        elif property_type in ['lattice_vector', 'lattice_vectors']:
            return "CELL PARAMETERS\n#Label Weight Param Mag\n"

    def trainsetin_section_footer(self, property_type):
        if property_type in ['energy']:
            return "ENDENERGY\n"
        elif property_type in ['charge', 'charges']:
            return "ENDCHARGE\n"
        elif property_type in ['force', 'forces']:
            return "ENDFORCES\n"
        elif property_type in ['distance', 'angle', 'dihedral', 'distances', 'angles', 'dihedrals']:
            return "ENDGEOMETRY\n"
        elif property_type in ['lattice_vector', 'lattice_vectors']:
            return "ENDCELL PARAMETERS\n"

    def _get_weights(self, weights, collection, default_weights=None):
        if isinstance(collection, PropertyCollection):
            values = [np.linalg.norm(prop.value) for prop in collection.properties] # To handle forces
            if isinstance(weights, int) or isinstance(weights, float):
                weights = [float(weights) for w in range(len(collection.properties))]
            elif isinstance(weights, list):
                if len(weights) == len(collection.properties):
                    weights = weights
                else:
                    if default_weights is not None:
                        print(f'{len(weights)} weights != {len(collection.properties)}; setting with default {default_weights}')
                        weights = WeightedSampler(values, default_weights).sample()
                    else:
                        print(f'{len(weights)} weights != {len(collection.properties)} and default_weights = None; setting all = 1.0')
                        weights = [1.0 for i in range(len(collection.properties))]
            elif isinstance(weights, dict):
                weights = WeightedSampler(values, weights).sample() # Pass weights dict as arguments
            elif isinstance(weights, str): # Primarily for internal regex passing
                weights = [weights for w in range(len(collection.properties))]
            else:
                if default_weights is not None:
                    print(f'Weights {weights} not appropriate; setting with default {default_weights}')
                    weights = WeightedSampler(values, default_weights).sample()
                else:
                    print(f'Weights {weights} not appropriate and default_weights = None; setting all = 1.0')
                    weights = [1.0 for i in range(len(collection.properties))]
        return weights 

    def energy_to_string(self):
        print(f'Not supported for ReaxFF! Use self.relative_energy_to_string()')

    def get_relative_energy(self, add=[], subtract=[], get_divisors=False):
        reax_objs = [self] + add + subtract
        signs = ['+'] + ['+' for reax in add] + ['-' for reax in subtract]
        if get_divisors is True:
            divisors = self._get_divisors_to_write(add, subtract)
            if divisors is not None:
                energy = Energy(value=self._compute_total_energy(reax_objs, divisors, signs), unit=self.energy.unit)
            else:
                print(f"No divisors compute relative energies of ReaxObj {self.label} passed!")
                divisors = [None for reax_obj in reax_objs]
                energy = Energy(value=None, unit=self.energy.unit)
        else:
            divisors = [1 for reax_obj in reax_objs]
            energy = Energy(value=self._compute_total_energy(reax_objs, divisors, signs), unit=self.energy.unit)
        return reax_objs, signs, divisors, energy

    def relative_energy_to_string(self, weight=1, sig_figs=3, add=[], subtract=[], get_divisors=False):
        relative_energy_str = ''
        reax_objs, signs, divisors, relative_energy = self.get_relative_energy(add, subtract, get_divisors)
        if relative_energy.value is not None: # No appropriate divisors found
            if not (self.write_zero is False and weight == 0):
                relative_energy_str += self._relative_energy_substring(weight, relative_energy, reax_objs, divisors, signs, sig_figs)
        return relative_energy_str

    def _relative_energy_substring(self, weight, relative_energy, reax_objs, divisors, signs, sig_figs):
        relative_energy_str = f' {weight}'
        for i, reax_obj in enumerate(reax_objs):
            if divisors[i] != 0:
                relative_energy_str += f'   {signs[i]}   {reax_obj.label} /{divisors[i]}'
        rounded_energy = np.round(relative_energy.value, sig_figs)
        if math.isclose(rounded_energy, 0):
            rounded_energy = 1e-4
        relative_energy_str += f'       {rounded_energy}\n'
        return relative_energy_str

    def _compute_total_energy(self, reax_objs, divisors, signs):
        total_energy = 0
        for i, reax_obj in enumerate(reax_objs):
            if divisors[i] != 0:
                if signs[i] == '+':
                    total_energy += (reax_obj.energy.value / divisors[i])
                else:
                    total_energy += -1 * (reax_obj.energy.value / divisors[i])
        return total_energy

    def _integer_divisors(self, integer):
        divisors = []
        for divisor in range(1, integer + 1):
            if integer % divisor == 0: # Divisible
                divisors.append(divisor)
        return divisors

    def _composition_divisors(self, comp_dct):
        el_div_lst = []
        for el in list(comp_dct.keys()):
            el_div_lst.append(self._integer_divisors(comp_dct[el]))
        common_divisors = list(set.intersection(*map(set, el_div_lst)))
        common_divisors_sorted = sorted(common_divisors, reverse=True) # Largest to smallest
        return common_divisors_sorted

    def _sums_to_zero(self, divisor_combo, all_dcts, all_signs):
        final_dct = {}
        for comp_i, comp_divisor in enumerate(divisor_combo):
            if comp_divisor == 0: # Makes it so some relative energies can be ignored
                divided_dct = {k: 0 for k in all_dcts[comp_i].keys()}
            else:
                divided_dct = {k: (v / comp_divisor)*(all_signs[comp_i]) for k, v in all_dcts[comp_i].items()}
            final_dct = {k: final_dct.get(k, 0) + divided_dct.get(k, 0) for k in set(final_dct) | set(divided_dct)}

        if all(v == 0 for v in final_dct.values()): # All values equal zero
            return True
        else:
            return False

    def _evaluate_divisor_combo(self, args):
        """
        Worker function to evaluate a divisor combination.
        """
        divisor_combo, all_dcts, all_multipliers, fewest_refs, reax_objs, signs = args
        if not self._sums_to_zero(divisor_combo, all_dcts, all_multipliers):
            return None

        if fewest_refs:
            zero_count = list(divisor_combo).count(0)
            active_refs = len(divisor_combo) - zero_count
            sum_divisors = sum(divisor_combo)
            return {"divisor_combo": list(divisor_combo), "active_refs": active_refs, "sum_divisors": sum_divisors}

        relative_energy = self._compute_total_energy(reax_objs, divisor_combo, signs)
        normalized_energy = relative_energy / (len(self.structure) / divisor_combo[0])
        return {"divisor_combo": list(divisor_combo), "normalized_energy": normalized_energy}

    def _find_best_divisor_combo(self, all_dcts, all_multipliers, all_divisors, add_reax_lst, subtract_reax_lst, fewest_refs):
        """
        Optimized function to find the best divisor combo.
        """
        
        # Generate all possible divisor combinations
        divisor_combos = list(product(*all_divisors))

        # Prepare input arguments for multiprocessing
        reax_objs = [self] + add_reax_lst + subtract_reax_lst
        signs = ['+' if m == 1 else '-' for m in all_multipliers]
        pool_args = [(combo, all_dcts, all_multipliers, fewest_refs, reax_objs, signs) for combo in divisor_combos]
        
        # Use serial processing to evaluate combinations
        results = []
        for pool_arg in pool_args:
            results.append(self._evaluate_divisor_combo(pool_arg))

        # Filter out invalid results
        valid_results = [result for result in results if result is not None]
        
        if fewest_refs:
            # Sort first by active_refs (ascending), then by sum_divisors (descending)
            valid_results.sort(key=lambda x: (x["active_refs"], -x["sum_divisors"]))
            return valid_results[0]["divisor_combo"] if valid_results else None

        else:
            # Find the combo with the largest normalized relative energy
            valid_results.sort(key=lambda x: x["normalized_energy"], reverse=True)
            return valid_results[0]["divisor_combo"] if valid_results else None

    def _get_divisors_to_write(self, add_reax_lst, subtract_reax_lst, fewest_refs=True):
        all_dcts, all_multipliers, all_divisors = [], [], []

        all_dcts.append(self.site_counts())
        all_multipliers.append(1) # Treat these as positives
        all_divisors.append(self._composition_divisors(self.site_counts())) # Don't add zero to self
        for acd in add_reax_lst:
            a_site_counts = acd.site_counts()
            all_dcts.append(a_site_counts)
            all_multipliers.append(1)
            all_divisors.append(acd._composition_divisors(a_site_counts) + [0]) # Add zero so these can be ignored
        for scd in subtract_reax_lst:
            s_site_counts = scd.site_counts()
            all_dcts.append(s_site_counts)
            all_multipliers.append(-1)
            all_divisors.append(scd._composition_divisors(s_site_counts) + [0]) # Add zero so these can be ignored
        
        divisor = self._find_best_divisor_combo(all_dcts, all_multipliers, all_divisors, add_reax_lst, subtract_reax_lst, fewest_refs)
        return divisor

    def charges_to_string(self, weights=None, default_weights=None, sig_figs=3):
        charges_string = ''
        if self.charges is not None:
            weights = self._get_weights(weights, self.charges, default_weights)
            for charge_ind, charge in enumerate(self.charges.properties):
                if not (self.write_zero is False and weights[charge_ind] == 0):
                    rounded_charge = np.round(charge.value, sig_figs)
                    charges_string += f"{self.label} {weights[charge_ind]} {charge.indice+1} {rounded_charge}\n"
        return charges_string

    def forces_to_string(self, weights=None, default_weights=None, sig_figs=3):
        forces_string = ''
        if self.forces is not None:
            weights = self._get_weights(weights, self.forces, default_weights)
            for force_ind, force in enumerate(self.forces.properties):
                if not (self.write_zero is False and weights[force_ind] == 0):
                    rounded_forces = np.round(force.vector, sig_figs)
                    atom_forces_string = ''.join([str(rounded_force) + ' ' for rounded_force in rounded_forces])
                    forces_string += f"{self.label}  {weights[force_ind]}  {force.indice+1}  {atom_forces_string}\n"
        return forces_string

    def geometries_to_string(self, geometry_type, weights=None, default_weights=None, sig_figs=3):
        geometry_string = ''
        
        if geometry_type in ['distance', 'distances']:
            props = self.distances
        elif geometry_type in ['angle', 'angles']:
            props = self.angles
        elif geometry_type in ['dihedral', 'dihedrals']:
            props = self.dihedrals
        
        if props is not None:
            weights = self._get_weights(weights, props, default_weights)
            for property_ind, prop in enumerate(props.properties):
                if not (self.write_zero is False and weights[property_ind] == 0):
                    rounded_value = np.round(prop.value, sig_figs)
                    add_one_indices = ''.join([str(indice+1) + '   ' for indice in prop.indices])
                    geometry_string += f"{self.label}  {weights[property_ind]}   {add_one_indices}   {rounded_value}\n"
        return geometry_string

    def lattice_vectors_to_string(self, weights=None, default_weights=None, sig_figs=3):
        lattice_vector_string = ''
        if self.lattice_vectors is not None:
            weights = self._get_weights(weights, self.lattice_vectors, default_weights)
            for lattice_vector_ind, lattice_vector in enumerate(self.lattice_vectors.properties):
                if not (self.write_zero is False and weights[lattice_vector_ind] == 0):
                    rounded_lattice_vector = np.round(lattice_vector.value, sig_figs)
                    lattice_vector_string += f"{self.label}  {weights[lattice_vector_ind]}    {lattice_vector.parameter}  {rounded_lattice_vector}\n"
        return lattice_vector_string

    def get_property_string(self, property_type, **kwargs):
        if property_type == 'energy':
            return self.relative_energy_to_string(**kwargs)
        elif property_type in ['charge', 'charges']:
            return self.charges_to_string(**kwargs)
        elif property_type in ['force', 'forces']:
            return self.forces_to_string(**kwargs)
        elif property_type in ['distance', 'distances']:
            return self.geometries_to_string(geometry_type=property_type, **kwargs)
        elif property_type in ['angle', 'angles']:
            return self.geometries_to_string(geometry_type=property_type, **kwargs)
        elif property_type in ['dihedral', 'dihedrals']:
            return self.geometries_to_string(geometry_type=property_type, **kwargs)
        elif property_type in ['lattice_vector', 'lattice_vectors']:
            return self.lattice_vectors_to_string(**kwargs)

