from abc import ABC, abstractmethod
from parse2fit.tools.fileconverter import StructuretoString 
from parse2fit.core.properties import Energy, PropertyCollection
from parse2fit.tools.weights import WeightedSampler
from parse2fit.tools.unitconverter import UnitConverter
from pymatgen.core.structure import Structure
from ase import Atoms
import numpy as np
import itertools

# Base class for Entries
class Entry(ABC):
    def __init__(self, label, structure=None, energy=None, charges=None, forces=None, 
            distances=None, angles=None, dihedrals=None, lattice_vectors=None, **kwargs):
        
        self.label = label
        self.structure = structure
        self.energy = energy
        self.charges = charges
        self.forces = forces
        self.distances = distances
        self.angles = angles
        self.dihedrals = dihedrals
        self.lattice_vectors = lattice_vectors

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def from_dict(self):
        pass

    @abstractmethod
    def structure_to_string(self):
        pass

    @abstractmethod
    def energy_to_string(self):
        pass

    @abstractmethod
    def charges_to_string(self):
        pass

    @abstractmethod
    def forces_to_string(self):
        pass

    @abstractmethod
    def geometries_to_string(self):
        pass

    @abstractmethod
    def lattice_vectors_to_string(self):
        pass


class ReaxEntry(Entry):
    def __init__(self, label, structure=None, energy=None, charges=None, forces=None,
            distances=None, angles=None, dihedrals=None, lattice_vectors=None, **kwargs):
        super().__init__(label, structure, energy, 
                         charges, forces, distances, 
                         angles, dihedrals, lattice_vectors)

        self._correct_units() # Yield the correct units for ReaxFF

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
       
    def _correct_units(self):
        units_dct = {'energy': 'kcal/mol', 'charge': 'e', 
                     'force': '(kcal/mol)/Angstrom', 'length': 'Angstrom', 
                     'angle': 'degrees'}
        if self.energy:
            self.energy.convert_units('energy', units_dct['energy'])
        if self.charges:
            for charge in self.charges.properties:
                charge.convert_units('charge', units_dct['charge'])
        if self.forces:
            for force in self.forces.properties:
                force.convert_units('force', units_dct['force'])
        if self.distances:
            for distance in self.distances.properties:
                distance.convert_units('length', units_dct['length'])
        if self.angles:
            for angle in self.angles.properties:
                angle.convert_units('angle', units_dct['angle'])
        if self.dihedrals:
            for dihedral in self.dihedrals.properties:
                dihedral.convert_units('angle', units_dct['angle'])
        if self.lattice_vectors:
            uc = UnitConverter()
            for lattice_vector in self.lattice_vectors.properties:
                lattice_vector.convert_units('length', units_dct['length'])
                lattice_vector.vector = uc.convert(lattice_vector.vector, lattice_vector.unit, 
                                                   units_dct['length'], 'length')
            
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
                print(f"No divisors compute relative energies of ReaxObjs passed!")
                divisors = [None for reax_obj in reax_objs]
                energy = Energy(value=None, unit=self.energy.unit)
        else:
            divisors = [1 for reax_obj in reax_objs]
            energy = Energy(value=self._compute_total_energy(reax_objs, divisors, signs), unit=self.energy.unit)
        return reax_objs, signs, divisors, energy

    def relative_energy_to_string(self, weight=1, sig_figs=3, add=[], subtract=[], get_divisors=False):
        relative_energy_str = ''
        reax_objs, signs, divisors, relative_energy = self.get_relative_energy(add, subtract, get_divisors)
        if relative_energy.value is not None and weight != 0: # No appropriate divisors found
            relative_energy_str += self._relative_energy_substring(weight, relative_energy, reax_objs, divisors, signs, sig_figs)
        return relative_energy_str

    def _relative_energy_substring(self, weight, relative_energy, reax_objs, divisors, signs, sig_figs):
        relative_energy_str = f' {weight}'
        for i, reax_obj in enumerate(reax_objs):
            relative_energy_str += f'   {signs[i]}   {reax_obj.label} /{divisors[i]}'
        rounded_energy = np.round(relative_energy.value, sig_figs)
        relative_energy_str += f'       {rounded_energy}\n'
        return relative_energy_str

    def _compute_total_energy(self, reax_objs, divisors, signs):
        total_energy = 0
        for i, reax_obj in enumerate(reax_objs):
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
        return common_divisors

    def _sums_to_zero(self, divisor_combo, all_dcts, all_signs):
        final_dct = {}
        for comp_i, comp_divisor in enumerate(divisor_combo):
            divided_dct = {k: (v / comp_divisor)*(all_signs[comp_i]) for k, v in all_dcts[comp_i].items()}
            final_dct = {k: final_dct.get(k, 0) + divided_dct.get(k, 0) for k in set(final_dct) | set(divided_dct)}

        if all(v == 0 for v in final_dct.values()): # All values equal zero
            return True
        else:
            return False

    def _get_divisors_to_write(self, add_reax_lst, subtract_reax_lst):
        all_dcts, all_multipliers, all_divisors = [], [], []

        all_dcts.append(self.site_counts())
        all_divisors.append(self._composition_divisors(self.site_counts()))
        all_multipliers.append(1) # Treat these as positives
        for acd in add_reax_lst:
            a_site_counts = acd.site_counts()
            all_dcts.append(a_site_counts)
            all_multipliers.append(1)
            all_divisors.append(acd._composition_divisors(a_site_counts))
        for scd in subtract_reax_lst:
            s_site_counts = scd.site_counts()
            all_dcts.append(s_site_counts)
            all_multipliers.append(-1)
            all_divisors.append(scd._composition_divisors(s_site_counts))
        divisor_combos = list(reversed(list(itertools.product(*all_divisors)))) # Largest divisors first

        use_divisors = None
        for divisor_combo in divisor_combos:
            if self._sums_to_zero(divisor_combo, all_dcts, all_multipliers) is True:
                use_divisors = list(divisor_combo) # Take the first one found
                break

        return use_divisors

    def charges_to_string(self, weights=None, default_weights=None, sig_figs=3):
        charges_string = ''
        if self.charges is not None:
            weights = self._get_weights(weights, self.charges, default_weights)
            for charge_ind, charge in enumerate(self.charges.properties):
                if weights[charge_ind] != 0:
                    rounded_charge = np.round(charge.value, sig_figs)
                    charges_string += f"{self.label} {weights[charge_ind]} {charge.indice+1} {rounded_charge}\n"
        return charges_string

    def forces_to_string(self, weights=None, default_weights=None, sig_figs=3):
        forces_string = ''
        if self.forces is not None:
            weights = self._get_weights(weights, self.forces, default_weights)
            for force_ind, force in enumerate(self.forces.properties):
                if weights[force_ind] != 0:
                    rounded_forces = np.round(force.value, sig_figs)
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
                if weights[property_ind] != 0:
                    rounded_value = np.round(prop.value, sig_figs)
                    add_one_indices = ''.join([str(indice+1) + '   ' for indice in prop.indices])
                    geometry_string += f"{self.label}  {weights[property_ind]}   {add_one_indices}   {rounded_value}\n"
        return geometry_string

    def lattice_vectors_to_string(self, weights=None, default_weights=None, sig_figs=3):
        lattice_vector_string = ''
        if self.lattice_vectors is not None:
            weights = self._get_weights(weights, self.lattice_vectors, default_weights)
            for lattice_vector_ind, lattice_vector in enumerate(self.lattice_vectors.properties):
                if weights[lattice_vector_ind] != 0:
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

