from abc import ABC, abstractmethod
from parse2fit.core.entries import ReaxEntry, FitSNAPEntry
from parse2fit.io.parsers import ParserFactory
from parse2fit.tools.weights import WeightedSampler
from collections import defaultdict
import warnings
import yaml
from pathlib import Path
import os
import sys
from multiprocessing import Pool
from copy import deepcopy
import re
import numpy as np
import logging

warnings.filterwarnings('ignore')

class YamlManager(ABC):
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self):
        if not self.config_path.exists():
            return {}
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file) or {}

    def save_config(self):
        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config, file)


class ReadWriteFactory(YamlManager):
    def __init__(self, config_path):
        super().__init__(config_path)

    def get_writer(self):
        if not self.config.get('output_format'):
            raise ValueError(f"No 'output_format' found in {self.config_path}")
        elif self.config.get('output_format') in ['jax-reaxff', 'reaxff', 'ReaxFF']:
            return ReaxRW(self.config, ReaxEntry)
        elif self.config.get('output_format') in ['FitSNAP', 'fitsnap']:
            return FitSNAPRW(self.config, FitSNAPEntry)

    def write_output(self, yaml_path):
        pass


class RW(ABC):
    def __init__(self, config_dct, entry):
        self.entry = entry
        self.config_dct = config_dct
        self.generation_parameters = self.config_dct.get('generation_parameters', {})
        self.method_parameters = self.config_dct.get('method_parameters', {})
        self.input_paths = self._get_absolute_paths()
        #self.input_paths = self.config_dct.get('input_paths', {})

    @abstractmethod
    def _load_input_files(self):
        pass

    @abstractmethod
    def _get_property_weights(self):
        pass

    @abstractmethod
    def _get_unique_files(self):
        pass

    @abstractmethod
    def _write_unique_files(self):
        pass

    @abstractmethod
    def _get_and_write_repeatable_files(self):
        pass

    @abstractmethod
    def write_input_files(self):
        pass

    def _get_absolute_paths(self):
        absolute_input_paths = deepcopy(self.config_dct.get('input_paths', {}))

        for label, label_dct in absolute_input_paths.items():
            label_dct['directories'] = [os.path.abspath(d) for d in label_dct.get('directories', [])]

            if isinstance(label_dct['energy'], dict):
                energy_dct = label_dct.get('energy', {})
                for key in ('add', 'subtract'):
                    if isinstance(energy_dct.get(key), list):
                        energy_dct[key] = [os.path.abspath(p) for p in energy_dct[key]]

        return absolute_input_paths

    def read_file(self, path, default_value=''):
        if os.path.exists(path):
            return self.file_to_string(path)
        logging.warning(f"File {path} not found! Using empty default.")
        return default_value

    def file_to_string(self, filepath):
        with open(filepath, "r") as text_file:
            data = text_file.read()
        return data

    def string_to_file(self, filepath, write_string):
        with open(filepath, "w") as text_file:
            text_file.write(write_string)
        return

    def get_unique_tags(self, dct, tag):
        unique_labels = set()

        for key, subdct in dct.items():
            if tag in subdct and isinstance(subdct[tag], list):
                unique_labels.update(subdct[tag])
        return list(unique_labels)

    def _set_default_weights(self, weight_dct):
        """Sets default weights for various properties."""
        default = {'split': 0.0, 'min': 0.0, 'max': 1.0, 'type': 'binary'}
        return {prop: weight_dct.get(prop, default) if isinstance(weight_dct.get(prop), dict) else default
                for prop in self.write_options}

    def _default_labeling(self, path, label='', depth=3):
        """Generates default labels by iterating up the directory hierarchy."""
        def sanitize_filename(filename, replacement="_"):
            # Define a regex pattern for invalid characters
            pattern = r'[\/:*?"<>|\' .]'  # Matches all restricted characters
            return re.sub(pattern, replacement, filename)

        if depth == 0:
            return label
        new_label = os.path.basename(path) if not label else f"{os.path.basename(path)}_{label}"
        new_label = sanitize_filename(new_label)
        return self._default_labeling(os.path.dirname(path), new_label, depth - 1)

    def _get_path_dictionary_values(self, group_name):
        configuration_path = self.input_paths.get(group_name, {})
        path_dct = {'group_name': group_name, 'pymatgen_structure': True}

        for prop in self.write_options:
            value = configuration_path.get(prop, None)
            if prop == 'energy':
                if value is None or value is True:
                    path_dct[prop] = {'weights': self.default_weights['energy']}
                elif isinstance(value, dict):
                    value.setdefault('weights', self.default_weights['energy'])
                    if isinstance(value['weights'], int):
                        value['weights'] = float(value['weights'])
                    path_dct[prop] = value
                else:
                    logging.warning(f"Invalid value {value} for {prop}; defaulting to True")
                    path_dct[prop] = {'weights': self.default_weights['energy']}

            else:
                if value is None or value is False:
                    path_dct[prop] = False
                elif value is True:
                    path_dct[prop] = {'weights': self.default_weights[prop]}
                elif isinstance(value, dict):
                    value.setdefault('weights', self.default_weights[prop])
                    if isinstance(value['weights'], list):
                        value['weights'] = [float(val) for val in value['weights']]
                    path_dct[prop] = value
                else:
                    logging.warning(f"Invalid value {value} for {prop}; defaulting to False")
                    path_dct[prop] = False

        return path_dct

    def _get_all_paths(self, paths, group_name, walk_roots=False):
        group_name_dictionary = self._get_path_dictionary_values(group_name)
        
        all_paths_dct = {}
        for path in paths:
            if walk_roots:
                for root, _, _, in os.walk(os.path.abspath(path), topdown=True):
                    all_paths_dct[root] = deepcopy(group_name_dictionary)
                    all_paths_dct[root]['label'] = self._default_labeling(root)
            else:
                all_paths_dct[path] = deepcopy(group_name_dictionary)
                all_paths_dct[path]['label'] = self._default_labeling(path)

        return all_paths_dct 

    def _path_root_dictionary(self):
        """Returns a dictionary with the following format:
           {input_root_path: {'directory': top_directory, 'group_name': group_name}}
        """
        path_root_dictionary = {}

        def append_to_dct(dct, paths, group_name, walk_roots):
            all_paths_dct = self._get_all_paths(paths, group_name, walk_roots=walk_roots)
            for root_path, directory_dct in all_paths_dct.items():
                directory_dct['group_name'] = group_name
                dct[root_path] = directory_dct
            return dct

        # Get the low-level paths first, so that they can be superseded
        for group_name, group_dct in self.input_paths.items():
            paths = []
            try:
                paths += group_dct['energy'].get('add', [])
            except (AttributeError, KeyError):
                pass
            try:
                paths += group_dct['energy'].get('subtract', [])
            except (AttributeError, KeyError):
                pass
            paths = sorted(paths, key=len, reverse=True)
            append_to_dct(path_root_dictionary, paths, group_name, walk_roots=False)

        # Now, get the higher level paths
        for group_name, group_dct in self.input_paths.items():
            paths = []
            try:
                paths += group_dct.get('directories', [])
            except KeyError:
                raise KeyError(f'Must supply "directories" tag for {group_name}!')
            paths = sorted(paths, key=len, reverse=True) # Shorter paths first for specificity
            append_to_dct(path_root_dictionary, paths, group_name, walk_roots=True)

        return path_root_dictionary

    def _dct_parser(self, path_dct, path):
        """Creates the Entry Object for each Path"""
        def true_if(val):
            if val is None or val is False:
                return False
            else:
                return True

        kwargs_dct = {
                      'pymatgen_structure': true_if(path_dct[path].get('pymatgen_structure')),
                      'periodic_geometries': true_if(self.generation_parameters.get('periodic_geometries')),
                      'energy': true_if(path_dct[path].get('energy')),
                      'charges': true_if(path_dct[path].get('charges')),
                      'forces': true_if(path_dct[path].get('forces')),
                      'distances': true_if(path_dct[path].get('distances')),
                      'angles': true_if(path_dct[path].get('angles')),
                      'dihedrals': true_if(path_dct[path].get('dihedrals')),
                      'lattice_vectors': true_if(path_dct[path].get('lattice_vectors')),
                      'stress_vectors': true_if(path_dct[path].get('stress_vectors')),
                    }
        parsed_dct = ParserFactory.create_parser(path, **kwargs_dct)

        if parsed_dct:
            new_entry = deepcopy(self.entry)(group_name=path_dct[path].get('group_name'), 
                                             label=path_dct[path].get('label')) 
            return path, new_entry.from_dict(data=parsed_dct.parse_all())
        else:
            return path, None

    def _build_objects_dct_by_root(self):
        # Step 1: construct the path_root_dictionary
        root_dct = self._path_root_dictionary()

        # Step 2: Parallelize parsing using multiprocessing
        with Pool() as pool:
            results = pool.starmap(self._dct_parser, [(root_dct, key) for key in root_dct.keys()])

        # Step 3: Aggregate results
        for i, result in enumerate(results):
            if result[1]:  # Ignore Nonetype Entry objects
                root_dct[result[0]]['entry'] = result[1]
            else: # Remove Nonetype Entry objects
                root_dct.pop(result[0])

        # Step 4: Replace add and subtract strings with the entry objects
        def replace_add_subtracts(full_dictionary):
            for root_path, root_dict in full_dictionary.items():
                for op in ['subtract', 'add']:
                    if op in root_dict.get('energy', {}):
                        path_list = root_dict['energy'][op]
                        root_dict['energy'][op] = [full_dictionary[rp]['entry'] for rp in path_list]
            return full_dictionary

        return replace_add_subtracts(root_dct)

    def _build_objects_dct_by_groupname(self, root_dct):
        # Takes the unique objects dictionary built and 
        #root_dct = self._build_objects_dct_by_root()
        groupname_dct = {}

        for root, root_dct in root_dct.items():
            group_name = root_dct['group_name']
            if group_name not in groupname_dct.keys():
                groupname_dct[group_name] = {}
            new_dct = {k: v for k, v in root_dct.items() if k != 'group_name'}
            groupname_dct[group_name][root] = new_dct
        
        return groupname_dct
    
    def _build_objects_dct_by_weights(self, root_dct):
        # Iterate the dictionary and assign by weighting scheme
        weights_dct_keys = {}
        weights_dct = {}
        count = 0

        def get_key_from_value(my_dict, target_value):
            for key, value in my_dict.items():
                if value == target_value:
                    return key
            return None

        for root, root_dct_entry in root_dct.items():  # Avoid overwriting root_dct
            for write_option in self.write_options:
                try:
                    weights = root_dct_entry[write_option]['weights']
                except (KeyError, TypeError):  # write_option not present or False
                    continue

                weight_key = get_key_from_value(weights_dct_keys, weights)
                if not weight_key:  # Doesn't exist in the dictionary
                    weight_key = f'weights_{count}'
                    count += 1
                    weights_dct_keys[weight_key] = weights
                    weights_dct[weight_key] = {}

                # Ensure write_option is initialized
                weights_dct[weight_key].setdefault(write_option, {})

                # Store a copy of root_dct_entry to prevent overwriting
                weights_dct[weight_key][write_option][root] = deepcopy(root_dct_entry)

        return weights_dct_keys, weights_dct

    def get_property_weights_dct(self, root_dct):
        """
        Builds a dictionary structure for property weights:
        {property: {'labels': [], 'values': [], 'weights': []}}
        """
        weight_objects_dct_keys, weight_objects_dct = self._build_objects_dct_by_weights(root_dct)
        weights_dct = {}

        for weight_key, write_options in weight_objects_dct.items():
            for write_option, paths in write_options.items():
                # Initialize the write_option in weights_dct if not present
                weights_dct.setdefault(write_option, {'labels': [], 'values': [], 'weights': []})
                
                labels, values = self._get_property_weights(write_option, paths)
                weights_v = weight_objects_dct_keys[weight_key]
                
                # Determine weights based on type of weights_v
                if isinstance(weights_v, dict):
                    try:
                        weights = list(WeightedSampler(values, weights_v).sample())
                    except TypeError:
                        print(f"TypeError for {weights_v.get('type', 'Unknown')}; defaulting to weights of {weights_v.get('max', 1.0)}")
                        weights = [float(weights_v.get('max', 1.0))] * len(labels)
                elif isinstance(weights_v, (float, int)):
                    weights = [float(weights_v)] * len(labels)
                else:
                    weights = [1.0] * len(labels)

                # Update weights_dct for the current write_option
                weights_dct[write_option]['labels'].extend(labels)
                weights_dct[write_option]['values'].extend(values)
                weights_dct[write_option]['weights'].extend(weights)

        return weights_dct

    def _write_unique_inputs(self, attempts_multiplier=5, **kwargs):
        runs_to_generate = self.generation_parameters['runs_to_generate']
        output_directory = self.generation_parameters['output_directory']

        unique_files_set = set()  # Use a set instead of a list for fast lookups
        attempts = 0
        max_attempts = attempts_multiplier * 1 # runs_to_generate
        
        while len(unique_files_set) < runs_to_generate:
             if attempts > max_attempts:
                 break

             unique_files = self._get_unique_files(**kwargs)

             if unique_files not in unique_files_set:
                 unique_files_set.add(unique_files)
                 subdirectory = f"{self.config_dct.get('output_format')}_run_{len(unique_files_set) - 1}"
                 write_directory = os.path.join(output_directory, subdirectory)
                 os.makedirs(write_directory, exist_ok=True)

                 self._write_unique_files(write_directory, unique_files)
                 self._get_and_write_repeatable_files(write_directory, **kwargs)
                 attempts = 0
             attempts += 1

        print(f'{len(unique_files_set)} input files written!')
        
        return 


class FitSNAPRW(RW):
    def __init__(self, config_dct, entry):
        super().__init__(config_dct, entry)
        self.entry = entry
        self.write_options = ['energy', 'forces', 'lattice_vectors', 'stress_vectors']
        self.default_weights = self._set_default_weights(self.config_dct.get('default_weights', {}))
        
        self.fitsnapin = self._load_input_files()

    def _load_input_files(self):
        """Loads fitsnap.in file if input_directory is specified."""
        input_dir = self.generation_parameters.get('input_directory', '')
        fitsnap_path = os.path.join(input_dir, 'fitsnap.in')

        return self.read_file(fitsnap_path)

    def _get_property_weights(self, write_option, paths):
        labels, values = [], []
        for path, path_data in paths.items():
            group_name = path_data['group_name']

            # Get index for the group_name or add it to labels and values
            try:
                add_index = labels.index(group_name)
            except ValueError:
                add_index = len(labels)
                labels.append(group_name)
                values.append(0)

            values[add_index] += 1
        return labels, values


    def get_label_weights_dct(self, property_weights_dct):
        weights_dct = {}
        unique_labels = self.get_unique_tags(property_weights_dct, 'labels')
        
        for write_option in self.write_options:
            for label in unique_labels:
                if label not in weights_dct.keys():
                    weights_dct[label] = {}
                weights_dct[label][write_option] = float(0)
                if write_option not in property_weights_dct.keys():
                    weights_dct[label][write_option] = float(0)
                elif label in property_weights_dct[write_option]['labels']:
                    try:
                        label_index = property_weights_dct[write_option]['labels'].index(label)
                        weight = property_weights_dct[write_option]['weights'][label_index]
                    except IndexError:
                        weight = float(0)
                    weights_dct[label][write_option] = weight 
                
        return weights_dct

    def get_group_section(self, label_weights_dct):
        group_section_string = "[GROUPS]\n"
        group_section_string += "group_sections = name training_size testing_size weight fweight vweight\n"
        group_section_string += "group_types = str float float float float float\n"
        
        if 'groups' in self.method_parameters.items():
            for method_parameter, parameter in self.method_parameters['groups'].items():
                group_section_string += f'{method_parameter} = {str(parameter)}\n'

        for label in label_weights_dct.keys():
            ew = label_weights_dct[label]['energy']
            fw = label_weights_dct[label]['forces']
            vw = label_weights_dct[label]['stress_vectors']
            group_section_string += f"{label} = 1.0    0.0   {ew}    {fw}    {vw}\n"

        group_section_string += "\n"

        return group_section_string

    def get_input_string(self, weights_dct):
        """ Add to generation string """
        copied_string = deepcopy(self.fitsnapin)
        methods_copy = deepcopy(self.method_parameters)

        """Write [GROUP] values first"""
        copied_string += self.get_group_section(weights_dct)
        try:
            methods_copy.pop('groups')
        except KeyError: 
            pass
        
        for method_parameter_key, method_parameter_key_dct in methods_copy.items():
            copied_string += f'[{method_parameter_key.upper()}]\n'
            for method_parameter, parameter in method_parameter_key_dct.items():
                copied_string += f'{method_parameter} = {str(parameter)}\n'

            copied_string += "\n"

        return copied_string

    def _get_unique_files(self, **kwargs):
        """ Gets the .in string with weighting scheme from root dictionary"""
        weights_dct = self.get_property_weights_dct(kwargs['root_dictionary'])
        label_weights_dct = self.get_label_weights_dct(weights_dct)
        
        return self.get_input_string(weights_dct=label_weights_dct)

    def _write_unique_files(self, write_directory, unique_files):
        """ Unique files are only the .in string """
        write_path = os.path.join(write_directory, 'fitsnap.in')
        
        return self.string_to_file(write_path, unique_files)

    def _get_and_write_repeatable_files(self, write_directory, **kwargs):
        """ How to write the repeatable files """
        filetype = 'json'
        if "scraper" in self.method_parameters.keys() and "scraper" in self.method_parameters["scraper"].keys():
            if self.method_parameters["scraper"]["scraper"] == 'xyz':
                filetype = 'xyz'
        
        if "path" in self.method_parameters.keys() and "dataPath" in self.method_parameters['path'].keys():
            write_to = os.path.join(write_directory, self.method_parameters["path"]["dataPath"])
        else:
            write_to = os.path.join(write_directory, filetype)

        for root_key, root_key_dct in kwargs['root_dictionary'].items():
            entry = root_key_dct['entry']
            if filetype == 'json':
                entry.write_dct(write_to)
            elif filetype == 'xyz':
                entry.write_xyz(write_to)

        return 
        
    def write_input_files(self, attempts_multiplier=5):
        root_dictionary = self._build_objects_dct_by_root()
        kwargs = {'root_dictionary': root_dictionary}
        self._write_unique_inputs(attempts_multiplier=attempts_multiplier, **kwargs) 
        
        return 


class ReaxRW(RW):
    def __init__(self, config_dct, entry):
        super().__init__(config_dct, entry)
        self.entry = entry
        self.write_options = ['charges', 'lattice_vectors', 'forces',
                              'distances', 'angles', 'dihedrals', 'energy']
        self.default_weights = self._set_default_weights(self.config_dct.get('default_weights', {}))

        self.geo, self.trainsetin = self._load_input_files()

    def _load_input_files(self):
        """Loads geo and trainset.in files if input_directory is specified."""
        input_dir = self.generation_parameters.get('input_directory', '')
        geo_path = os.path.join(input_dir, 'geo')
        trainsetin_path = os.path.join(input_dir, 'trainset.in')
        
        return self.read_file(geo_path), self.read_file(trainsetin_path)

    def _get_property_weights(self, write_option, paths):
        labels, values = [], []
        for path, path_data in paths.items():
            reax_obj = path_data['entry']
            if write_option == 'energy':
                add_objects = path_data[write_option].get('add', []) 
                subtract_objects = path_data[write_option].get('subtract', [])
                get_divisors = path_data[write_option].get('get_divisors', False)

                if add_objects or subtract_objects:
                    if reax_obj not in add_objects + subtract_objects:
                        reax_objs, signs, divisors, rel_energy_obj = reax_obj.get_relative_energy(add=add_objects, 
                                                                   subtract=subtract_objects,
                                                                   get_divisors=get_divisors)
                        labels.append(path)
                        values.append(rel_energy_obj.value)
            else:
                property_collection = getattr(reax_obj, write_option)
                if property_collection:
                    new_values = [prop.value for prop in property_collection.properties]
                    new_labels = [path for i in range(len(new_values))]
                
                    labels += new_labels
                    values += new_values
        
        return labels, values

    def get_input_string(self, weights_dct, **kwargs):
        """ Add to generation string """
        copied_string = deepcopy(self.trainsetin)
        empty_entry = ReaxEntry()
    
        header_footer_lst = ['charges', 'lattice_vectors', 'forces', 'distances', 'energy']
        header_footer = defaultdict(lambda: {'header': False, 'footer': False}, 
                                {key: {'header': False, 'footer': key == 'distances'} for key in header_footer_lst})
    
        geometry_types = [gt for gt in ['distances', 'angles', 'dihedrals'] if gt in weights_dct]
    
        for prop_key in self.write_options:
            if prop_key not in weights_dct:
                continue
        
            check_key = 'distances' if prop_key in ['distances', 'angles', 'dihedrals'] else prop_key
        
            # Header writing
            if not header_footer[check_key]['header']:
                copied_string += empty_entry.trainsetin_section_header(check_key)
                header_footer[check_key]['header'] = True 
        
            prop_key_dct = weights_dct[prop_key]
            unique_paths = np.unique(prop_key_dct['labels'])
        
            for unique_path in unique_paths:
                root_dict = kwargs['root_dictionary'][unique_path]
                reax_entry = root_dict['entry']
                property_weights = [w for i, w in enumerate(prop_key_dct['weights']) 
                                if prop_key_dct['labels'][i] == unique_path]
            
                # Use a dictionary mapping for method calls
                method_map = {
                    'energy': lambda: reax_entry.relative_energy_to_string(
                        weight=property_weights[0],
                        add=root_dict[prop_key].get('add', []),
                        subtract=root_dict[prop_key].get('subtract', []),
                        get_divisors=root_dict[prop_key].get('get_divisors', False)
                    ),
                    'lattice_vectors': lambda: reax_entry.lattice_vectors_to_string(weights=property_weights),
                    'forces': lambda: reax_entry.forces_to_string(weights=property_weights),
                    'charges': lambda: reax_entry.charges_to_string(weights=property_weights),
                    'distances': lambda: reax_entry.geometries_to_string(geometry_type=prop_key, weights=property_weights),
                    'angles': lambda: reax_entry.geometries_to_string(geometry_type=prop_key, weights=property_weights),
                    'dihedrals': lambda: reax_entry.geometries_to_string(geometry_type=prop_key, weights=property_weights),
                }
            
                copied_string += method_map[prop_key]()

                if prop_key in ['distances', 'angles', 'dihedrals'] and prop_key == geometry_types[-1]:
                    header_footer['distances']['footer'] = False  

            # Footer writing
            if not header_footer[check_key]['footer']:
                copied_string += empty_entry.trainsetin_section_footer(check_key)
                header_footer[check_key]['footer'] = True

        return copied_string

    def _get_unique_files(self, **kwargs):
        weights_dct = self.get_property_weights_dct(kwargs['root_dictionary'])
        
        return self.get_input_string(weights_dct, **kwargs)

    def _write_unique_files(self, write_directory, unique_files):
        """ Unique files are only the .in string """
        write_path = os.path.join(write_directory, 'trainset.in')
        
        return self.string_to_file(write_path, unique_files)

    def _get_and_write_repeatable_files(self, write_directory, **kwargs):
        write_string = deepcopy(self.geo) # Copy the geo file string
        for path, path_dct in kwargs['root_dictionary'].items():
            reax_obj = path_dct['entry']
            reax_obj_str = reax_obj.structure_to_string(**path_dct.get('structure', {}))
            if reax_obj_str not in write_string: # Check to see if full string is present
                write_string += reax_obj_str
        write_path = os.path.join(write_directory, 'geo')
        
        return self.string_to_file(write_path, write_string)

    def write_input_files(self, attempts_multiplier=5):
        root_dictionary = self._build_objects_dct_by_root()
        kwargs = {'root_dictionary': root_dictionary}
        self._write_unique_inputs(attempts_multiplier=attempts_multiplier, **kwargs)
        
        return


    # Old input file modification code; could re-introduce
    '''
    def _modify_string(self, main_str, check_str, sub_str, insert_point_str=None, placeholder='weight', pattern=r"(\d+\.\d*)"):
        """ Modifies a string by replacing or inserting values based on regex matching. """

        # Escape special regex characters and replace placeholder
        regex_pattern = re.escape(check_str).replace(placeholder, pattern)
        compiled_pattern = re.compile(regex_pattern)

        match = compiled_pattern.search(main_str)

        if match:
            if self.generation_parameters.get('fix_input_weights', True):
                return main_str  # No modifications if weights are fixed

            if match.group() != sub_str:  # If actual replacement is needed
                weight_match = compiled_pattern.search(sub_str)
                weight = float(weight_match.group(1)) if weight_match else None

                if weight == 0:
                    return main_str.replace(match.group(), '')  # Remove zero-weight entries
                else:
                    return main_str.replace(match.group(), sub_str)  # Replace with new string
            return main_str  # No change needed
        else:
            # Handle case where check_str isn't found in main_str
            weight_match = compiled_pattern.search(sub_str)
            try:
                weight = float(weight_match.group(1))
            except (AttributeError, IndexError):
                weight = None

            if weight == 0:
                return main_str  # Don't insert zero-weight strings

            if insert_point_str:
                pos = main_str.find(insert_point_str)
                if pos != -1:
                    return main_str[:pos + len(insert_point_str)] + sub_str + main_str[pos + len(insert_point_str):]
            return main_str + sub_str  # Default append if no insert point

    def _split_lines_and_modify(self, main_str, regex_str, weights_str, insert_point_str=None, placeholder='weight', pattern=r"(\d+\.\d*)"):
        """ Splits strings line by line and modifies matching sections. """

        regex_list = [r + n for r, n in zip(re.split(r'(\n)', regex_str)[::2], re.split(r'(\n)', regex_str)[1::2])]
        weights_list = [w + n for w, n in zip(re.split(r'(\n)', weights_str)[::2], re.split(r'(\n)', weights_str)[1::2])]

        for regex, weight in zip(regex_list, weights_list):
            main_str = self._modify_string(main_str, regex, weight, insert_point_str, placeholder, pattern)

        return main_str

    '''
