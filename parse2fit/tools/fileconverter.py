import os
from pymatgen.core.structure import Structure
from ase import Atoms
# Load openbabel, pybel
from openbabel import openbabel # Issues importing this before others
from openbabel import pybel

class StructuretoString:
    def __init__(self, structure, label, **kwargs):
        self.structure = structure
        self.label = label
        self.options = kwargs

    def to_bgf_string(self):
        temp_vasp = 'temp_poscar.vasp'
        temp_bgf = 'temp_bgf.bgf'

        if isinstance(self.structure, Structure):
            self.structure.to(fmt='poscar', filename=temp_vasp)
        elif isinstance(self.structure, Atoms):
            self.structure.write(filename=temp_vasp, format='vasp')

        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("VASP")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, temp_vasp)

        pybelmol = pybel.Molecule(mol)
        pybelmol.write("bgf", temp_bgf, overwrite=True)
        old_bgf, new_bgf = self._format_bgf(temp_bgf)
        
        os.remove(temp_vasp)
        os.remove(temp_bgf)
        return new_bgf

    def _format_bgf(self, path):
        new_string = ''
        old_string = ''

        with open(path) as bgf_file:
            for newline in bgf_file:
                old_string += newline
                if newline.strip(): # Ignore empty lines
                    if self.options.get("periodic") is True and self.options["periodic"] is False:
                        pass
                    else:
                        newline = newline.replace('BIOGRF', 'XTLGRF') # Indicate periodicity
                    if 'DESCRP' in newline: # Put title as a REMARK
                        new_string += f'DESCRP {self.label}\n'
                        newline = newline.replace('DESCRP', 'REMARK')

                        # Refer to https://www.engr.psu.edu/ADRI/Upload/reax_um.pdf for supported RUTYPE keys
                        if self.options.get('rutype') is not None and isinstance(self.options.get("rutype"), str):
                            newline += f'RUTYPE {self.options.get("rutype")}\n' # Add the string specified here

                    if 'CRYSTX' in newline:
                        start = newline.split(".")[0]
                        end = self._adjust_period_spacing(newline, 11) # Code specifications
                        newline = end.rstrip() + '\n'
                    if 'HETATM' in newline: # Format proper
                        items_split = newline.split()
                        el = self._clean_element(items_split[2])
                        newline = newline.replace(items_split[2], el)
                        for i in range(3, 6): # Region to get rid of
                            newline = newline.replace(' ' + items_split[i] + ' ', ' ' * (len(items_split[i])+2))
                    new_string += newline
        return old_string + '\n', new_string + '\n'

    def _clean_element(self, dirty_el):
        clean_el = ''
        count = 0
        for idx, i in enumerate(dirty_el):
            if not i.isdigit():
                if count == 0: # Only capitalize first letter
                    clean_el += i.upper()
                else:
                    clean_el += i.lower()
                count += 1
            else:
                clean_el += ' ' # Ensure correct spacing
        return clean_el

    def _adjust_period_spacing(self, input_string, spacing):
        result = []
        last_period_index = None

        i = 0
        while i < len(input_string):
            if input_string[i] == '.':
                if last_period_index is not None:
                    # Calculate the distance from the last period
                    distance = len(result) - last_period_index

                    if distance < spacing:
                        # If there are fewer than 'spacing' characters, add spaces to make it 'spacing' characters
                        while distance < spacing:
                            # Add a space before the current period
                            if result[last_period_index + 1] == ' ':
                                result.insert(last_period_index + 1, ' ')
                            else:
                                result.append(' ')
                            distance += 1
                    elif distance > spacing:
                        # If there are more than 'spacing' characters
                        num_extra = distance - spacing

                        # First remove spaces between periods
                        j = last_period_index + 1
                        while num_extra > 0 and j < len(result):
                            if result[j] == ' ':
                                result.pop(j)
                                num_extra -= 1
                            else:
                                j += 1

                        # If still more characters need to be removed, remove from the right of the last period
                        if num_extra > 0:
                            result = result[:last_period_index + spacing]

                last_period_index = len(result)
            result.append(input_string[i])
            i += 1

        # Handle the case where the string ends with a period and needs to be truncated or padded
        if last_period_index is not None:
            distance_to_end = len(result) - last_period_index
            if distance_to_end < spacing:
                result.append(' ' * (spacing - distance_to_end))
            elif distance_to_end > spacing:
                result = result[:last_period_index + spacing]
        return ''.join(result)
