import os
from pymatgen.core.structure import Structure
import numpy as np
from ase import Atoms

class StructuretoString:
    def __init__(self, structure, label, **kwargs):
        self.structure = structure
        self.label = label
        self.options = kwargs

    def to_bgf_string(self):

        def xtlgrf_writer():
            return 'XTLGRF 200\n'

        def descrp_writer(description):
            return 'DESCRP ' + description + '\n'

        def remark_writer(s):
            return 'REMARK ' + str(s.composition) + '\n'

        def rutype_writer(rutype):
            if rutype:
                return f'RUTYPE {rutype}\n'
            else:
                return ''

        def forcefield_dreiding_writer():
            return 'FORCEFIELD DREIDING\n'

        def format_atom_writer():
            return 'FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)\n'
        
        def crystx_writer(s):
            crystx = 'CRYSTX '
            a = f"{np.round(s.lattice.a, 5):.5f}"
            b = f"{np.round(s.lattice.b, 5):.5f}"
            c = f"{np.round(s.lattice.c, 5):.5f}"
            alpha = f"{np.round(s.lattice.alpha, 5):.5f}"
            beta = f"{np.round(s.lattice.beta, 5):.5f}"
            gamma = f"{np.round(s.lattice.gamma, 5):.5f}"
            for i, entry in enumerate([a, b, c, alpha, beta, gamma]):
                add_space = " " * (11 - len(entry))
                crystx += add_space + entry
            return crystx + '\n'

        def hetatoms_writer(s):
            hetatoms = ''
            xs = [f"{np.round(site.coords[0], 5):.5f}" for site in s]
            ys = [f"{np.round(site.coords[1], 5):.5f}" for site in s]
            zs = [f"{np.round(site.coords[2], 5):.5f}" for site in s]
            max_el = np.max([len(str(site.specie)) for site in s])

            for i, site in enumerate(s):
                hetatom = 'HETATM'
                # Atom number
                hetatom += " " * (7 - len(str(i+1))) + str(i+1)
                # Atom type
                hetatom += " " + str(site.specie) + " " * (max_el - len(str(site.specie)))
                # x-coordinate
                hetatom += " " * (11 - len(xs[i])) + xs[i]
                # y-coordinate
                hetatom += " " * (11 - len(ys[i])) + ys[i]
                # z-coordinate
                hetatom += " " * (11 - len(zs[i])) + zs[i]
                # Force field type (same as atom type for ReaxFF)
                hetatom += " " + str(site.specie) + " " * (max_el - len(str(site.specie)))
                # Two switches not used by ReaxFF and partial charge (not used)
                hetatom += " " * 4 + str(1) + " " + str(0) + " " + f"{0:.5f}"
                hetatoms += hetatom + '\n'
            hetatoms += 'END\n\n'
            return hetatoms
        
        def bgf_writer(s, description, rutype=None):
            bgf = ''
            bgf += xtlgrf_writer()
            bgf += descrp_writer(description)
            bgf += remark_writer(s)
            bgf += rutype_writer(rutype)
            bgf += forcefield_dreiding_writer()
            bgf += crystx_writer(s)
            bgf += format_atom_writer()
            bgf += hetatoms_writer(s)
            return bgf

        return bgf_writer(self.structure, self.label, self.options.get("rutype"))

