class UnitConverter:
    # Define conversion factors (example values): EDIT THESE TO BE CORRECT
    conversion_factors = {
        'energy': {
            'eV': 1.0,
            'eV/atom': 1.0,
            'kcal/mol': 23.0605478,
            'kcal': 23.0605478,
            'Rydberg': 0.0734986,
            'Rydberg/atom': 0.0734986,
            'Hartree': 0.0367493,
            'Hartree/atom': 0.0367493,
            'Ha': 0.0367493, 
            # Add more energy units here
        },
        'charge': {
            'e': 1.0,
            'C': 1.60218e-19,
            # Add more charge units here
        },
        'force': {
            'eV/Angstrom': 1.0,
            'Rydberg/Bohr': 0.038893784,
            'Hartree/Bohr': 0.019446892,
            'Ha/a0': 0.0367493, # For RMG
            '(kcal/mol)/Angstrom': 23.0605478,  # (kcal/mol)/Å to eV/Å
            # Add more force units here
        },
        'length': {
            'Angstrom': 1.0,
            'Bohr': 1.889726124565062,
        },
        'angle': {
            'degrees': 1.0,
            'radians': 0.0174533,
        },

        'pressure': {
            'bar': 1.0,
            'kbar': 0.001, 
        },
        # Add other properties and their units here
    }

    @staticmethod
    def convert(value, from_unit, to_unit, property_type):
        factors = UnitConverter.conversion_factors.get(property_type)
        if not factors:
            raise ValueError(f"Unknown property type: {property_type}")

        if from_unit not in factors or to_unit not in factors:
            raise ValueError(f"Invalid units for {property_type}: {from_unit} to {to_unit}")
        
        if type(value) == list:
            return [v * (factors[to_unit] / factors[from_unit]) for v in value]
        elif type(value) == float or type(value) == int:
            return value * (factors[to_unit] / factors[from_unit])
