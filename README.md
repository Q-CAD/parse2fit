# `parse2fit`

![alt text](Ensemble-FF-Fit.jpg?raw=true)

`parse2fit` parses DFT calculations into input files for force field training. It currently supports VASP and RMG DFT codes as inputs and the Reax reactive-MD and FitSNAP machine-learned force fields as outputs.

Future updates will add support for more force fields. 

## Installation
You can install `parse2fit` using pip:

```bash
pip install git+https://code.ornl.gov/rym/parse2fit.git
```

Alternatively, if you're developing the package, clone the repository and install it in editable mode. 

```bash
git clone https://code.ornl.gov/rym/parse2fit.git
cd parse2fit
pip install -e .
```

# Code Design

`parse2fit` employs `Parser()` classes (e.g., `VaspParser` or `RMGParser`) to convert raw data into the `Energy()` or `PropertyCollection()` objects,  which are collections of `Property()` classes (e.g., `Charge()`, `Force()`, `Geometry()`, `LatticeVector()`, `StressVector()`). 

`Entries()` (e.g., `ReaxEntry()` or `FitSNAPEntry()`) are built from multiple `Property()` classes, and contain logic to transform raw data into input files, including unit conversions and relative energy calculations for Reax.   

`ParserFactory()` automatically constructs objects based on detected files (e.g., `vasprun.xml` or `forcefield.xml`) and the correct `Parser()` based on files found in the directory passed. 

The `RW()` classes (e.g., `ReaxEntry()` or `FitSNAPEntry()`) use `.yml` instructions to generate the appropriate `Property()` and `Entry()` objects and write the corresponding input files for force field training (see `examples` directory). Below is the `rmg` example for `ReaxRW()`. 

# Usage
The following code generates input files as specified by a configuration `.yml` file.

    config_path = '/path/to/your/configuration.yml'
    rw = ReadWriteFactory(config_path=config_path).get_writer()
    rw.write_trainsetins()

Descriptions of supported `.yml` configurations are provided below. 

## Reax

Reax reactive force field fitting minimizes an objective function built from structures included in a `geo` file and training data included in a `trainset.in` file. The Reax `.yml` format supports data sampling and weighting for flexibility.

## FitSNAP

FitSNAP machine-learned force field fitting minimizes an objective function built from structures included in a `.xyz` or `.json` files and training parameters included in a `fitsnap.in` file. The FitSNAP `.yml` format supports data sampling and weighting for flexibility.

### 1. `output_format`:

(str) Specifies the `.yml` format. `reaxff` and `fitsnap` are supported.

### 2. `generation_parameters`:
    
`runs_to_generate`: (int): Number of unique input combinations to generate.

`output_directory`: (str): Path to write new Reax input files. 

### 3. `default_weights`:

Allows for automated weighting of data points. Supported schemes include:

1. Binary: Assigns `min` or `max` weights randomly, with probabilities defined by `split`.

2. Magnitude: Weights are exponential values within the range `[min, max]`, and are scaled by `kT` and `scale`.

3. Normal: Distributes weights normally between `min` and `max`, with spread controlled by `sigma` and `scale`.

### 4. `input_paths`:

Defines the data to be sampled and its weighting scheme. The top level is a named group label for included directories:

Directories: Specify paths to DFT run directories or the directory trees where paths should be searched for.

Data types: Select properties (e.g., energy, forces, charges) to parse or include (e.g., structure).

Weights: Optionally override default_weights for a given property or label.

Energy references: Specify reference states for adding and subtracting energies with ReaxFF (e.g., for formation or defect energies).

# ReaxFF Layout

```bash
output_format: 'reaxff' # consistent tag across all .yaml

generation_parameters:
  runs_to_generate: 1
  output_directory: 'reaxff_rmg'

default_weights: # Default is 50/50 0 or 1
  energy: 1

input_paths:
  "Elemental_Bi_Reference":
    directories:
      - rmg/Bi
    energy: True

  "Elemental_Se_Reference":
    directories:
      - rmg/Se
    energy: True

  "Bi2Se3_Formation_Energy":
    directories:
      - rmg/Bi2Se3
    structure:
      rutype: "NORMAL RUN"
    energy:
      weights: 5
      subtract:
        - rmg/Bi
        - rmg/Se
      get_divisors: True
    forces: True
    distances: True
    angles: True
    dihedrals: True
    lattice_vectors: True
    charges: True
```

Energies are written to `trainset.in` without duplicates, and shorter or named directory paths are prioritized.



