import sys
import os
from parse2fit.io.readwrite import ReadWriteFactory

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_yaml_file>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.isfile(path):
        print(f"Error: The specified file '{path}' does not exist.")
        sys.exit(1)

    if not path.endswith(".yml") and not path.endswith(".yaml"):
        print("Error: The input file must be a .yml or .yaml file.")
        sys.exit(1)

    rw = ReadWriteFactory(path).get_writer()
    rw.write_input_files()

if __name__ == '__main__':
    main()

