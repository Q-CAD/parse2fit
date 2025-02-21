from parse2fit.io.readwrite import ReadWriteFactory
import sys

path = sys.argv[1]

rw = ReadWriteFactory(path).get_writer() 
rw.write_input_files()


