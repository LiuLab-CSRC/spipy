import sys
import os
import argparse
import yaml
import h5py
import re
from spipy.image import io as spio

if __name__ == '__main__':

    filetype = ["cxi", "hdf5", "tif"]

    # parse cmd arguments
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description = "Parse CXI/HDF5/TIF... files and list events.")
    parser.add_argument("-f", "--file", type=str, help="The data file to parse events from.", required=True)
    parser.add_argument("-o", "--output", type=str, help="The output events file.", required=True)
    parser.add_argument("-r", "--reg", type=str, default="", help="Data location inside HDF5 file, use '%%' as wildcard. Eg, 'entry_1/%%/data'. Default=none.")
    parser.add_argument("-y", "--filetype", type=str, choices=filetype, default=filetype[0], help="Input data file type, default=cxi")
    args = parser.parse_args()

    # get parameters
    CXIFILE = spio._CXIDB()
    if args.reg.upper() != "NONE":
        reg = re.sub(r"^/*", "", args.reg)
        reg = re.sub(r"%", r"\\S*", reg)
        reg = r"%s" % reg
    else:
        reg = None

    # parse
    if args.filetype in filetype[0:2]:
        events = CXIFILE.list_events(os.path.abspath(args.file), reg, 2)
    elif args.filetype == filetype[2]:
        events = {os.path.abspath(args.file):{"default":1}}
    else:
        raise ValueError("Unknown file type %s." % args.filetype)

    with open(args.output, "a") as fp:
        serial_evt = yaml.dump(events)
        fp.write(serial_evt)
