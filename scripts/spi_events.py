import sys
import os
import argparse
import yaml
import h5py
import re
import glob
from spipy.image import io as spio

if __name__ == '__main__':

    filetype = ["none", "cxi", "hdf5", "tif"]

    # parse cmd arguments
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description = "Parse CXI/HDF5/TIF... files and list events.")
    parser.add_argument("-f", "--files", type=str, help="The data files to parse events from, use '%%' as wildcard. Eg, './dataset/%%.h5'.", required=True)
    parser.add_argument("-o", "--output", type=str, help="The output events file.", required=True)
    parser.add_argument("-r", "--reg", type=str, default="none", help="Data location inside HDF5 file, use '%%' as wildcard. Eg, 'entry_1/%%/data'. Default=none.")
    parser.add_argument("-y", "--filetype", type=str, choices=filetype, default=filetype[0], help="Force input data file type, default=none and the data file type will be determined by its extension.")
    args = parser.parse_args()

    # get parameters
    CXIFILE = spio._CXIDB()
    if args.reg.upper() != "NONE":
        reg = re.sub(r"^/*", "", args.reg)
        reg = re.sub(r"%", r"\\S*", reg)
        reg = r"%s" % reg
    else:
        reg = None

    # get files
    tmp = re.sub(r"%", r"*", args.files)
    files = glob.glob(os.path.abspath(tmp))

    # parse
    for afile in files:
        # file type
        if args.filetype == filetype[0]:
            file_ext = os.path.splitext(afile)[-1].lower()
            if file_ext in [".h5", ".cxi"]:
                this_filetype = filetype[1]
            elif file_ext == ".tif":
                this_filetype = filetype[3]
            else:
                raise RuntimeError("Input file format is unknown (%s)." % afile)
        else:
            this_filetype = args.filetype
        # parse
        if this_filetype in filetype[1:3]:
            events = CXIFILE.list_events(os.path.abspath(afile), reg, 2)
        elif this_filetype == filetype[3]:
            events = {os.path.abspath(afile):{"default":[1, None]}}
        else:
            raise ValueError("Unknown file type %s." % args.filetype)

        with open(args.output, "a") as fp:
            serial_evt = yaml.dump(events)
            fp.write(serial_evt)
