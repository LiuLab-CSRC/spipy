import sys
import os
import argparse
import subprocess

if __name__ == '__main__':

    modes = ["2dviewer","3dviewer","maskmaker"]

    # parse cmd arguments
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description = "Dataset visualization.")
    parser.add_argument("-m", "--mode", type=str, choices=modes, help="Choose which kind of visualization.", required=True)
    args = parser.parse_args()

    # show
    this_folder = os.path.dirname(__file__)
    if args.mode == modes[0]:
        # 2dviewer
        codepath = os.path.join(this_folder,"2dViewer","viewer.py")
    elif args.mode == modes[1]:
        # 3dviewer
        codepath = os.path.join(this_folder,"3dViewer","viewer.py")
    elif args.mode == modes[2]:
        # maskmaker
        codepath = os.path.join(this_folder,"2dViewer","maskMaker.py")
    else:
        print("Unrecognized mode.")
        sys.exit(1)

    subprocess.check_call("python %s" % codepath, shell=True)