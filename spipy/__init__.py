'''
from . import analyse
from . import image
from . import simulate
from . import phase
from . import merge
'''
from . import info

def help():
    import os
    print("spipy software includes modules : ")
    dir_name = os.path.dirname(__file__)
    for f in os.listdir(__file__.split('__init__.py')[0]):
        if (not os.path.isdir(os.path.join(dir_name, f))) or ('.' in f) or f.startswith("__"):
            continue
        else:
            print("    |- " + f)
            try:
                eval(f).help()
            except:
                pass
