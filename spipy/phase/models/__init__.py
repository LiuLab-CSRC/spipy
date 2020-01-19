from .phInput import phInput
from .ERA import ERA
from .DM import DM
from .RAAR import RAAR
from .HIO import HIO
from .phOutput import phOutput

def get_model_from_classname(class_name):
	if class_name == phInput.__name__:
		return phInput
	elif class_name == phOutput.__name__:
		return phOutput
	elif class_name == ERA.__name__:
		return ERA
	elif class_name == DM.__name__:
		return DM
	elif class_name == RAAR.__name__:
		return RAAR
	elif class_name == HIO.__name__:
		return HIO
	else:
		raise ValueError("class name error !")