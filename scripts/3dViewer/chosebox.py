from traits.api import *
from traitsui.api import *

class CB_Handler(Handler):

	def _OKbutton_fired(self, info):
		if not info.initialized:
			return
		info.ui.dispose()

	def _NObutton_fired(self, info):
		info.ui.dispose()


class chosebox(HasTraits):

	data = List(['1', '2', '3'])
	select = Str()
	OKbutton = Action(name="OK", action='_OKbutton_fired')
	NObutton = Action(name="Cancel", action='_NObutton_fired')
	returned_info = None

	view = View(
			Item('select', width=100,
				editor=EnumEditor(name='object.data'),
				tooltip='Choose data from input file',
				label='Choose Data'),
			title='Choose Box',
			resizable=False,
			kind='modal',
			statusbar=None,
			handler=CB_Handler(),
			buttons=[OKbutton, NObutton]
		)

	def _OKbutton_fired(self):
		self.returned_info = self.select
		print(self.returned_info)

	def set_choose(self, enumlist):
		self.data = enumlist
		self.returned_info = None


def show_chosebox(enumlist):
	mychosebox = chosebox()
	mychosebox.set_choose(enumlist)
	mychosebox.edit_traits()
	return mychosebox.select

	