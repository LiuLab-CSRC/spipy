import numpy as np
from traits.api import *
from traitsui.api import *

#import fix_mayavi_bugs

#fix_mayavi_bugs.fix_mayavi_bugs()

"""
class MyHandler(Handler):
	def _process_start(self, info):
		if info.object.flag == False:
			info.ui.dispose()
"""

class movie_settings(HasTraits):
	father = None
	# define flag to control gui
	flag = True
	now = CFloat
	# define input text
	total_time = Int
	# define a saving path
	dir_name = Directory()
	# define current percent
	curr = Property(depends_on='now')
	# define button
	start = Button('Make movie')
	rotate = Button('Rotate')
	# define view
	view = View(
		Group(
			Group(
				Item('total_time', label='Movie length (second):'),
				Item('dir_name', style='simple', label='File saving to')
			),
			HGroup(
				Group(
					Item('rotate', show_label=False)
				),
				Group(
					Item('start', show_label=False)
				)
			),
			orientation = 'vertical',
			show_border = True
		),
		title='Movie Maker',
		kind='live',
		resizable=True,
        statusbar=[StatusItem(name='curr')]
		)

	def set_now(self, nowtime):
		self.now = nowtime

	def quit(self):
		info = Instance(UIInfo)
		info.ui.dispose()

	# events
	def _get_curr(self):
		pre = 100*self.now/(self.total_time+1e-5)
		return ('Progress: %d %% recorded.' % int(pre))

	def _start_fired(self):
		if self.total_time <= 0:
			message('Movie length should not <= 0 !')
		else:
			self.father.flag = 1

	def _rotate_fired(self):
		self.father.rotate()

ctrl = None

def init(fatherr):
	global ctrl
	ctrl = movie_settings()
	ctrl.father = fatherr
	ctrl.edit_traits()
	ctrl.dir_name = fatherr.f.scene.movie_maker.directory

def set_now(time):
	global ctrl
	if time=='finished':
		ctrl.set_now(float(ctrl.total_time))
	else:
		ctrl.set_now(time)

def get_total_time():
	global ctrl
	return ctrl.total_time

def quit():
	global ctrl
	ctrl.quit()

def get_dir():
	return ctrl.dir_name

if __name__ == '__main__':
	init(None)
	ctrl.close()
