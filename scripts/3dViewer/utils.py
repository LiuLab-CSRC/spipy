import numpy as np
import sys
import h5py

def processImage(infiles, savepath):
	from PIL import Image
	if len(infiles)<2:
		return "Not enough figures."
	im = Image.open(infiles[0])
	images = []
	for f in infiles[1:]:
		images.append(Image.open(f))
	im.save(savepath, save_all=True, append_images=images, loop=1, duration=1)

def generateAVI(infiles, savepath):
	try:
		import cv2
	except:
		from traitsui.message import message
		message("I got some problems on loading opencv, try 'conda install opencv'.")
		return

	frame = cv2.imread(infiles[0])
	height, width, layers = frame.shape
	video = cv2.VideoWriter(savepath, -1, 1, (width,height))
	for f in infiles:
		im = cv2.imread(f)
		video.write(im)

	video.release()
	cv2.destroyAllWindows()

def h5_parser(fp):
	keys = []
	def _get_all_dataset(key):
		if isinstance(fp[key], h5py._hl.dataset.Dataset):
			keys.append(key)
	fp.visit(_get_all_dataset)
	return keys


if __name__=="__main__":
	import glob
	infiles = glob.glob('/Users/shiyingchen/Documents/mayavi_movies/movie001/*.png')
	savepath = "/Users/shiyingchen/Documents/mayavi_movies/movie001.avi"
	generateAVI(infiles, savepath)