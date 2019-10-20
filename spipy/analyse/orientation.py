import numpy as np
import os

def help(module):
	if module=="proc_Hammer":
		print("This function is used to calculate Hammer-Aitoff profection of data")
		print("    -> Input: qlist { <1> quaternions list : numpy.ndarray([w(:),qx(:),qy(:),qz(:)]), shape = (Nq,4) ;or")
		print("                      <2> xyz coordinates : numpy.ndarray([x(:),y(:),z(:)]), shape = (Nq,3) }")
		print("              data { intensity that want to be shown : numpy.ndarray, shape = (Nq,) }")
		print("    [Notice] w = cos(the/2) , q? = q? * sin(theta/2). BE CAREFUL of the order!")
		print("    [Notice] The order of data should matches bewteen qlist and data")
		print("    -> Return: [[logitute,latitute,value],[x,y,value]]")
		print("    [Notice] The returned list contains both (logitute,latitue) coordinate and (x,y) coordinate in Hammer-Aitoff projection map")
		return
	elif module=="draw_hammer":
		print("This function is used to plot and show Hammer-Aitoff projection in a easy way")
		print("    -> Input: logi_lati (data with coordinate, numpy.ndarray([logitute(:),latitue(:),data(:)]), shape = (Nd,3))")
		print("     *option: save_dir (dirname, save the figure to this folder, ABSOLUTE PATH !, default=None)")
		return
	elif module=="draw_ori_Df":
		print("This function is used to do draw probability distribution of orientations from Dragonfly output")
		print("    -> Input: ori_bin (path of Dragonfly orientation output 'orientations_xxx.bin')")
		print("              q_level (int, the 'num_div' parameter used in Dragonfly)")
		return
	elif module=="Sphere_randp":
		print("This function returns randomly/uniformly ditributed points on spherical surface")
		print("    -> Input: algo (str, algorithms to use : 'random' or 'uniform-1' or 'uniform-2')")
		print("              radius (positive float, radius of the spherical surface)")
		print("              num (positive int, how many points do you want)")
		print("    -> Return: list, contains cartisian description and radius description : ")
		print("               [numpy.ndarray shape=(Nd,3), numpy.ndarray shape=(Nd,2)]")
		print("                shape = (num,3)                , shape = (num,2)")
		print("[Notice] The returned theta is azimuth angle [0~2pi) and phi is zenith angle [-pi/2~pi/2]")
		return
	else:
		raise ValueError("No module names "+str(module))

def _vec2llxy(n1):
	# n1 = [x,y,z]
	# transfer 3d vector to logitute/latitute and x,y values hammer-aitoff coordinate
	if np.abs(np.linalg.norm(n1)-1)>1e-3:
			raise ValueError('Rotation matrix ???')
	latitute = np.arcsin(n1[2,0]/np.linalg.norm(n1,2))
	if n1[0,0]>=0:
		logitute = np.arcsin(n1[1,0]/(1e-10+np.linalg.norm(n1[:2,0],2)))
	elif n1[0,0]<0 and n1[1,0]>=0:
		logitute = np.pi - np.arcsin(n1[1,0]/(1e-10+np.linalg.norm(n1[:2,0],2)))
	elif n1[0,0]<0 and n1[1,0]<0:
		logitute = -np.pi - np.arcsin(n1[1,0]/(1e-10+np.linalg.norm(n1[:2,0],2)))
	else:
		raise ValueError("!")
	z_squr = 1 + np.cos(latitute) * np.cos(logitute/2.0)
	x = np.cos(latitute) * np.sin(logitute/2.0) / z_squr
	y = np.sin(latitute) / z_squr
	return [logitute,latitute,x,y]

def _xyz2ang(n, center):
	# n = [x,y,z] or [x,y]
	# center = (cx,cy,cz) or (cx,cy)
	# returned theta is azimuth angle [0~2pi) and phi is zenith angle [0~pi]
	n1 = np.array(n) - np.array(center)
	if len(n1) == 3:
		R = np.linalg.norm(n1)
		r = np.linalg.norm(n1[0:2])
		if R==0:
			return np.array([0,0])
		phi = np.arccos(n1[2]/R)
		if r==0:
			return np.array([0,phi])
		if n1[1]>=0:
			theta = np.arccos(n1[0]/r)
		elif n1[1]<0:
			theta = 2*np.pi-np.arccos(n1[0]/r)
		return np.array([theta,phi])
	elif len(n1) == 2:
		r = np.linalg.norm(n1)
		if r==0:
			return 0
		if n1[1]>=0:
			theta = np.arccos(n1[0]/r)
		elif n1[1]<0:
			theta = 2*np.pi-np.arccos(n1[0]/r)
		return theta
	else:
		raise ValueError("Input data format error")

# process data to fit hammer-aitoff projection
def proc_Hammer(qlist, data):
	if qlist.shape[1]<3 or qlist.shape[1]>4:
		raise ValueError("Check your input qlist please. Exit")
	import sys
	sys.path.append(__file__.split('/analyse/orientation.py')[0] + '/image/')
	from ..image import quat
	n0 = np.matrix([[1],[0],[0]])
	hammer = []
	n = []
	for ind,q in enumerate(qlist):
		if data[ind]==0:
			continue
		if len(q)==4:
			rot_m = quat.quat2rot(q)
			n1 = rot_m * n0
		else:
			n1 = q.reshape(3,1)
		logitute, latitute, x, y = _vec2llxy(np.array(n1))
		hammer.append([x,y,data[ind]])
		n.append([logitute,latitute,data[ind]])
	return [np.array(n),np.array(hammer)]

# show Hammer-Aitoff projection
def draw_hammer(logi_lati, save_dir=None):
	n = logi_lati
	import matplotlib.pyplot as plt
	plt.figure(figsize=(16,8))
	ax = plt.subplot(111, projection="hammer")
	logitute = n[:,0]
	latitute = n[:,1]
	P = n[:,2]
	T_ = P/(P.max()/20.0)
	gci = ax.scatter(logitute,latitute,c=P,s=np.pi*T_**2,alpha=0.5,marker='o',cmap='jet')
	plt.colorbar(gci)
	plt.grid(True)
	if save_dir:
		save_dir = os.path.dirname(save_dir)
		plt.savefig(save_dir+"/Hammer.png",dpi=300)
	plt.show()

# Probability distribution of orientations(quaternions) from Dragonfly
def draw_ori_Df(ori_bin, q_level):

	def gen_prob_dist(ori_file, q_level = 10):
		q_num = 10*(q_level+5*q_level**3)
		f = np.loadtxt(ori_file,delimiter=',')
		ori_prob = f[:,0].astype(float)
		q_ind = f[:,1].astype(int)
		q_prob = np.zeros(q_num)
		for i in range(len(f)):
			q = q_ind[i]
			prob = ori_prob[i]
			q_prob[q] += prob
		return q_prob

	prob = gen_prob_dist(ori_bin, q_level)
	# generate glist
	c_path = __file__.split('/analyse/orientation.py')[0] + '/image/qlist_dir'
	q_path = c_path + "/quat_"+str(q_level)+".txt"
	if not os.path.exists(q_path):
		cmd = c_path + "/gen_quat " + str(q_level)
		import subprocess
		subprocess.call(cmd, shell=True)
	qlist = np.loadtxt(q_path)

	ll, xy = proc_Hammer(qlist, prob)
	draw_hammer(ll)

# randomly get points on a spherical surface
def Sphere_randp(algo, radius, num):

	if algo == "uniform-1":
		# Spherical Fibonacci Lattice
		# num_points = num
		fab = (1.0+np.sqrt(5.0))/2.0
		i = np.linspace(-(num-1),(num-1),num,dtype=int)
		theta = 2.0*np.pi*i/fab
		sphi = i/float(num);
		cphi = np.sqrt((num+i)*(num-i))/num
		phi = np.arccos(sphi)

		x = radius*cphi*np.sin(theta)
		y = radius*cphi*np.cos(theta)
		z = radius*sphi

		return np.vstack([x,y,z]).T, np.vstack([theta,phi]).T

	if algo == "uniform-2":
		# Modified spherical Fibonacci Mapping
		# num_points = 4*n^2+2, which will be a slightly different from input "num"
		class Spherical(object):

			def __init__(self, radial = 1.0, polar = 0.0, azimuthal = 0.0):
				self.radial = radial
				self.polar = polar
				self.azimuthal = azimuthal
		 
			def toCartesian(self):
				r = np.sin(self.azimuthal) * self.radial
				x = np.cos(self.polar) * r
				y = np.sin(self.polar) * r
				z = np.cos(self.azimuthal) * self.radial
				return x, y, z

			def zpolar(self):
				return self.polar, self.azimuthal

		s = Spherical(radial=np.abs(radius))
		limit = np.abs(num)
		n = int(np.ceil(np.sqrt((limit - 2) / 4)))
		azimuthal = 0.5 * np.pi / n
		shell_xyz = np.array([0.0,0.0,0.0])
		shell_polar = np.array([0.0,0.0])
		for a in range(-n, n + 1):
			s.polar = 0
			size = (n - abs(a)) * 4 or 1
			polar = 2 * np.pi / size
			for i in range(size):
				shell_xyz = np.vstack((shell_xyz,s.toCartesian()))
				shell_polar = np.vstack((shell_polar,s.zpolar()))
				s.polar += polar
			s.azimuthal += azimuthal

		return shell_xyz[1:], shell_polar[1:]

	elif algo == "random":

		u = (np.random.random(num) - 0.5) * 2
		# [theta, phi] description
		phi = np.arccos(u)
		theta = np.random.random(num) * 2 * np.pi
		# [x,y,z] description
		x = radius * np.sqrt(1-u**2) * np.cos(theta)
		y = radius * np.sqrt(1-u**2) * np.sin(theta)
		z = radius * u

		return np.vstack((x,y,z)).T, np.vstack((theta,phi)).T
	
