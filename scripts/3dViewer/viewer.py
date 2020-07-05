import numpy as np
import scipy.io as sio
import h5py
import sys
import glob

from traits.api import *
from traitsui.api import *
from traitsui.file_dialog import open_file
from traitsui.message import message
from tvtk.pyface.scene_editor import SceneEditor 
from mayavi.tools.mlab_scene_model import MlabSceneModel
from mayavi.core.ui.mayavi_scene import MayaviScene
from mayavi import mlab
#import fix_mayavi_bugs
import popwin
import chosebox
import utils

#fix_mayavi_bugs.fix_mayavi_bugs()

class FieldViewer(HasTraits):

    # define data var
    s = None
    # define plot button
    plotbutton = Button("Import Selected File")
    rotatebutton = Button("Rorate and Movie")
    # define mayavi scene
    scene = Instance(MlabSceneModel, ()) 
    # define a file trait to view:
    file_name = File()
    openFile = Button('Open...')
    # init plot type => self._plotbutton_fired()
    plot_types = ['scalar', 'vector', 'points']
    plot_type = None
    # plot scence => self.plot() && self.*
    plot_scene_scalar = ['counter','cut plane','volume']
    plot_scene_vector = ['quiver','cut plane','Lorentz attractor trajectory']
    plot_scene_points = ['common', 'with lines', 'with intensity']
    plot_scene = List(['None','None','None'])
    select = Str()

    flag = Int

    G1 = VGroup(
                HGroup(
                    Item('openFile', show_label=False),
                    Item('file_name', style='readonly', width=150)
                ),
                Item('file_name', style='custom', label=''),
                Item('_'),
                #Item('file_name', style='readonly', label='File name'),
                Item('plotbutton'), show_labels=False
            )

    G2 = VGroup(
            HGroup(
                Item('scene', 
                    editor=SceneEditor(scene_class=MayaviScene), 
                    resizable=True,
                    height=600,
                    width=600,
                    label='Scene'),
                show_labels=False
                ), 
            HGroup(
                Group(
                    Item('select', 
                        editor=EnumEditor(name='object.plot_scene'),
                        tooltip='Display functions',
                        label='Plot')
                    ),
                Group(
                    Item('rotatebutton', show_label=False)
                    )
                )
            )
    
    view = View(
        HSplit(
            G1
            ,
            G2
        ),
        width = 900, resizable=True, title="3dView"
    )

    def _openFile_changed(self):
        filen = open_file()
        if filen != '':
            self.file_name = filen

    def _select_changed(self):
        self.plot()
      
    def _plotbutton_fired(self):
        try:
            attr = self.file_name.split('.')[-1]
            if attr=='npy':
                s = np.load(self.file_name)
            elif attr=='mat':
                temp = sio.loadmat(self.file_name)
                chosen = chosebox.show_chosebox(list(temp.keys()))
                if len(chosen) == 0:
                    return
                s = temp[chosen]
            elif attr=='h5':
                temp = h5py.File(self.file_name, 'r')
                all_dataset = utils.h5_parser(temp)
                chosen = chosebox.show_chosebox(all_dataset)
                if len(chosen) == 0:
                    return
                s = temp[chosen][()]
                temp.close()
            elif attr=='bin':
                s = np.fromfile(self.file_name)
                size = int(round(len(s)**(1.0/3.0)))
                s.shape = (size,size,size)
            else:
                raise ValueError("")
            s.astype(float)
            if len(s.shape)==3:
                self.plot_type = self.plot_types[0]
                self.plot_scene = self.plot_scene_scalar
            elif len(s.shape)==2 and s.shape[0]==6:
                self.plot_type = self.plot_types[1]
                self.plot_scene = self.plot_scene_vector
            elif len(s.shape)==2 and s.shape[0]==3:
                self.plot_type = self.plot_types[2]
                self.plot_scene = self.plot_scene_points[:2]
            elif len(s.shape)==2 and s.shape[0]==4:
                self.plot_type = self.plot_types[2]
                self.plot_scene = self.plot_scene_points
            else:
                np.zeros('Woop!')
            self.s = s
            self.select = self.plot_scene[0]
            self.plot()
        except Exception as e:
            message("I can't handle your file!\nError: %s" % e)
            pass

    def plot(self):
        s = self.s
        self.scene.mlab.gcf().scene.background = (0.6, 0.6, 0.6)
        if self.plot_type == self.plot_types[0]:
            if self.select == self.plot_scene[0]:
                self.plot_scalar_scene_1(s)
            elif self.select == self.plot_scene[1]:
                self.plot_scalar_scene_2(s)
            elif self.select == self.plot_scene[2]:
                self.plot_scalar_scene_3(s)
            else :
                pass
        elif self.plot_type == self.plot_types[1]:
            if self.select == self.plot_scene[0]:
                self.plot_vector_scene_1(s)
            elif self.select == self.plot_scene[1]:
                self.plot_vector_scene_2(s)
            elif self.select == self.plot_scene[2]:
                self.plot_vector_scene_3(s)
            else:
                pass
        elif self.plot_type == self.plot_types[2]:
            if self.select == self.plot_scene[0]:
                self.plot_points_scene_1(s)
            elif self.select == self.plot_scene[1]:
                self.plot_points_scene_2(s)
            elif self.select == self.plot_scene[2]:
                self.plot_points_scene_3(s)
            else:
                pass
        else:
            pass

    def plot_points_scene_1(self, s):
        self.scene.mlab.clf()
        g = self.scene.mlab.points3d(s[0], s[1], s[2], color=(1,0.94,0.98), scale_factor=0.9)
        self.g = g

    def plot_points_scene_2(self, s):
        self.scene.mlab.clf()
        g = self.scene.mlab.plot3d(s[0], s[1], s[2], color=(1,0.94,0.98), tube_radius=0.1)
        self.g = g

    def plot_points_scene_3(self, s):
        self.scene.mlab.clf()
        g = self.scene.mlab.points3d(s[0], s[1], s[2], s[3], colormap='rainbow', scale_factor=1)
        self.g = g

    def plot_scalar_scene_1(self, s):
        self.scene.mlab.clf()
        g = self.scene.mlab.contour3d(s, contours=10, transparent=True)
        g.actor.property.opacity = 0.4
        self.g = g

    def plot_scalar_scene_2(self, s):
        self.scene.mlab.clf()
        field = self.scene.mlab.pipeline.scalar_field(s)
        self.scene.mlab.pipeline.volume(field, vmin=s.min(), vmax=s.max())
        cut = self.scene.mlab.pipeline.scalar_cut_plane(field.children[0], plane_orientation="y_axes")
        cut.enable_contours = True
        cut.contour.number_of_contours = 20
        self.g = field
      
    def plot_scalar_scene_3(self, s):
        self.scene.mlab.clf()
        v = self.scene.mlab.pipeline.volume(self.scene.mlab.pipeline.scalar_field(s))
        self.g = v

    def plot_vector_scene_1(self, s):
        self.scene.mlab.clf()
        x,y,z,u,v,w = s[0],s[1],s[2],s[3],s[4],s[5]
        vectors = self.scene.mlab.quiver3d(x, y, z, u, v, w)
        vectors.glyph.mask_input_points = True
        vectors.glyph.mask_points.on_ratio = 10
        vectors.glyph.glyph.scale_factor = 5.0
        self.g = vectors

    def plot_vector_scene_2(self, s):
        self.scene.mlab.clf()
        x,y,z,u,v,w = s[0],s[1],s[2],s[3],s[4],s[5]
        src = self.scene.mlab.pipeline.vector_field(x, y, z, u, v, w)
        src_cut = self.scene.mlab.pipeline.vector_cut_plane(src, mask_points=2, scale_factor=5, plane_orientation='x_axes')
        src = self.scene.mlab.pipeline.vector_field(x, y, z, u, v, w)
        magnitude = self.scene.mlab.pipeline.extract_vector_norm(src)
        surface = self.scene.mlab.pipeline.iso_surface(magnitude, opacity=0.3)
        self.g = src_cut

    def plot_vector_scene_3(self, s):
        self.scene.mlab.clf()
        x,y,z,u,v,w = s[0],s[1],s[2],s[3],s[4],s[5]
        f = self.scene.mlab.flow(x, y, z, u, v, w)
        self.g = f

    def _rotatebutton_fired(self):
        try:
            self.s.shape
        except:
            message("Open a data file first!")
        self.f = mlab.gcf()
        self.f.scene.movie_maker.record = True
        self.dir_path = self.f.scene.movie_maker.directory
        popwin.init(self)

    def _flag_changed(self, old, new):
        if new == 1:
            self.make_movie()

    def make_movie(self):
        delay = 50

        @mlab.animate(delay=delay)
        def anim_movie(fignums):
            f = self.f
            f.scene.movie_maker.directory = popwin.get_dir()
            i = 0
            while i<fignums:
                f.scene.camera.azimuth(1)
                f.scene.render()
                i += 1
                popwin.set_now(i*delay/1000.0)
                yield
            popwin.set_now('finished')

            # render movie
            dirs = glob.glob(popwin.get_dir()+'/*')
            if len(dirs)<1:
                message("Do not change default save path !")
                return
            files = glob.glob(dirs[-1]+'/*.png')
            savename = dirs[-1].split('/')[-1]
            utils.processImage(files, popwin.get_dir()+'/'+savename+'.gif')

            self.flag = 0
            message("Movie making completed! Files are located on '"+popwin.get_dir()+"'")
            popwin.set_now(0)
            return

        time = popwin.get_total_time()
        fignums = int(time*1000/delay)
        a = anim_movie(fignums)

    def rotate(self):
        delay = 50

        @mlab.animate(delay=delay)
        def anim():
            f = mlab.gcf()
            f.scene.movie_maker.record = False
            while 1:
                f.scene.camera.azimuth(1)
                f.scene.render()
                yield

        anim()


if __name__ == '__main__':
    app = FieldViewer()
    app.configure_traits()
###1###