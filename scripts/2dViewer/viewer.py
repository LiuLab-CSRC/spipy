#! /opt/local/bin/python2.7

"""
Usage:
    image_viewer.py

Options:
    -h --help       Show this screen.
"""

import sys
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
from pyqtgraph import PlotDataItem, PlotItem
from scipy.signal import savgol_filter, argrelmax, argrelmin
from util import *
import numpy as np
import glob

from layout import Ui_MainWindow
from util import DatasetTreeWidget, MyImageView, MyParameterTree, MyPlotWidget

data_viewer_window = None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        _dir = os.path.dirname(os.path.abspath(__file__))
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.profileWidget.hide()
        self.ui.smallDataWidget.hide()
        self.ui.splitter_3.setSizes([self.width()*0.7, self.width()*0.3])   # splitter between image_view/plot_widget and file list
        self.ui.splitter_2.setSizes([self.height() * 0.7, self.height() * 0.3])  # splitter between image_view and plot widget
        self.ui.splitter.setSizes([self.ui.splitter.width()/3., self.ui.splitter.width()/3., self.ui.splitter.width()/3.])
        self.setAcceptDrops(True)
        self.ui.fileList.setColumnWidth(0, 200)

        self.filepath = None  # current filepath
        self.imageData = None  # original image data, 2d or 3d
        self.imageShape = None
        self.dispData = None  # 2d data for plot
        self.dispShape = None
        self.mask = None
        self.curve = None
        self.h5obj = None
        self.acceptedFileTypes = [u'npy', u'npz', u'h5', u'mat', u'cxi', u'tif']

        self.dispItem = self.ui.imageView.getImageItem()
        self.ringItem = pg.ScatterPlotItem()
        self.centerMarkItem = pg.ScatterPlotItem()
        self.ui.imageView.getView().addItem(self.ringItem)
        self.ui.imageView.getView().addItem(self.centerMarkItem)

        self.curveItem = PlotDataItem()
        self.ui.curveWidget.addItem(self.curveItem)
        self.ui.curveWidget.getPlotItem().setTitle(title='Curve Plot')

        # basic operation for image and curve plot
        self.showImage = True
        self.axis = 'x'
        self.frameIndex = 0
        self.maskFlag = False
        self.imageLog = False
        self.binaryFlag = False
        self.FFT = False
        self.FFTSHIFT = False
        self.dispThreshold = 0
        self.center = [0, 0]
        self.showRings = False
        self.ringRadiis = []
        self.showCurvePlot = True

        self.profileItem = PlotDataItem(pen=pg.mkPen('y', width=1, style=QtCore.Qt.SolidLine), name='profile')
        self.smoothItem = PlotDataItem(pen=pg.mkPen('g', width=2, style=QtCore.Qt.DotLine), name='smoothed profile')
        self.thresholdItem = PlotDataItem(pen=pg.mkPen('r'), width=1, style=QtCore.Qt.DashLine, name='threshold')
        self.ui.profileWidget.addLegend()
        self.ui.profileWidget.addItem(self.profileItem)
        self.ui.profileWidget.addItem(self.smoothItem)
        self.ui.profileWidget.addItem(self.thresholdItem)
        self.ui.profileWidget.getPlotItem().setTitle(title='Profile Plot')
        # profile option
        self.showProfile = False
        self.profileType = 'radial'
        self.profileMode = 'sum'
        # extrema search
        self.extremaSearch = False
        self.extremaType = 'max'
        self.extremaWinSize = 11
        self.extremaThreshold = 1.0  # ratio compared to mean value
        # angular option
        self.angularRmin = 0.
        self.angularRmax = np.inf
        # across center line option
        self.lineAngle = 0.
        self.lineWidth = 1
        # profile smoothing
        self.smoothFlag = False
        self.smoothWinSize = 15
        self.polyOrder = 3
        # small data option
        self.smallDataItem = pg.ScatterPlotItem()
        self.ui.smallDataWidget.addItem(self.smallDataItem)
        self.ui.smallDataWidget.getPlotItem().setTitle(title='Small Data')
        self.showSmallData = False
        self.smallDataFile = None
        self.smallDataset = None
        self.smallDataSorted = False
        self.smallDataPaths = None
        self.smallDataFrames = None
        self.smallData = None
        # display option
        self.imageAutoRange = False
        self.imageAutoLevels = False
        self.imageAutoHistogramRange = True
        self.curvePlotAutoRange = True
        self.curvePlotLog = False
        self.profilePlotAutoRange = True
        self.profilePlotLog = False
        self.smallDataPlotAutoRange = True
        self.smallDataPlotLog = False

        params_list = [
                        {'name': 'Data Info', 'type': 'group', 'children': [
                            {'name': 'File', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Dataset', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Mask', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Image Shape', 'type': 'str', 'value': 'unknown', 'readonly': True},
                            {'name': 'Image Friedel Score', 'type': 'float', 'readonly': True},
                            {'name': 'Curve Length', 'type': 'int', 'readonly': 'True'}
                        ]},
                        {'name': 'Basic Operation', 'type': 'group', 'children': [
                            {'name': 'Image', 'type': 'group', 'children': [
                                {'name': 'Axis', 'type': 'list', 'values': ['x','y','z'], 'value': self.axis},
                                {'name': 'Frame Index', 'type': 'int', 'value': self.frameIndex},
                                {'name': 'Apply Mask', 'type': 'bool', 'value': self.imageLog},
                                {'name': 'Apply Log', 'type': 'bool', 'value': self.maskFlag},
                                {'name': 'Apply FFT', 'type': 'bool', 'value': self.FFT},
                                {'name': 'Apply FFT-SHIFT', 'type': 'bool', 'value': self.FFTSHIFT},
                                {'name': 'Binaryzation', 'type': 'bool', 'value': self.binaryFlag},
                                {'name': 'Threshold', 'type': 'float', 'value': self.dispThreshold},
                                {'name': 'Center x', 'type': 'int', 'value': self.center[1]},
                                {'name': 'Center y', 'type': 'int', 'value': self.center[0]},
                                {'name': 'Show Rings', 'type': 'bool'},
                                {'name': 'Ring Radiis', 'type': 'str', 'value': ''},
                            ]},
                            {'name': 'Curve Plot', 'type': 'group', 'children': [
                                {'name': 'TODO', 'type': 'str'},
                            ]}
                        ]},
                        {'name': 'Image to Feature Profile', 'type': 'group', 'children': [
                            {'name': 'Show Profile', 'type': 'bool', 'value': self.showProfile},
                            {'name': 'Feature', 'type': 'list', 'values': ['radial','angular', 'across center line'], 'value': self.profileType},
                            {'name': 'Profile Mode', 'type': 'list', 'values': ['sum','mean'], 'value': self.profileMode},
                            {'name': 'Angular Option', 'type': 'group', 'children': [
                                {'name': 'R min', 'type': 'float', 'value': self.angularRmin},
                                {'name': 'R max', 'type': 'float', 'value': self.angularRmax},
                            ]},
                            {'name': 'Across Center Line Option', 'type': 'group', 'children': [
                                {'name': 'Angle', 'type': 'float', 'value': self.lineAngle},
                                {'name': 'Width/pixel', 'type': 'int', 'value': self.lineWidth},
                            ]},
                            {'name': 'Smoothing', 'type': 'group', 'children': [
                                {'name': 'Enable Smoothing', 'type': 'bool', 'value': self.smoothFlag},
                                {'name': 'Window Size', 'type': 'int', 'value': self.smoothWinSize},
                                {'name': 'Poly-Order', 'type': 'int', 'value': self.polyOrder},
                            ]},
                            {'name': 'Extrema Search', 'type': 'group', 'children': [
                                {'name': 'Enable Extrema Search', 'type': 'bool', 'value': self.extremaSearch},
                                {'name': 'Extrema Type', 'type': 'list', 'values': ['max', 'min'], 'value': self.extremaType},
                                {'name': 'Extrema WinSize', 'type': 'int', 'value': self.extremaWinSize},
                                {'name': 'Extrema Threshold', 'type': 'float', 'value': self.extremaThreshold},
                            ]}
                        ]},
                        {'name': 'Small Data', 'type': 'group', 'children': [
                            {'name': 'Filepath', 'type': 'str'},
                            {'name': 'Dataset', 'type': 'str'},
                            {'name': 'Show data', 'type': 'bool', 'value': self.showSmallData},
                            {'name': 'Sort', 'type': 'bool', 'value': self.smallDataSorted},
                        ]},
                        {'name': 'Display Option', 'type': 'group', 'children': [
                            {'name': 'Image', 'type': 'group', 'children': [
                                {'name': 'autoRange', 'type': 'bool', 'value': self.imageAutoRange},
                                {'name': 'autoLevels', 'type': 'bool', 'value': self.imageAutoLevels},
                                {'name': 'autoHistogramRange', 'type': 'bool', 'value': self.imageAutoHistogramRange},
                            ]},
                            {'name': 'Curve Plot', 'type': 'group',  'children': [
                                {'name': 'autoRange', 'type': 'bool', 'value': self.curvePlotAutoRange},
                                {'name': 'Log', 'type': 'bool', 'value': self.curvePlotLog},
                            ]},
                            {'name': 'Profile Plot', 'type': 'group', 'children': [
                                {'name': 'autoRange', 'type': 'bool', 'value': self.profilePlotAutoRange},
                                {'name': 'Log', 'type': 'bool', 'value': self.profilePlotLog},
                            ]},
                            {'name': 'Small Data Plot', 'type': 'group', 'children': [
                                {'name': 'autoRange', 'type': 'bool', 'value': self.smallDataPlotAutoRange},
                                {'name': 'Log', 'type': 'bool', 'value': self.smallDataPlotLog},
                            ]},
                        ]}
                      ]
        self.params = Parameter.create(name='params', type='group', children=params_list)
        self.ui.parameterTree.setParameters(self.params, showTop=False)

        self.ui.fileList.itemDoubleClicked.connect(self.changeDatasetSlot)
        self.ui.fileList.customContextMenuRequested.connect(self.showFileMenuSlot)
        self.ui.imageView.scene.sigMouseMoved.connect(self.mouseMoved)
        self.smallDataItem.sigClicked.connect(self.smallDataClicked)
        self.ui.lineEdit.returnPressed.connect(self.addFilesSlot)

        self.params.param('Basic Operation', 'Image', 'Axis').sigValueChanged.connect(self.axisChangedSlot)
        self.params.param('Basic Operation', 'Image', 'Frame Index').sigValueChanged.connect(self.frameIndexChangedSlot)
        self.params.param('Basic Operation', 'Image', 'Apply Mask').sigValueChanged.connect(self.applyMaskSlot)
        self.params.param('Basic Operation', 'Image', 'Apply Log').sigValueChanged.connect(self.applyImageLogSlot)
        self.params.param('Basic Operation', 'Image', 'Binaryzation').sigValueChanged.connect(self.binaryImageSlot)
        self.params.param('Basic Operation', 'Image', 'Apply FFT').sigValueChanged.connect(self.applyFFTSlot)
        self.params.param('Basic Operation', 'Image', 'Apply FFT-SHIFT').sigValueChanged.connect(self.applyFFTSHIFTSlot)
        self.params.param('Basic Operation', 'Image', 'Binaryzation').sigValueChanged.connect(self.binaryImageSlot)
        self.params.param('Basic Operation', 'Image', 'Threshold').sigValueChanged.connect(self.setDispThresholdSlot)
        self.params.param('Basic Operation', 'Image', 'Center x').sigValueChanged.connect(self.centerXChangedSlot)
        self.params.param('Basic Operation', 'Image', 'Center y').sigValueChanged.connect(self.centerYChangedSlot)
        self.params.param('Basic Operation', 'Image', 'Show Rings').sigValueChanged.connect(self.showRingsSlot)
        self.params.param('Basic Operation', 'Image', 'Ring Radiis').sigValueChanged.connect(self.ringRadiiSlot)

        self.params.param('Image to Feature Profile', 'Show Profile').sigValueChanged.connect(self.showProfileSlot)
        self.params.param('Image to Feature Profile', 'Feature').sigValueChanged.connect(self.setProfileTypeSlot)
        self.params.param('Image to Feature Profile', 'Profile Mode').sigValueChanged.connect(self.setProfileModeSlot)
        self.params.param('Image to Feature Profile', 'Angular Option', 'R min').sigValueChanged.connect(self.setAngularRminSlot)
        self.params.param('Image to Feature Profile', 'Angular Option', 'R max').sigValueChanged.connect(self.setAngularRmaxSlot)
        self.params.param('Image to Feature Profile', 'Across Center Line Option', 'Angle').sigValueChanged.connect(self.setLineAngleSlot)
        self.params.param('Image to Feature Profile', 'Across Center Line Option', 'Width/pixel').sigValueChanged.connect(self.setLineWidthSlot)
        self.params.param('Image to Feature Profile', 'Smoothing', 'Enable Smoothing').sigValueChanged.connect(self.setSmoothSlot)
        self.params.param('Image to Feature Profile', 'Smoothing', 'Window Size').sigValueChanged.connect(self.setWinSizeSlot)
        self.params.param('Image to Feature Profile', 'Smoothing', 'Poly-Order').sigValueChanged.connect(self.setPolyOrderSlot)
        self.params.param('Image to Feature Profile', 'Extrema Search', 'Enable Extrema Search').sigValueChanged.connect(self.setExtremaSearchSlot)
        self.params.param('Image to Feature Profile', 'Extrema Search', 'Extrema Type').sigValueChanged.connect(self.setExtremaTypeSlot)
        self.params.param('Image to Feature Profile', 'Extrema Search', 'Extrema WinSize').sigValueChanged.connect(self.setExtremaWinSizeSlot)
        self.params.param('Image to Feature Profile', 'Extrema Search', 'Extrema Threshold').sigValueChanged.connect(self.setExtremaThresholdSlot)

        self.params.param('Small Data', 'Filepath').sigValueChanged.connect(self.setSmallDataFilepathSlot)
        self.params.param('Small Data', 'Dataset').sigValueChanged.connect(self.setSmallDatasetSlot)
        self.params.param('Small Data', 'Show data').sigValueChanged.connect(self.showSmallDataSlot)
        self.params.param('Small Data', 'Sort').sigValueChanged.connect(self.sortSmallDataSlot)

        self.params.param('Display Option', 'Image', 'autoRange').sigValueChanged.connect(self.imageAutoRangeSlot)
        self.params.param('Display Option', 'Image', 'autoLevels').sigValueChanged.connect(self.imageAutoLevelsSlot)
        self.params.param('Display Option', 'Image', 'autoHistogramRange').sigValueChanged.connect(self.imageAutoHistogramRangeSlot)
        self.params.param('Display Option', 'Curve Plot', 'autoRange').sigValueChanged.connect(self.curvePlotAutoRangeSlot)
        self.params.param('Display Option', 'Curve Plot', 'Log').sigValueChanged.connect(self.curvePlotLogModeSlot)
        self.params.param('Display Option', 'Profile Plot', 'autoRange').sigValueChanged.connect(self.profilePlotAutoRangeSlot)
        self.params.param('Display Option', 'Profile Plot', 'Log').sigValueChanged.connect(self.profilePlotLogModeSlot)
        self.params.param('Display Option', 'Small Data Plot', 'autoRange').sigValueChanged.connect(self.smallDataPlotAutoRangeSlot)
        self.params.param('Display Option', 'Small Data Plot', 'Log').sigValueChanged.connect(self.smallDataPlotLogModeSlot)

    def addFilesSlot(self):
        filePattern = str(self.ui.lineEdit.text())
        #print_with_timestamp('file(s) pattern: %s' %filePattern)
        files = glob.glob(filePattern)
        #print_with_timestamp('adding file(s): %s \n Total Num: %d' %(str(files), len(files)))
        for i in range(len(files)):
            self.maybeAddFile(files[i])

    def smallDataClicked(self, _, points):
        _temp_file = '.temp.npy'
        pos = points[0].pos()
        index = int(points[0].pos()[0])
        if self.smallDataSorted:
            index = np.argsort(self.smallData)[index]
        filepath = self.smallDataPaths[index]
        frame = self.smallDataFrames[index]
        #print_with_timestamp('showing file: %s, frame: %d' %(filepath, frame))
        make_temp_file(filepath, frame, _temp_file)
        maybeExistIndex = self.ui.fileList.indexOf(_temp_file)
        if maybeExistIndex != -1:
            self.ui.fileList.takeTopLevelItem(maybeExistIndex)
        item = FileItem(filepath=_temp_file)
        self.ui.fileList.insertTopLevelItem(0, item)
        self.changeDatasetSlot(item, 0)

    def setSmallDataFilepathSlot(self, _, filepath):
        #print_with_timestamp('set filepath for small data: %s' % str(filepath))
        self.smallDataFile = filepath
        self.maybeShowSmallData()

    def setSmallDatasetSlot(self, _, dataset):
        #print_with_timestamp('set dataset for small data: %s' % str(dataset))
        self.smallDataset = dataset
        self.maybeShowSmallData()

    def showSmallDataSlot(self, _, showSmallData):
        #print_with_timestamp('set show small data: %s' % str(showSmallData))
        # self.showSmallData = showSmallData
        if showSmallData:
            self.ui.smallDataWidget.show()
        else:
            self.ui.smallDataWidget.hide()
        self.maybeShowSmallData()

    def sortSmallDataSlot(self, _, sort):
        #print_with_timestamp('set show small data sorted: %s' % str(sort))
        self.smallDataSorted = sort
        self.maybeShowSmallData()

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            if url.toLocalFile().startswith('/.file/id='):
                dropFile = getFilepathFromLocalFileID(url)
            else:
                dropFile = url.toLocalFile()
            fileInfo = QtCore.QFileInfo(dropFile)
            ext = fileInfo.suffix()
            if ext in self.acceptedFileTypes:
                event.accept()
                return None
        event.ignore()
        return None

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            if url.toLocalFile().startswith('/.file/id='):
                dropFile = getFilepathFromLocalFileID(url)
            else:
                dropFile = url.toLocalFile()
            self.maybeAddFile(dropFile)

    def maybeAddFile(self, filepath):
        ext = QtCore.QFileInfo(filepath).suffix()
        if ext in self.acceptedFileTypes:
            maybeExistIndex = self.ui.fileList.indexOf(filepath)
            if maybeExistIndex != -1:
                self.ui.fileList.takeTopLevelItem(maybeExistIndex)
            item = FileItem(filepath=filepath)
            if item.childCount() > 0:
                self.ui.fileList.insertTopLevelItem(0, item)

    def loadData(self, filepath, datasetName):
        self.filepath = str(filepath)
        self.curve = None
        self.imageData = None
        basename = os.path.basename(self.filepath)
        data = load_data(self.filepath, datasetName)
        if type(data) == list:
            if self.h5obj is not None:
                self.h5obj.close()
                self.h5obj = None
            self.h5obj = data[0]
            data = data[1]
        if len(data.shape) == 1:  # 1d curve
            if data.size == 1:  # single number
                return None
            else:
                self.curve = data
                return None
        self.imageData = data
        self.imageShape = data.shape
        _shape_str = ''
        if len(self.imageShape) == 2:
            _x, _y = self.imageShape
            _shape_str = 'x: %d, y: %d' %(_x, _y)
        elif len(self.imageShape) == 3:
            _x, _y, _z = self.imageShape
            _shape_str = 'x: %d, y: %d, z: %d' %(_x, _y, _z)
        self.params.param('Data Info', 'Image Shape').setValue(_shape_str)
        self.params.param('Data Info', 'File').setValue(basename)
        self.params.param('Data Info', 'Dataset').setValue(datasetName)
        if len(self.imageShape) == 3:
            self.dispShape = self.imageData.shape[1:3]
        else:
            self.dispShape = self.imageShape
        self.center = self.calcCenter()
        self.setCenterInfo()

    def setCenterInfo(self):
        self.params.param('Basic Operation', 'Image', 'Center x').setValue(self.center[0])
        self.params.param('Basic Operation', 'Image', 'Center y').setValue(self.center[1])

    def calcCenter(self):
        if len(self.imageShape) == 2:
            center = [self.imageShape[1]//2, self.imageShape[0]//2]
            return center
        assert len(self.imageShape) == 3
        if self.axis == 'x':
            center = [self.imageShape[2]//2, self.imageShape[1]//2]
        elif self.axis == 'y':
            center = [self.imageShape[2]//2, self.imageShape[0]//2]
        else:
            center = [self.imageShape[1]//2, self.imageShape[0]//2]
        return center

    def maybeShowSmallData(self):
        if self.showSmallData:
            if not self.ui.smallDataWidget.isVisible():
                self.ui.smallDataWidget.show()
            self.ui.smallDataWidget.setLabels(bottom='index', left='metric')
            if self.smallDataFile is not None and self.smallDataset is not None:
                paths, frames, smallData = load_smalldata(self.smallDataFile, self.smallDataset)
                self.smallDataPaths = paths
                self.smallDataFrames = frames
                self.smallData = smallData
                if self.smallDataSorted:
                    index_array = np.argsort(self.smallData).astype(np.int32)
                    self.smallDataItem.setData(x=np.arange(self.smallData.size), y=self.smallData[index_array])
                else:
                    self.smallDataItem.setData(x=np.arange(self.smallData.size), y=self.smallData)
            
    def maybePlotProfile(self):
        if self.showProfile and self.dispData is not None:
            if self.mask is None:
                self.mask = np.ones_like(self.dispData)
            mask = self.mask.copy()
            if self.maskFlag == True:
                assert mask.shape == self.dispData.shape
            if self.profileType == 'radial':
                profile = calc_radial_profile(self.dispData, self.center, mask=mask, mode=self.profileMode)
            elif self.profileType == 'angular':
                annulus_mask = make_annulus(self.dispShape, self.angularRmin, self.angularRmax)
                profile = calc_angular_profile(self.dispData, self.center, mask=mask*annulus_mask, mode=self.profileMode)
            else:  # across center line
                profile = calc_across_center_line_profile(self.dispData, self.center, angle=self.lineAngle, width=self.lineWidth, mask=self.mask, mode=self.profileMode)
            if self.profilePlotLog == True:
                profile[profile < 1.] = 1.
            self.profileItem.setData(profile)
            if self.profileType == 'radial':
                self.ui.profileWidget.setTitle('Radial Profile')
                self.ui.profileWidget.setLabels(bottom='r/pixel')
            elif self.profileType == 'angular':
                self.ui.profileWidget.setTitle('Angular Profile')
                self.ui.profileWidget.setLabels(bottom='theta/deg')
            else:
              self.ui.profileWidget.setTitle('Across Center Line Profile')
              self.ui.profileWidget.setLabels(bottom='index/pixel')
            if self.smoothFlag == True:
                smoothed_profile = savgol_filter(profile, self.smoothWinSize, self.polyOrder)
                if self.profilePlotLog == True:
                    smoothed_profile[smoothed_profile < 1.] = 1.
                self.smoothItem.setData(smoothed_profile)
            else:
                self.smoothItem.clear()
            profile_with_noise = profile.astype(np.float) + np.random.rand(profile.size)*1E-5  # add some noise to avoid same integer value in profile
            if self.extremaSearch == True:
                if self.extremaType == 'max':
                    extrema_indices = argrelmax(profile_with_noise, order=self.extremaWinSize)[0]
                else:
                    extrema_indices = argrelmin(profile_with_noise, order=self.extremaWinSize)[0]
                #print_with_timestamp('before filtered by threshold: %s' %str(extrema_indices))
                extremas = profile[extrema_indices]
                filtered_extrema_indices = extrema_indices[extremas>self.extremaThreshold*profile.mean()]
                filtered_extremas = profile[filtered_extrema_indices]
                #print_with_timestamp('after filtered by threshold: %s' %(filtered_extrema_indices))
                self.thresholdItem.setData(np.ones_like(profile)*profile.mean()*self.extremaThreshold)
            else:
                self.thresholdItem.clear()

    def calcDispData(self):
        if self.imageData is None:
            return None
        elif len(self.imageShape) == 3:
            _x, _y, _z = self.imageShape
            if self.axis == 'x':
                if 0 <= self.frameIndex < _x:
                    dispData = self.imageData[self.frameIndex, :, :]
                else:
                    #print_with_timestamp("ERROR! Index out of range. %s axis frame %d" %(self.axis, self.frameIndex))
                    QtWidgets.QMessageBox.question(self, 'Error',
                            "ERROR! Index out of range. %s axis frame %d" % (self.axis, self.frameIndex), QtWidgets.QMessageBox.Ok)
                    return None
            elif self.axis == 'y':
                if 0 <= self.frameIndex < _y:
                    dispData = self.imageData[:, self.frameIndex, :]
                else:
                    #print_with_timestamp("ERROR! Index out of range. %s axis frame %d" %(self.axis, self.frameIndex))
                    QtWidgets.QMessageBox.question(self, 'Error',
                            "ERROR! Index out of range. %s axis frame %d" % (self.axis, self.frameIndex), QtWidgets.QMessageBox.Ok)
                    return None
            else:
                if 0 <= self.frameIndex < _z:
                    dispData = self.imageData[:, :, self.frameIndex]
                else:
                    #print_with_timestamp("ERROR! Index out of range. %s axis frame %d" %(self.axis, self.frameIndex))
                    QtWidgets.QMessageBox.question(self, 'Error',
                            "ERROR! Index out of range. %s axis frame %d" % (self.axis, self.frameIndex), QtWidgets.QMessageBox.Ok)
                    return None
        elif len(self.imageShape) == 2:
            dispData = self.imageData
        if isinstance(dispData, np.ndarray):
            dispData = dispData.copy()
        else:
            dispData = np.asarray(dispData).copy()
        if self.imageLog:
            dispData[dispData<1.] = 1.
            dispData = np.log(dispData)
        if self.FFT:
            dispData = np.abs(np.fft.fft2(dispData))
        if self.FFTSHIFT:
            dispData = np.fft.fftshift(dispData)
        return dispData

    def closeEvent(self, event):
        global data_viewer_window
        reply = QtWidgets.QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            #print_with_timestamp('Bye-Bye.')
            #pg.exit()
            if self.h5obj is not None:
                self.h5obj.close()
            del data_viewer_window
            data_viewer_window = None
        else:
            event.ignore()

    def changeDatasetSlot(self, item, column):
        if isinstance(item, DatasetItem):
            datasetItem = item
            fileItem = datasetItem.parent()
        else:
            assert isinstance(item, FileItem)
            fileItem = item
            datasetItem = fileItem.child(0)
        self.loadData(fileItem.filepath, datasetItem.datasetName)  # maybe 1d, 2d or 3d dataset
        self.maybeChangeCurve(name=datasetItem.datasetName)
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def maybeChangeCurve(self, name=None):
        if not self.showCurvePlot or self.curve is None:
            return None
        self.curveItem.setData(self.curve)
        if name is not None:
            self.ui.curveWidget.getPlotItem().setTitle('Curve Plot -- %s' %name)

    def maybeChangeDisp(self):
        if not self.showImage or self.imageData is None:
            return None
        dispData = self.calcDispData()
        if dispData is None:
            return None
        self.dispShape = dispData.shape
        if self.maskFlag:
            if self.mask is None:
                self.mask = np.ones(self.dispShape)
            assert self.mask.shape == self.dispShape
            dispData *= self.mask
        if self.binaryFlag:
            dispData[dispData < self.dispThreshold] = 0.
            dispData[dispData >= self.dispThreshold] = 1.
        self.dispData = dispData
        # set dispData to distItem. Note: transpose the dispData to show image with same manner in matplotlib
        self.ui.imageView.setImage(self.dispData.T, autoRange=self.imageAutoRange, autoLevels=self.imageAutoLevels, autoHistogramRange=self.imageAutoHistogramRange)
        if self.showRings:
            if len(self.ringRadiis) > 0:
                _cx = np.ones_like(self.ringRadiis) * self.center[0]
                _cy = np.ones_like(self.ringRadiis) * self.center[1]
                self.ringItem.setData(_cx, _cy, size=self.ringRadiis*2., symbol='o', brush=(255,255,255,0), pen='r', pxMode=False)
        self.centerMarkItem.setData([self.center[0]], [self.center[1]], size=10, symbol='+', brush=(255,255,255,0), pen='r', pxMode=False)
        Friedel_score = calc_Friedel_score(self.dispData, self.center, mask=self.mask, mode='relative')
        self.params.param('Data Info', 'Image Friedel Score').setValue(Friedel_score)

    def setLineAngleSlot(self, _, lineAngle):
        #print_with_timestamp('set line angle: %s' %str(lineAngle))
        self.lineAngle = lineAngle
        self.maybePlotProfile()

    def setLineWidthSlot(self, _, lineWidth):
        #print_with_timestamp('set line width: %s' %str(lineWidth))
        self.lineWidth = lineWidth
        self.maybePlotProfile()

    def applyImageLogSlot(self, _, imageLog):
        #print_with_timestamp('set image log: %s' %str(imageLog))
        self.imageLog = imageLog
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def applyFFTSlot(self, _, FFT):
        #print_with_timestamp('set image fft: %s' %str(FFT))
        self.FFT = FFT
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def applyFFTSHIFTSlot(self, _, FFTSHIFT):
        #print_with_timestamp('set image fft-shift: %s' %str(FFTSHIFT))
        self.FFTSHIFT = FFTSHIFT
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def setExtremaSearchSlot(self, _, extremaSearch):
        #print_with_timestamp('set extrema search: %s' %str(extremaSearch))
        self.extremaSearch = extremaSearch
        self.maybePlotProfile()

    def setExtremaWinSizeSlot(self, _, extremaWinSize):
        #print_with_timestamp('set extrema window size: %s' %str(extremaWinSize))
        self.extremaWinSize = extremaWinSize
        self.maybePlotProfile()

    def setExtremaTypeSlot(self, _, extremaType):
        #print_with_timestamp('set extrema type: %s' %str(extremaType))
        self.extremaType = extremaType
        self.maybePlotProfile()

    def setExtremaThresholdSlot(self, _, extremaThreshold):
        #print_with_timestamp('set extrema threshold: %s' %str(extremaThreshold))
        self.extremaThreshold = extremaThreshold
        self.maybePlotProfile()

    def axisChangedSlot(self, _, axis):
        #print_with_timestamp('axis changed.')
        self.axis = axis
        self.center = self.calcCenter()
        self.setCenterInfo()
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def frameIndexChangedSlot(self, _, frameIndex):
        #print_with_timestamp('frame index changed')
        self.frameIndex = frameIndex
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def showProfileSlot(self, _, showProfile):
        #print_with_timestamp('show or hide radial profile changed')
        if showProfile:
            #print_with_timestamp('show profile')
            self.ui.profileWidget.show()
        else:
            self.ui.profileWidget.hide()
        self.maybePlotProfile()

    def centerXChangedSlot(self, _, centerX):
        #print_with_timestamp('center X changed')
        self.center[0] = centerX
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def centerYChangedSlot(self, _, centerY):
        #print_with_timestamp('center Y changed')
        self.center[1] = centerY
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def showFileMenuSlot(self, position):
        fileMenu = QtWidgets.QMenu()
        item = self.ui.fileList.currentItem()
        if isinstance(item, DatasetItem):
            setAsMask = fileMenu.addAction("Set As Mask")
            action = fileMenu.exec_(self.ui.fileList.mapToGlobal(position))
            if action == setAsMask:
                filepath = item.parent().filepath
                mask = load_data(filepath, item.datasetName)
                if len(mask.shape) != 2:
                    raise ValueError('%s:%s can not be used as mask. Mask data must be 2d.' %(filepath, item.datasetName))
                self.mask = np.asarray(mask)
                self.params.param('Data Info', 'Mask').setValue("%s::%s" %(os.path.basename(filepath), item.datasetName))
        elif isinstance(item, FileItem):
            deleteAction = fileMenu.addAction("Delete")
            action = fileMenu.exec_(self.ui.fileList.mapToGlobal(position))
            if action == deleteAction:
                #print('deleting selected file')
                for item in self.ui.fileList.selectedItems():
                    self.ui.fileList.takeTopLevelItem(self.ui.fileList.indexOfTopLevelItem(item))

    def applyMaskSlot(self, _, mask):
        #print_with_timestamp('turn on mask: %s' % str(mask))
        self.maskFlag = mask
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def imageAutoRangeSlot(self, _, imageAutoRange):
        #print_with_timestamp('set image autorange: %s' % str(imageAutoRange))
        self.imageAutoRange = imageAutoRange
        self.maybeChangeDisp()

    def imageAutoLevelsSlot(self, _, imageAutoLevels):
        #print_with_timestamp('set image autolevels: %s' % str(imageAutoLevels))
        self.imageAutoLevels = imageAutoLevels
        self.maybeChangeDisp()

    def imageAutoHistogramRangeSlot(self, _, imageAutoHistogramRange):
        #print_with_timestamp('set image autohistogram: %s' % str(imageAutoHistogramRange))
        self.imageAutoHistogramRange = imageAutoHistogramRange
        self.maybeChangeDisp()

    def curvePlotAutoRangeSlot(self, _, plotAutoRange):
        #print_with_timestamp('set curve plot autorange: %s' % str(plotAutoRange))
        self.curvePlotAutoRange = plotAutoRange

    def curvePlotLogModeSlot(self, _, log):
        #print_with_timestamp('set curve plot log mode: %s' % str(log))
        if log:
            self.ui.curveWidget.setLogMode(y=True)
        else:
            self.ui.curveWidget.setLogMode(y=False)

    def profilePlotAutoRangeSlot(self, _, plotAutoRange):
        #print_with_timestamp('set profile plot autorange: %s' % str(plotAutoRange))
        if plotAutoRange:
            self.ui.profileWidget.getViewBox().enableAutoRange()
        else:
            self.ui.profileWidget.getViewBox().disableAutoRange()
        self.maybePlotProfile()

    def profilePlotLogModeSlot(self, _, log):
        #print_with_timestamp('set profile plot log mode: %s' % str(log))
        self.maybePlotProfile()
        if log:
            self.ui.profileWidget.setLogMode(y=True)
        else:
            self.ui.profileWidget.setLogMode(y=False)

    def smallDataPlotAutoRangeSlot(self, _, plotAutoRange):
        #print_with_timestamp('set small data plot autorange: %s' % str(plotAutoRange))
        if plotAutoRange:
            self.ui.smallDataWidget.getViewBox().enableAutoRange()
        else:
            self.ui.smallDataWidget.getViewBox().disableAutoRange()
        self.maybePlotProfile()

    def smallDataPlotLogModeSlot(self, _, log):
        #print_with_timestamp('set small data log mode: %s' % str(log))
        if log:
            self.ui.smallDataWidget.setLogMode(y=True)
        else:
            self.ui.smallDataWidget.setLogMode(y=False)

    def showRingsSlot(self, _, showRings):
        #print_with_timestamp('show rings: %s' %showRings)
        self.showRings = showRings
        if not self.showRings:
            self.ringItem.clear()
        self.maybeChangeDisp()

    def ringRadiiSlot(self, _, ringRadiis):
        #print_with_timestamp('set ring radiis: %s' %str(ringRadiis))
        ringRadiisStrList = ringRadiis.split()
        ringRadiis = []
        for ringRadiisStr in ringRadiisStrList:
            ringRadiis.append(float(ringRadiisStr))
        self.ringRadiis = np.asarray(ringRadiis)
        self.maybeChangeDisp()

    def setProfileTypeSlot(self, _, profileType):
        #print_with_timestamp('set profile type: %s' %str(profileType))
        self.profileType = profileType
        self.maybePlotProfile()

    def setProfileModeSlot(self, _, profileMode):
        #print_with_timestamp('set profile mode: %s' %str(profileMode))
        self.profileMode = profileMode
        self.maybePlotProfile()

    def binaryImageSlot(self, _, binaryImage):
        #print_with_timestamp('apply Binaryzation: %s' %str(binaryImage))
        self.binaryFlag = binaryImage
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def setDispThresholdSlot(self, _, threshold):
        #print_with_timestamp('set disp threshold: %.1f' %threshold)
        self.dispThreshold = threshold
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def setSmoothSlot(self, _, smooth):
        #print_with_timestamp('set smooth: %s' %str(smooth))
        self.smoothFlag = smooth
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def setWinSizeSlot(self, _, winSize):
        winSize = winSize
        if winSize % 2 == 0:
            winSize += 1  # winSize must be odd
        #print_with_timestamp('set smooth winsize: %d' %winSize)
        self.smoothWinSize = winSize
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def setPolyOrderSlot(self, _, polyOrder):
        #print_with_timestamp('set poly order: %d' %polyOrder)
        self.polyOrder = polyOrder
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def setAngularRminSlot(self, _, Rmin):
        #print_with_timestamp('set angular Rmin to %.1f' %Rmin)
        self.angularRmin = Rmin
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def setAngularRmaxSlot(self, _, Rmax):
        #print_with_timestamp('set angular Rmax to %.1f' %Rmax)
        self.angularRmax = Rmax
        self.maybeChangeDisp()
        self.maybePlotProfile()

    def mouseMoved(self, pos):
        if self.dispShape is None:
            return None
        mouse_point = self.ui.imageView.view.mapToView(pos)
        x, y = int(mouse_point.x()), int(mouse_point.y())
        filename = os.path.basename(str(self.filepath))
        if 0 <= x < self.dispData.shape[1] and 0 <= y < self.dispData.shape[0]:
            self.ui.statusbar.showMessage("%s x:%d y:%d I:%.2E" %(filename, x, y, self.dispData[y, x]), 5000)
        else:
            pass




'''
Call this function to start data viewer
'''
def show_data_viewer(parent):
    global data_viewer_window
    data_viewer_window = MainWindow(None)
    data_viewer_window.resize(900, 600)
    data_viewer_window.setWindowTitle("Scientific Data Viewer")
    try:
        data_viewer_window.show()
    except:
        QtWidgets.QMessageBox.question(data_viewer_window, 'Error',
                    "Sorry for th bug T_T" , QtWidgets.QMessageBox.Ok)


def is_shown():
    global data_viewer_window
    if data_viewer_window is not None:
        return True
    else:
        return False


def add_files(files):
    global data_viewer_window

    for f in files:
        try:
            if not os.path.isfile(f):
                raise ValueError("boomb!")
            data_viewer_window.maybeAddFile(f)
        except:
            reply = QtWidgets.QMessageBox.question(None, 'Error',
                    "Faile to load file %s" % f, QtWidgets.QMessageBox.Ok)

if __name__ == '__main__':
    # add signal to enable CTRL-C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QtGui.QApplication(sys.argv)
    win = MainWindow(None)
    win.resize(900, 600)
    win.setWindowTitle("Scientific Data Viewer")

    win.show()
    app.exec_()
            
