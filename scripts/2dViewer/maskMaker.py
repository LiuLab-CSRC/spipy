# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
from pyqtgraph import PlotDataItem, PlotItem
from scipy.signal import savgol_filter, argrelmax, argrelmin
from util import *
import numpy as np
import glob
from functools import partial

from layout_mm import Ui_MainWindow
from util import DatasetTreeWidget, MyImageView, MyParameterTree, MyPlotWidget

data_viewer_window = None


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        _dir = os.path.dirname(os.path.abspath(__file__))
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.splitter_3.setSizes([self.width()*0.7, self.width()*0.3])   # splitter between image_view and file list
        self.setAcceptDrops(True)
        self.ui.fileList.setColumnWidth(0, 200)

        self.acceptedFileTypes = [u'npy', u'npz', u'h5', u'mat', u'cxi', u'tif']
        self.filepath = None  # current filepath
        self.imageData = None  # original image data, 2d or 3d
        self.imageShape = None
        self.dispData = None  # 2d data for plot
        self.dispShape = None
        self.mask = None    # mask buffer, 'True' means good pixel, 'False' means masked pixel
        self.maskDisp = None
        self.h5obj = None

        self.dispItem = self.ui.imageView.getImageItem()
        self.centerMarkItem = pg.ScatterPlotItem()
        self.maskItem = pg.ImageItem()
        self.rectroi = pg.RectROI([50,0],[30,30],pen=(4,9))
        self.rectroi.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
        self.circroi = pg.CircleROI([0,50], [30,30],pen=(4,9))
        self.circroi.setAcceptedMouseButtons(QtCore.Qt.AllButtons)
        self.polyroi = pg.PolyLineROI([[0,0],[30,0],[30,30],[0,30]],closed=True,pen=(4,9))
        self.polyroi.setAcceptedMouseButtons(QtCore.Qt.AllButtons)

        self.ui.imageView.getView().addItem(self.maskItem)
        self.ui.imageView.getView().addItem(self.centerMarkItem)
        if self.ui.circCheck.isChecked():
            self.ui.imageView.getView().addItem(self.circroi)
        if self.ui.rectCheck.isChecked():
            self.ui.imageView.getView().addItem(self.rectroi)
        if self.ui.polyCheck.isChecked():
            self.ui.imageView.getView().addItem(self.polyroi)

        # basic operation for image plot
        self.showImage = True
        self.axis = 'x'
        self.frameIndex = 0
        self.imageLog = False
        self.FFTSHIFT = False
        self.FFTSHIFT_mask = False
        self.center = [0, 0]
        self.mask_good_bad = [0, 1]
        self.show_mask_flag = True

        # display option
        self.imageAutoRange = False
        self.imageAutoLevels = False

        params_list = [
        				{'name': 'Data Info', 'type': 'group', 'children': [
                            {'name': 'File', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Dataset', 'type': 'str', 'value': 'not set', 'readonly': True},
                            {'name': 'Image Shape', 'type': 'str', 'value': 'unknown', 'readonly': True},
                            {'name': 'Image Friedel Score', 'type': 'float', 'readonly': True},
                            {'name': 'Mask', 'type': 'str', 'value': 'not set', 'readonly': True},
                        ]},
                        {'name': 'Basic Operation', 'type': 'group', 'children': [
                            {'name': 'Image', 'type': 'group', 'children': [
                                {'name': 'Axis', 'type': 'list', 'values': ['x','y','z'], 'value': self.axis},
                                {'name': 'Frame Index', 'type': 'int', 'value': self.frameIndex},
                                {'name': 'Log Scale', 'type': 'bool', 'value': self.imageLog},
                                {'name': 'FFT-SHIFT', 'type': 'bool', 'value': self.FFTSHIFT},
                                {'name': 'Center x', 'type': 'int', 'value': self.center[1]},
                                {'name': 'Center y', 'type': 'int', 'value': self.center[0]},
                            ]},
                            {'name': 'Mask', 'type': 'group', 'children': [
                                {'name': 'FFT-SHIFT', 'type': 'bool', 'value': self.FFTSHIFT_mask},
                                {'name': 'Good Pixel', 'type': 'int', 'value': self.mask_good_bad[0]},
                                {'name': 'Bad Pixel', 'type': 'int', 'value': self.mask_good_bad[1]},
                                {'name': 'Show Mask', 'type': 'bool', 'value': self.show_mask_flag},
                            ]},
                        ]},
                        {'name': 'Display Option', 'type': 'group', 'children': [
                            {'name': 'autoRange', 'type': 'bool', 'value': self.imageAutoRange},
                            {'name': 'autoLevels', 'type': 'bool', 'value': self.imageAutoLevels},
                        ]}
        ]

        self.params = Parameter.create(name='params', type='group', children=params_list)
        self.ui.parameterTree.setParameters(self.params, showTop=False)

        self.ui.fileList.itemDoubleClicked.connect(self.changeDatasetSlot)
        self.ui.fileList.customContextMenuRequested.connect(self.showFileMenuSlot)
        self.ui.imageView.scene.sigMouseMoved.connect(self.mouseMoved)
        self.ui.lineEdit.returnPressed.connect(self.addFilesSlot)
        self.rectroi.sigClicked.connect(self.roibuttonclickevent)
        self.circroi.sigClicked.connect(self.roibuttonclickevent)
        self.polyroi.sigClicked.connect(self.roibuttonclickevent)
        self.ui.circCheck.stateChanged.connect(partial(self.roistatechanged, 0))
        self.ui.rectCheck.stateChanged.connect(partial(self.roistatechanged, 1))
        self.ui.polyCheck.stateChanged.connect(partial(self.roistatechanged, 2))
        self.ui.saveButton.clicked.connect(self.saveMask)
        self.ui.clearButton.clicked.connect(self.clearAll)

        self.params.param('Basic Operation', 'Image', 'Axis').sigValueChanged.connect(self.axisChangedSlot)
        self.params.param('Basic Operation', 'Image', 'Frame Index').sigValueChanged.connect(self.frameIndexChangedSlot)
        self.params.param('Basic Operation', 'Image', 'Log Scale').sigValueChanged.connect(self.applyImageLogSlot)
        self.params.param('Basic Operation', 'Image', 'FFT-SHIFT').sigValueChanged.connect(self.applyFFTSHIFTSlot)
        self.params.param('Basic Operation', 'Image', 'Center x').sigValueChanged.connect(self.centerXChangedSlot)
        self.params.param('Basic Operation', 'Image', 'Center y').sigValueChanged.connect(self.centerYChangedSlot)
        self.params.param('Basic Operation', 'Mask', 'FFT-SHIFT').sigValueChanged.connect(self.applyMaskFFTSHIFTSlot)
        self.params.param('Basic Operation', 'Mask', 'Good Pixel').sigValueChanged.connect(self.GoodPixelChangeSlot)
        self.params.param('Basic Operation', 'Mask', 'Bad Pixel').sigValueChanged.connect(self.BadPixelChangeSlot)
        self.params.param('Basic Operation', 'Mask', 'Show Mask').sigValueChanged.connect(self.applyShowMaskSlot)

        self.params.param('Display Option', 'autoRange').sigValueChanged.connect(self.imageAutoRangeSlot)
        self.params.param('Display Option', 'autoLevels').sigValueChanged.connect(self.imageAutoLevelsSlot)

    def closeEvent(self, event):
        global data_viewer_window
        reply = QtWidgets.QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            if self.h5obj is not None:
                self.h5obj.close()
            del data_viewer_window
            data_viewer_window = None
        else:
            event.ignore()

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

    def addFilesSlot(self):
        filePattern = str(self.ui.lineEdit.text())
        files = glob.glob(filePattern)
        for i in range(len(files)):
            self.maybeAddFile(files[i])

    def maybeAddFile(self, filepath):
        ext = QtCore.QFileInfo(filepath).suffix()
        if ext in self.acceptedFileTypes:
            maybeExistIndex = self.ui.fileList.indexOf(filepath)
            if maybeExistIndex != -1:
                self.ui.fileList.takeTopLevelItem(maybeExistIndex)
            item = FileItem(filepath=filepath)
            if item.childCount() > 0:
                self.ui.fileList.insertTopLevelItem(0, item)

    def changeDatasetSlot(self, item, column):
        if self.mask is not None:
            reply = QtWidgets.QMessageBox.warning(self, "Warning",
                "You are changing dataset, the mask you just made will be cleaned. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return
        if isinstance(item, DatasetItem):
            datasetItem = item
            fileItem = datasetItem.parent()
        else:
            assert isinstance(item, FileItem)
            fileItem = item
            datasetItem = fileItem.child(0)
        self.loadData(fileItem.filepath, datasetItem.datasetName)  # maybe 1d, 2d or 3d dataset
        self.maybeChangeDisp()

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
        if len(data.shape) < 2 or len(data.shape) > 3:
            QtWidgets.QMessageBox.critical(self, 'Error',
                        	"ERROR! Only support 2D or 3D data !", QtWidgets.QMessageBox.Ok)
            return
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
        # mask
        self.mask = np.ones(self.dispShape, dtype=bool)
        self.disp_mask(None, True, False)
        # center
        self.center = self.calcCenter()
        self.setCenterInfo()

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

    def setCenterInfo(self):
        self.params.param('Basic Operation', 'Image', 'Center x').setValue(self.center[0])
        self.params.param('Basic Operation', 'Image', 'Center y').setValue(self.center[1])

    def maybeChangeDisp(self):
        if not self.showImage or self.imageData is None:
            return None
        dispData = self.calcDispData()
        if dispData is None:
            return None
        self.dispShape = dispData.shape
        self.dispData = dispData
        # set dispData to distItem. Note: transpose the dispData to show image with same manner in matplotlib
        self.ui.imageView.setImage(self.dispData.T, autoRange=self.imageAutoRange, autoLevels=self.imageAutoLevels, autoHistogramRange=True)
        self.centerMarkItem.setData([self.center[0]], [self.center[1]], size=10, symbol='+', brush=(255,255,255,0), pen='r', pxMode=False)
        self.changeFriedelScore()

    def changeFriedelScore(self):
        Friedel_score = calc_Friedel_score(self.dispData, self.center, mask=self.mask, mode='mean')
        self.params.param('Data Info', 'Image Friedel Score').setValue(Friedel_score)

    def calcDispData(self):
        if self.imageData is None:
            return None
        elif len(self.imageShape) == 3:
            _x, _y, _z = self.imageShape
            if self.axis == 'x':
                if 0 <= self.frameIndex < _x:
                    dispData = self.imageData[self.frameIndex, :, :]
                else:
                    QtWidgets.QMessageBox.critical(self, 'Error',
                            "ERROR! Index out of range. %s axis frame %d" % (self.axis, self.frameIndex), QtWidgets.QMessageBox.Ok)
                    return None
            elif self.axis == 'y':
                if 0 <= self.frameIndex < _y:
                    dispData = self.imageData[:, self.frameIndex, :]
                else:
                    QtWidgets.QMessageBox.critical(self, 'Error',
                            "ERROR! Index out of range. %s axis frame %d" % (self.axis, self.frameIndex), QtWidgets.QMessageBox.Ok)
                    return None
            else:
                if 0 <= self.frameIndex < _z:
                    dispData = self.imageData[:, :, self.frameIndex]
                else:
                    QtWidgets.QMessageBox.critical(self, 'Error',
                            "ERROR! Index out of range. %s axis frame %d" % (self.axis, self.frameIndex), QtWidgets.QMessageBox.Ok)
                    return None
        elif len(self.imageShape) == 2:
            dispData = self.imageData
        if isinstance(dispData, np.ndarray):
            dispData = dispData.copy()
        else:
            dispData = np.asarray(dispData).copy()
        if self.imageLog:
            dispData[dispData<0] = 0
            dispData = np.log(dispData+1)
        if self.FFTSHIFT:
            dispData = np.fft.fftshift(dispData)
        return dispData

    def showFileMenuSlot(self, position):
        fileMenu = QtWidgets.QMenu()
        item = self.ui.fileList.currentItem()
        if isinstance(item, DatasetItem):
            setAsMask = fileMenu.addAction("Set As Mask")
            addToMask = fileMenu.addAction("Add to Mask")
            action = fileMenu.exec_(self.ui.fileList.mapToGlobal(position))
            if action == setAsMask or action == addToMask:
                filepath = item.parent().filepath
                data = load_data(filepath, item.datasetName)
                if type(data) == list:
                    mask = data[1][()]
                    data[0].close()
                else:
                    mask = data
                if len(mask.shape) != 2:
                    QtWidgets.QMessageBox.critical(self, 'Error', 
                        '%s:%s can not be used as mask. Mask data must be 2d.' %(filepath, item.datasetName))
                    return
                if action == setAsMask:
                    success = self.disp_mask(np.asarray(mask).T, True, False)
                elif action == addToMask:
                    success = self.disp_mask(np.asarray(mask).T, True, True)
                if not success:
                    QtWidgets.QMessageBox.critical(self, 'Error', 
                        "Mask file '%s:%s' is not compatible with 'Data Shape', 'Good Pixel' or 'Bad Pixel' settings." %(filepath, item.datasetName))
                    return
                self.params.param('Data Info', 'Mask').setValue("%s::%s" %(os.path.basename(filepath), item.datasetName))
        elif isinstance(item, FileItem):
            deleteAction = fileMenu.addAction("Delete")
            action = fileMenu.exec_(self.ui.fileList.mapToGlobal(position))
            if action == deleteAction:
                for item in self.ui.fileList.selectedItems():
                    self.ui.fileList.takeTopLevelItem(self.ui.fileList.indexOfTopLevelItem(item))

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

    def roistatechanged(self, obj):
        if obj == 0:
            if self.ui.circCheck.checkState() == QtCore.Qt.Checked:
                self.ui.imageView.getView().addItem(self.circroi)
            else:
                self.ui.imageView.getView().removeItem(self.circroi)
        elif obj == 1:
            if self.ui.rectCheck.checkState() == QtCore.Qt.Checked:
                self.ui.imageView.getView().addItem(self.rectroi)
            else:
                self.ui.imageView.getView().removeItem(self.rectroi)
        elif obj == 2:
            if self.ui.polyCheck.checkState() == QtCore.Qt.Checked:
                self.ui.imageView.getView().addItem(self.polyroi)
            else:
                self.ui.imageView.getView().removeItem(self.polyroi)
        else:
            pass

    def roibuttonclickevent(self, obj, e):
        if self.mask is None: return
        if e.button() == QtCore.Qt.RightButton or \
            (e.button() == QtCore.Qt.LeftButton and e.double()):
            temp = obj.getArrayRegion(np.ones(self.mask.shape).T, self.maskItem)
            topleft = obj.parentBounds().topLeft()
            topleft_p = np.array([[topleft.x(),topleft.y()]]).T # shape=(2,1)
            xy = np.array(np.where(temp==1))  # shape=(2,N)
            x,y = np.round(xy + topleft_p).astype(int)
            if e.double():
                self.maskDisp[(x,y)] = np.array([[0,0,0,0]]*len(x))
                self.mask[(x,y)] = True
            else:
                self.maskDisp[(x,y)] = np.array([[255,0,0,64]]*len(x))
                self.mask[(x,y)] = False
            self.disp_mask(None, False, False)
            self.changeFriedelScore()

    def disp_mask(self, new_mask=None, refresh_maskDisp=True, add_to_mask=False):
        if new_mask is not None:
            # check new_mask
            if new_mask.shape != self.dispShape: return False
            asy = list(set(new_mask.flatten()))
            if len(asy) > 2: return False
            for v in asy:
                if v not in self.mask_good_bad: return False
            if add_to_mask:
                if self.mask is None:
                    self.mask = np.zeros(new_mask.shape, dtype=bool)
            else:
                self.mask = np.zeros(new_mask.shape, dtype=bool)
                self.mask[np.where(new_mask==self.mask_good_bad[0])] = True
            self.mask[np.where(new_mask==self.mask_good_bad[1])] = False
        else:
            if self.mask is None: return False
        temp = list(self.mask.shape)
        temp.append(4)
        if refresh_maskDisp:
            del self.maskDisp
            self.maskDisp = np.zeros(temp)
            xy = np.where(self.mask==False)
            if len(xy[0]) > 0:
                self.maskDisp[xy] = np.array([[255,0,0,64]]*len(xy[0]))
                # apply operations
                if self.FFTSHIFT_mask:
                    self.maskDisp = np.fft.fftshift(self.maskDisp, axes=(0,1))
        if self.show_mask_flag:
            self.maskItem.setImage(self.maskDisp, opacity=0.7)
        else:
            self.maskItem.setImage(np.zeros(temp), opacity=0.7)
        return True

    def saveMask(self):
        if self.mask is None:
            QtWidgets.QMessageBox.critical(self, 'Error', "No mask detected.")
            return
        current_folder = os.path.dirname(self.filepath)
        fileName, fileType = QtWidgets.QFileDialog.getSaveFileName(self, \
            'Save Mask', current_folder, "Numpy File(*.npy);;HDF5 File(*.h5);;Matlab File(*.mat)")
        fileName = str(fileName)
        if fileName:
            fext = os.path.splitext(fileName)[-1]
            saved_mask = np.zeros(self.mask.shape, dtype=int)
            saved_mask[np.where(self.mask==False)] = self.mask_good_bad[1]
            saved_mask[np.where(self.mask==True)] = self.mask_good_bad[0]
            if fext == ".npy":
                np.save(fileName, saved_mask.T)
            elif fext == ".h5":
                with h5py.File(fileName, "w") as fp:
                    fp.create_dataset("mask", data=saved_mask.T, chunks=True, compression="gzip")
            elif fext == ".mat":
                sio.savemat(fileName, {'mask':saved_mask.T})
            else:
                QtWidgets.QMessageBox.critical(self, 'Error', "File type not supported.")
                return

    def clearAll(self):
        if self.mask is None:
            QtWidgets.QMessageBox.information(self, 'info', "These is no mask yet.")
            return
        reply = QtWidgets.QMessageBox.question(self, 'Warning',
            "The mask you just made will be cleaned, continue ?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            newmask = np.zeros(self.mask.shape, dtype=int) + self.mask_good_bad[0]
            success = self.disp_mask(newmask, True, False)
            if not success:
                QtWidgets.QMessageBox.critical(self, 'Error', 
                            "Failed to clear all masks.")
            return

    def axisChangedSlot(self, _, axis):
        self.axis = axis
        self.center = self.calcCenter()
        self.setCenterInfo()
        self.maybeChangeDisp()

    def frameIndexChangedSlot(self, _, frameIndex):
        self.frameIndex = frameIndex
        self.maybeChangeDisp()

    def applyImageLogSlot(self, _, imageLog):
        self.imageLog = imageLog
        self.maybeChangeDisp()

    def applyFFTSHIFTSlot(self, _, FFTSHIFT):
        self.FFTSHIFT = FFTSHIFT
        self.maybeChangeDisp()

    def centerXChangedSlot(self, _, centerX):
        self.center[0] = centerX
        self.maybeChangeDisp()

    def centerYChangedSlot(self, _, centerY):
        self.center[1] = centerY
        self.maybeChangeDisp()

    def applyMaskFFTSHIFTSlot(self, _, FFTSHIFT_mask):
        self.FFTSHIFT_mask = FFTSHIFT_mask
        success = self.disp_mask(None, True, False)
        if not success:
            QtWidgets.QMessageBox.critical(self, 'Error', 
                            "FFT shift of mask is failed.")
            return

    def applyShowMaskSlot(self, _, show_mask_flag):
        self.show_mask_flag = show_mask_flag
        success = self.disp_mask(None, True, False)
        if not success:
            QtWidgets.QMessageBox.critical(self, 'Error', 
                            "Show mask flag failed to change.")
            return

    def GoodPixelChangeSlot(self, _, good_pixel_value):
        self.mask_good_bad[0] = good_pixel_value

    def BadPixelChangeSlot(self, _, bad_pixel_value):
        self.mask_good_bad[1] = bad_pixel_value

    def imageAutoRangeSlot(self, _, imageAutoRange):
        self.imageAutoRange = imageAutoRange
        self.maybeChangeDisp()

    def imageAutoLevelsSlot(self, _, imageAutoLevels):
        self.imageAutoLevels = imageAutoLevels
        self.maybeChangeDisp()


if __name__ == '__main__':
    # add signal to enable CTRL-C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QtGui.QApplication(sys.argv)
    win = MainWindow(None)
    win.resize(900, 600)
    win.setWindowTitle("Pattern Mask Maker")

    win.show()
    app.exec_()