import sys
import os
SVFHOME = os.environ['SVFHOME'] + "/"
CACHEDIR = SVFHOME + "Cache/"
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebKit import *
from PyQt5.QtWebKitWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
import qdarkstyle
# Import Pillow:
from PIL import Image
from io import StringIO
from io import BytesIO
import numpy as np
from math import *
import requests
import scipy
import argparse
import math
import cv2
import time
import cv2
import caffe
import shapefile
EARTH_CIRCUMFERENCE = 6378137 

class GSVCapture(QtCore.QObject):
    EARTH_CIRCUMFERENCE = 6378137     # earth circumference in meters
    def great_circle_distance(self, lat1, lon1,lat2, lon2): # doctest: +ELLIPSIS
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
                math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                math.sin(dLon / 2) * math.sin(dLon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = EARTH_CIRCUMFERENCE * c
        return d
    def API_KEY(self):
        return "&key=AIzaSyBHxvklM3VDUILKVcGQmqJ07LyIbaKbvws"
    def API_HEADER(self):
        return "https://maps.googleapis.com/maps/api/streetview"
    #def getImage(self, outfile, panoid, xsize, ysize, fov, heading, pitch):
    #    url = self.API_HEADER() + "?size=" + str(xsize) + "x" + str(ysize) + "&pano=" + panoid + "&fov=" + str(fov) + "&heading=" + str(heading) + "&pitch=" + str(pitch) + self.API_KEY();
    #    mp3file = urllib2.urlopen(url)
    #    with open(outfile,'wb') as output:
    #         output.write(mp3file.read())
    #    print url
    def getImage(self, panoId, x, y, zoom,outdir):
        url = "https://" + "geo0.ggpht.com/cbk?cb_client=maps_sv.tactile&authuser=0&hl=en&panoid=" + panoId + "&output=tile&x=" + str(x) + "&y=" + str(y) + "&zoom=" + str(zoom) + "&nbt&fover=2"
        outfile = outdir + "/" + str(x) + "_" + str(y) + ".jpg"
        #http = urllib3.PoolManager()
        #response = http.request('GET', url)
        response = requests.get(url)
        file = BytesIO(response.content)
        return file
        #mp3file = urllib3.urlopen(url)
        #with open(outfile,'wb') as output:
        #     output.write(mp3file.read())
    def equirectangular2fisheye(self, infile, outfile,isClassified):
        img = Image.open(infile)
        width, height = img.size
        img = img.crop((0,0,width,height/2))
        width, height = img.size
        nparr = np.asarray(img.copy())
        red, green, blue = img.split()
        red = np.asarray(red)
        red.flags.writeable = True
        green = np.asarray(green)
        green.flags.writeable = True
        blue = np.asarray(blue)
        blue.flags.writeable = True
        #green[np.where(green == 128)] = 0
        #blue[np.where(blue == 128)] = 0
        fisheye = np.ndarray(shape=(512,512,3), dtype=np.uint8)
        fisheye.fill(0) # Transpose back needed
        fisheyesize = 512
        x = np.arange(0,512,dtype=float)
        x = x / 511.0;
        x = (x - 0.5) * 2;
        x = np.tile(x,(512,1))
        y = x.transpose();
        dist2ori = np.sqrt((y * y) + (x * x))

        zenithD = dist2ori * 90.0
        zenithD[np.where(zenithD <= 0.000000001)] = 0.000000001
        zenithR = zenithD * 3.1415926 / 180.0
        wproj = np.sin(zenithR) / (zenithD / 90.0);#weight for equal-areal projection
        x2 = np.ndarray(shape=(512,512),dtype=float)
        x2.fill(0.0)
        y2 = np.ndarray(shape=(512,512),dtype=float)
        y2.fill(1.0)
        cosa = (x*x2 + y*y2) / np.sqrt((x*x + y*y) * (x2*x2+ y2*y2));
        lon = np.arccos(cosa) * 180.0 / 3.1415926;
        indices = np.where(x > 0)
        lon[indices] = 360.0 - lon[indices]
        lon = 360.0 - lon
        lon = 1.0 -(lon / 360.0)
        outside = np.where(dist2ori > 1)
        lat = dist2ori
        srcx = (lon*(width-1)).astype(int)
        srcy = (lat*(height-1)).astype(int)
        srcy[np.where(srcy > 255)] = 0
        maxx = np.max(srcx)
        maxy = np.max(srcy)
        indices = (srcx + srcy*width).tolist();

        red = np.take(red,np.array(indices))
        red[outside] = 0
        green = np.take(green,np.array(indices))
        green[outside] = 0
        blue = np.take(blue,np.array(indices))
        blue[outside] = 0
        red[outside] = 255
        green[outside] = 255
        blue[outside] = 255
        svf = 1    
        if isClassified:
            combined = 65536 * red + 256 * green + blue
            skyIndices = np.where(combined == (65536 * 128 + 256 * 128 + 128))
            nonSkyIndices = np.where((combined != (65536 * 128 + 256 * 128 + 128)) & (combined != 0))
            svf = np.sum(wproj[skyIndices]) / (np.sum(wproj[skyIndices]) + np.sum(wproj[nonSkyIndices]))
            red[skyIndices] = 0
            green[skyIndices] = 0
            blue[skyIndices] = 255
            red[nonSkyIndices] = 255
            green[nonSkyIndices] = 0
            blue[nonSkyIndices] = 0
            red[outside] = 255
            green[outside] = 255
            blue[outside] = 255
        fisheye = np.dstack((red, green, blue))
        Image.fromarray(fisheye).save(outfile)
        return svf
    @QtCore.pyqtSlot(str)
    def capture(self, arg):
        splits = arg.split(",")
        lat = str(splits[0]).strip()
        lon = str(splits[1]).strip()
        panoid = str(splits[2])
        print (lat + "," + lon + "," + panoid)
        outdir = CACHEDIR + panoid
        if not os.path.exists(outdir):
           os.makedirs(outdir) 
        #self.getImage(outdir + "/POS_Y.jpg",panoid,600,600,90,0,0)
        #self.getImage(outdir + "/POS_X.jpg",panoid,600,600,90,90,0)
        #self.getImage(outdir + "/NEG_Y.jpg",panoid,600,600,90,180,0)
        #self.getImage(outdir + "/NEG_X.jpg",panoid,600,600,90,270,0)
        #self.getImage(outdir + "/NEG_Z.jpg",panoid,600,600,90,0,-90)
        #self.getImage(outdir + "/POS_Z.jpg",panoid,600,600,90,0,90)
        tilesize = 512
        numtilesx = 4
        numtilesy = 2
        mosaicxsize = tilesize*numtilesx
        mosaicysize = tilesize*numtilesy
        mosaic = Image.new("RGB", (mosaicxsize, mosaicysize), "black")
        for x in range(0, numtilesx):
            for y in range(0, numtilesy):
                 img = Image.open(self.getImage(panoid, x, y, 2,outdir));
                 #img.save(outdir + "/" + str(x) + "_" + str(y) + ".jpg")
                 mosaic.paste(img,(x*tilesize,y*tilesize,x*tilesize+tilesize,y*tilesize+tilesize))
        xstart =  (512 - 128) / 2;
        xsize = mosaicxsize - xstart * 2;
        ysize = mosaicysize - (512 - 320);
        mosaic = mosaic.crop((xstart,0,xstart+xsize,ysize))
        mosaic = mosaic.resize((1024,512))
        mosaic.save(outdir + "/mosaic.png")
        self.classify(outdir + "/mosaic.png",outdir + "/mosaic_segnet.png")
        self.equirectangular2fisheye(outdir + "/mosaic.png",outdir + "/fisheye.jpg",False)
        svf = self.equirectangular2fisheye(outdir + "/mosaic_segnet.png",outdir + "/fisheye_segnet.png",True)
        file = open(outdir + "/panoinfo.txt","w") 
        file.write(lat + "," + lon + "," + panoid + "," + str(svf)) 
        file.close() 
        return [svf, "file:" + "///" + outdir + "/fisheye.jpg"];  
    @QtCore.pyqtSlot(str)
    def updateCoords(self, arg):
        splits = arg.split(",")
        lat = float(str(splits[0]).strip())
        lon = float(str(splits[1]).strip())
        mapInfo.GUI.coords_line.setText("%05f,%05f" % (lat, lon))
    def classify(self,infile,outfile):
        input_image = cv2.imread(infile)
        input_image = cv2.resize(input_image, (self.input_shape[3],self.input_shape[2]))
        input_image = input_image.transpose((2,0,1))
        input_image = np.asarray([input_image])
        out = self.segnet.forward_all(data=input_image)
        segmentation_ind = np.squeeze(self.segnet.blobs['argmax'].data)
        segmentation_ind_3ch = np.resize(segmentation_ind,(3,self.input_shape[2],self.input_shape[3]))
        segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
        segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
        cv2.LUT(segmentation_ind_3ch,self.label_colours,segmentation_rgb)
        scipy.misc.toimage(segmentation_rgb, cmin=0.0, cmax=255).save(outfile)
    def initialize(self,useCUDA):

        segnetModel = SVFHOME + "SegNet-Tutorial-master/Example_Models/segnet_model_driving_webdemo.prototxt"
        segnetWeights = SVFHOME +"SegNet-Tutorial-master/Example_Models/segnet_weights_driving_webdemo.caffemodel"
        segnetColours = SVFHOME +"SegNet-Tutorial-master/Scripts/camvid11.png"
        if os.path.exists(segnetWeights) == False:
            f = open(segnetWeights, 'wb')
            f1 = open(SVFHOME +"SegNet-Tutorial-master/Example_Models/segnet_weights_driving_webdemo_1.caffemodel", 'rb')
            f.write(f1.read())
            f1.close()
            f2 = open(SVFHOME +"SegNet-Tutorial-master/Example_Models/segnet_weights_driving_webdemo_2.caffemodel", 'rb')
            f.write(f2.read())
            f2.close()
            f.close()
        #f = open(segnetWeights, 'rb')
        #f.seek(0, os.SEEK_END)
        #fsize = f.tell()
        #f.seek(0)
        #half = fsize / 2
        #fdata = f.read(half)
        #f1 = open(SVFHOME +"SegNet-Tutorial-master/Example_Models/segnet_weights_driving_webdemo_1.caffemodel", 'wb')
        #f1.write(fdata)
        #f1.close()
        #fdata = f.read(fsize-half)
        #f2 = open(SVFHOME +"SegNet-Tutorial-master/Example_Models/segnet_weights_driving_webdemo_2.caffemodel", 'wb')
        #f2.write(fdata)
        #f2.close()
        #f.close()
        self.segnet = caffe.Net(segnetModel,segnetWeights,caffe.TEST) 
        caffe.set_mode_cpu()   
        if useCUDA == True:
           caffe.set_mode_gpu()
        self.input_shape = self.segnet.blobs['data'].data.shape
        self.output_shape = self.segnet.blobs['argmax'].data.shape   
        self.label_colours = cv2.imread(segnetColours).astype(np.uint8)  
class Browser(QMainWindow):
    PanoIDs = []
    closeEmitApp = QtCore.pyqtSignal() 
    def createBTN(self, maxsize,name):
        btn = QPushButton()
        btn.setMaximumWidth(maxsize);
        btn.setText(name)
        btn.setMaximumHeight(25);
        return btn
    def createLabel(self, maxsize,name):
        label = QLabel()
        label.setMaximumWidth(maxsize);
        label.setText(name)
        label.setMaximumHeight(15);
        return label
    def __init__(self):
        QMainWindow.__init__(self)
        self.resize(800,600)
        self.centralwidget = QWidget(self)

        self.mainLayout = QVBoxLayout(self.centralwidget)
        self.mainLayout.setSpacing(0)

        self.horizontalLayout = QGridLayout()

        self.horizontalLayout.setAlignment(Qt.AlignLeft);
        self.coords_line = QLineEdit()
        self.coords_line.setMaximumWidth(150);
        self.coords_line.setMaximumHeight(25);
        self.coords_line.setToolTip("Coordinate format:\"lat,lon\"")

        self.bt_add = self.createBTN(100,"Add")
        self.bt_add.setToolTip('Add a sample point at the current coordinates')
        self.bt_add.clicked.connect(self.addPoint)

        self.bt_pick = self.createBTN(100,"Pick")
        self.bt_pick.setCheckable(True);
        self.bt_pick.setChecked(False)
        QToolTip.setFont(QFont('SansSerif', 10))
        #self.setToolTip('Enter picking mode to interactively gather sample points')
        self.bt_pick.setToolTip('Enter picking mode to interactively gather sample points.')
      


        self.bt_interpolate = self.createBTN(100,"Interpolate")
        self.bt_interpolate.setCheckable(True);
        self.bt_interpolate.setChecked(False)
        self.bt_interpolate.setToolTip('Interpolate between two sample points.')

        self.bt_zoom = self.createBTN(100,"Zoom")
        self.bt_zoom.setToolTip('Zoom to the given coordinates.')

        self.bt_remove = self.createBTN(100,"Remove")
        self.bt_remove.setToolTip('Drop the last sample point.')

        self.bt_compute = self.createBTN(100,"ComputeSVF")
        self.bt_compute.setToolTip('Starting computing SVF for the collection of sample points.')

        self.bt_rectSel = self.createBTN(100,"Select")
        self.bt_rectSel.setToolTip('Select sample points by dragging a rectangle.')
        self.bt_rectSel.setCheckable(True);
        self.bt_rectSel.setChecked(False)

        self.bt_export = self.createBTN(100,"Export")
        self.bt_export.setToolTip('Export selected sample points.')

        self.bt_sampledist =  QSpinBox();
        self.bt_sampledist.setMaximumWidth(125);
        self.bt_sampledist.setMaximumHeight(25);
        self.bt_sampledist.setRange(5, 1000);
        self.bt_sampledist.setSingleStep(5);
        self.bt_sampledist.setValue(5);
        self.bt_sampledist.setToolTip('The sample distance (in meters) determines how frequently interpolation is applied between two sample points')

        self.bt_zoom.clicked.connect(self.zoom)
        self.bt_remove.clicked.connect(self.removePoint)
        self.bt_compute.clicked.connect(self.compute)
        self.bt_rectSel.clicked.connect(self.select)
        self.bt_export.clicked.connect(self.export)

        self.horizontalLayout.addWidget(self.coords_line,0,0)
        self.horizontalLayout.addWidget(self.bt_pick,0,1)
        self.horizontalLayout.addWidget(self.bt_interpolate,0,2)
        self.horizontalLayout.addWidget(self.bt_sampledist,0,3)
        self.horizontalLayout.addWidget(self.bt_add,0,4)
        self.horizontalLayout.addWidget(self.bt_zoom,0,5)
        self.horizontalLayout.addWidget(self.bt_remove,0,6)
        self.horizontalLayout.addWidget(self.bt_compute,0,7)
        self.horizontalLayout.addWidget(self.bt_rectSel,0,8)
        self.horizontalLayout.addWidget(self.bt_export,0,9)
        
        self.mainLayout.addLayout(self.horizontalLayout)

        self.html = QWebView()
        self.html.settings().setAttribute(QWebSettings.DeveloperExtrasEnabled, True)
        self.html.settings().setAttribute(QWebSettings.PluginsEnabled, True)
        self.html.settings().setAttribute(QWebSettings.JavascriptEnabled, True)
        self.html.settings().setAttribute(QWebSettings.AutoLoadImages, True)
        self.html.settings().setAttribute(QWebSettings.LocalStorageEnabled, True)
        self.html.settings().setAttribute(QWebSettings.LocalStorageDatabaseEnabled, True)
        self.html.settings().setAttribute(QWebSettings.LocalContentCanAccessRemoteUrls, True)

        self.html.settings().setAttribute(QWebSettings.JavascriptCanOpenWindows, True)
        self.html.settings().setAttribute(QWebSettings.JavascriptCanAccessClipboard, True)
        self.html.settings().setAttribute(QWebSettings.PrintElementBackgrounds, True)
        self.html.settings().setAttribute(QWebSettings.LocalContentCanAccessFileUrls, True)


        self.mainLayout.addWidget(self.html)
        self.setCentralWidget(self.centralwidget)

        self.default_url = "file:///" + SVFHOME + "GoogleMaps.html"
       # self.tb_url.setText(self.default_url)
        #self.browse()
        self.html.load(QtCore.QUrl(self.default_url))
        extractAction = QAction("&GET TO THE CHOPPAH!!!", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('Leave The App')
        #extractAction.triggered.connect(self.close_application)
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
    def addPoint(self):
        self.html.page().mainFrame().evaluateJavaScript("addPoint()")
    def select(self):
        self.html.page().mainFrame().evaluateJavaScript("select()")
    def export(self):
        selection = self.html.page().mainFrame().evaluateJavaScript("exportSel()")
        splits = selection.split(",")
        shp = shapefile.Writer(shapefile.POINT)
        shp.autoBalance = 1
        # create the field names and data type for each.
        shp.field("lat", 'N', decimal=15)
        shp.field("lon",  'N', decimal=15)
        shp.field("panoid", "C")
        shp.field("svf",  'N', decimal=10)

        fileout = open(self.mapInfo.CACHEDIR + "/test.csv","w") 
        fileout.write("lat,lon,panoid,svf\n") 
        for i in range(0, len(splits)):
            idx = int(splits[i])
            print(self.PanoIDs[idx])
            panoid = self.PanoIDs[idx]
            outdir = self.mapInfo.CACHEDIR + panoid
            file = open(outdir + "/panoinfo.txt","r") 
            line = file.readline()
            fields = line.split(",")
            file.close() 
            lon = float(fields[1])
            lat = float(fields[0])
            shp.point(lon,lat)
            shp.record(lat, lon, fields[2], float(fields[3]))
            fileout.write(line + "\n") 
        fileout.close() 
        shp.save(self.mapInfo.CACHEDIR + "/test.shp")
    def removePoint(self):
        self.html.page().mainFrame().evaluateJavaScript("removePoint()")
    def zoom(self):
        coorstr = self.coords_line.text();
        if coorstr == "":
           return
        splits = coorstr.split(",")
        if splits.count < 2:
           return
        lat = float(str(splits[0]).strip())
        lon = float(str(splits[1]).strip())
        self.html.page().mainFrame().evaluateJavaScript("zoom({0},{1})".format(lat,lon))
    def compute(self):
        samplenum = int(self.html.page().mainFrame().evaluateJavaScript("getSampleNum()"))
        for i in range(0, samplenum):
            results = self.html.page().mainFrame().evaluateJavaScript("getSample({0})".format(i))
            splits = results.split(",")
            lat = str(splits[0]).strip()
            lon = str(splits[1]).strip()
            panoid = str(splits[2])
            svf_results = self.mapInfo.capture(results)
            self.PanoIDs.append(panoid)
            print("setMarker({0},{1},\"{2}\")".format(i,svf_results[0],svf_results[1]))
            self.html.page().mainFrame().evaluateJavaScript("setMarker({0},{1},\"{2}\")".format(i,svf_results[0],svf_results[1]))
        self.html.page().mainFrame().evaluateJavaScript("clear()")
    @QtCore.pyqtSlot(str)
    def click(self, arg):
        splits = arg.split(",")
        lat = float(str(splits[0]).strip())
        lon = float(str(splits[1]).strip())
        setSamplingArgs = "setSampling ({0},{1})".format(str(self.bt_interpolate.isChecked()).lower(),self.bt_sampledist.value())
        #setSamplingArgs = "setSampling()"
        #print(setSamplingArgs)
        if self.bt_pick.isChecked():
           self.html.page().mainFrame().evaluateJavaScript(setSamplingArgs)
           self.html.page().mainFrame().evaluateJavaScript("addPoint()")
        self.coords_line.setText("%05f,%05f" % (lat, lon))
        #self.html.page().mainFrame().evaluateJavaScript("great_circle_distance(0,0,0,0)")
    @QtCore.pyqtSlot(str)
    def dblclick(self, arg):
        splits = arg.split(",")
        lat = float(str(splits[0]).strip())
        lon = float(str(splits[1]).strip())
        #mapInfo.GUI.coords_line.setText("%05f,%05f" % (lat, lon))