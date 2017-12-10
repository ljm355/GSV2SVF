import sys
import os
SVFHOME = os.environ['SVFHOME'] + "/"
sys.path.insert(0, SVFHOME + 'SegnetCUDA/pycaffe')
import SVFCore


if __name__ == "__main__":
    app = SVFCore.QApplication(sys.argv)
    #app.setStyle("plastique")
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    main = SVFCore.Browser()
    main.show()
    mapInfo = SVFCore.GSVCapture()
    mapInfo.GUI = main

    main.mapInfo = mapInfo
    mapInfo.initialize(True)
    #mapInfo.equirectangular2fisheyeMatrix("C:/GSV2SVF/Cache/Oq7WvBlQzmpm_eWBhA2ZJQ/mosaic.jpg","C:/GSV2SVF/Cache/Oq7WvBlQzmpm_eWBhA2ZJQ/fisheye.jpg")
    #main.html.page().mainFrame().addToJavaScriptWindowObject("mapInfo", mapInfo)
    main.html.page().mainFrame().addToJavaScriptWindowObject("mainWin", main)
    sys.exit(app.exec_())