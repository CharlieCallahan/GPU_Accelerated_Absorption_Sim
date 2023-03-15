from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)

import pyqtgraph as pg
import numpy as np
import gaas_ocl as gaas
import os

class FloatSlider(QWidget):
    def __init__(self, min, max, name):
        QWidget.__init__(self)
        self.min = min
        self.max = max
        self.layout = QHBoxLayout()
        label_name = QLabel(name)
        self.layout.addWidget(label_name, stretch=1)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0,1000)
        self.slider.setSliderPosition(500)
        self.layout.addWidget(self.slider, stretch=4)
        self.label_val = QLabel(str((max-min)/2))
        self.layout.addWidget(self.label_val)
        self.setLayout(self.layout)
        self.slider.valueChanged[int].connect(self.updateVal)
        
    def updateVal(self, value):
        text = "%.2f" % (self.getVal())
        self.label_val.setText(text)

    def getVal(self):
        sliderVal = self.slider.value()
        return self.min + sliderVal/1000*(self.max-self.min)


class PlotWindow(QDialog):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16)

        pg.setConfigOption('background', 'w')
        self.plot = pg.plot()
        self.plot.showGrid(x = True, y = True)
        self.plot.setXRange(0, 10)       
        # setting vertical range
        self.plot.setYRange(0, 20)
        self.plot.addLegend()

        self.mainLayout.addWidget(self.plot,0,0)
        self.make_sliders()

        self.startWavenum = 5000
        self.endWavenum = 5600
        self.wavenumStep = 0.04 #wavenums per simulation step
        self.mol = 'H2O' #'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3'
        self.iso = 1 #isotopologue num
        cwd = os.path.dirname(os.path.realpath(__file__))
        if sys.platform == 'win32' or sys.platform == 'win64':
            dbdir = cwd + "\\DBDir\\"
        else:
            dbdir = cwd + "/DBDir/"
        if (not os.path.isdir(dbdir)):
            #need to make database directory
            os.mkdir(dbdir)

        self.absDB = gaas.gen_abs_db("H2O",self.iso,self.startWavenum,self.endWavenum,dbdir,loadFromHITRAN=True)
        self.tipsCalc = gaas.get_tips_calc(self.mol,self.iso)
        self.nus= np.zeros(100) 
        self.coefs = np.zeros(100)
        self.updateSim()

    def update_plot(self):
        pen = pg.mkPen('r', width=3)
        self.plot.plot(self.nus, self.coefs, pen = pen, clear=True)

    def make_sliders(self):
        self.Temp = FloatSlider(200,4000,"Temp (K)")
        self.Conc = FloatSlider(0.001,0.5,"Concentration")
        self.Pressure = FloatSlider(0.1,50,"Pressure")
        self.mainLayout.addWidget(self.Temp)
        self.mainLayout.addWidget(self.Conc)
        self.mainLayout.addWidget(self.Pressure)
        self.Temp.slider.valueChanged.connect(self.updateSim)
        self.Conc.slider.valueChanged.connect(self.updateSim)
        self.Pressure.slider.valueChanged.connect(self.updateSim)

    def updateSim(self):
        T = self.Temp.getVal()
        C = self.Conc.getVal()
        P = self.Pressure.getVal()
        print("T: ",T)
        print("C: ",C)
        print("P: ",P)
        print("Simulating")
        
        self.nus,self.coefs = gaas.simVoigt(T,P,C,self.wavenumStep,self.startWavenum,self.endWavenum,self.mol,self.iso,self.absDB,self.tipsCalc)

    
if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = PlotWindow()
    gallery.show()
    sys.exit(app.exec())