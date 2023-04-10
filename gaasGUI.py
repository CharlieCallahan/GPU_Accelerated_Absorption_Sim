from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5 import QtGui
from PyQt5.QtGui import QMovie, QPainter
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)

import threading
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

from PyQt5 import QtWidgets, QtGui

class FloatInput(QWidget):
    def __init__(self, label_text="", default_value=0.0, parent=None):
        super().__init__(parent)

        # Create the label
        self.label = QLabel(label_text)
        
        # Create the input box
        self.input = QLineEdit(str(default_value))
        self.input.returnPressed.connect(self._on_value_changed)
        # Set the validator to allow only floating point numbers
        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.input.setValidator(validator)
        
        # Set the layout
        layout = QHBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        
        # Set the placeholder text to indicate the expected input format
        self.input.setPlaceholderText('Enter a number')
        
    def get_value(self):
        return float(self.input.text())
    
    def set_callback(self, func):
        self.callback = func
        
    def _on_value_changed(self):
        text = self.input.text()
        if self.callback:
            try:
                value = float(text)
            except ValueError:
                value = None
            self.callback(value)

class LoadingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up the layout and label
        layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        # self.label.setFont(QFont("Arial", 16))
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Set up the timer and counter
        self.timer = QTimer()
        self.counter = 0
        self.timer.timeout.connect(self.updateMessage)
        self.start()

    def start(self):
        # Set the loading message and start the timer
        self.label.setText("Loading")
        # self.show()
        self.timer.start(1000)

    def stop(self):
        # Stop the timer and hide the widget
        self.timer.stop()
        # self.hide()

    def updateMessage(self):
        # Update the loading message with ellipses
        self.counter = (self.counter + 1) % 4
        self.label.setText("Loading" + "." * self.counter)

class DropdownMenu(QWidget):
    def __init__(self, labels):
        super().__init__()

        # Create a label to display the selected value
        self.selected_label = QLabel()
        self.callback = None
        # Create the dropdown menu
        self.dropdown = QComboBox()
        self.dropdown.addItems(labels)
        self.dropdown.currentIndexChanged.connect(self.update_label)
        self.labels = labels
        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.dropdown)
        layout.addWidget(self.selected_label)
        self.setLayout(layout)

    def update_label(self, index):
        # Update the selected label when the dropdown value is changed
        self.selected_label.setText(f"Selected: {self.dropdown.currentText()}")
        if(self.callback):
            self.callback(self.labels[index])

    def set_callback(self, callback):
        self.callback = callback

class PlotWindow(QDialog):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.setWindowTitle("GAAS GUI")
        self.originalPalette = QApplication.palette()

        self.mainLayout = QGridLayout()
        self.setLayout(self.mainLayout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16)

        pg.setConfigOption('background', 'w')
        self.plot = pg.plot()
        self.plot.showGrid(x = True, y = True)
        self.plot.setLabel('left', "Absorbance")
        self.plot.setLabel('bottom', "Wavenumber (cm-1)")

        # setting vertical range
        self.plot.addLegend()

        self.mainLayout.addWidget(self.plot,0,0)
        self.make_sliders()

        self.startWavenum = 2000
        self.endWavenum = 5000
        self.wavenumStep = 0.04 #wavenums per simulation step
        self.mol = 'H2O' #'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO', 'SO2', 'NO2', 'NH3', 'HNO3'
        self.iso = 1 #isotopologue num
        cwd = os.path.dirname(os.path.realpath(__file__))
        if sys.platform == 'win32' or sys.platform == 'win64':
            self.dbdir = cwd + "\\DBDir\\"
        else:
            dbdir = cwd + "/DBDir/"
        if (not os.path.isdir(self.dbdir)):
            #need to make database directory
            os.mkdir(self.dbdir)

        self.minWvnEdit = FloatInput("min. Wavenum",self.startWavenum)
        self.minWvnEdit.set_callback(self.updateMinWvn)
        self.maxWvnEdit = FloatInput("max. Wavenum",self.endWavenum)
        self.maxWvnEdit.set_callback(self.updateMaxWvn)

        self.mainLayout.addWidget(self.minWvnEdit)
        self.mainLayout.addWidget(self.maxWvnEdit)

        self.absDB = gaas.gen_abs_db(self.mol,self.iso,self.startWavenum,self.endWavenum,self.dbdir,loadFromHITRAN=True)
        self.dbLock = threading.Lock()

        self.moleculeSelector = DropdownMenu(gaas.getHITRANMolecules())
        self.moleculeSelector.set_callback(self.setMolecule)
        self.mainLayout.addWidget(self.moleculeSelector)
        
        self.dbLoadingIcon = LoadingWidget()
        self.mainLayout.addWidget(self.dbLoadingIcon)
        self.numLinesLabel = QLabel()
        self.numLinesLabel.setText("Num Lines: " + str(len(self.absDB)))
        self.mainLayout.addWidget(self.numLinesLabel)
        self.dbLoadingIcon.hide()
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

    def updateMinWvn(self,value : float):
        print("AAAAAAAAAAAA")
        self.startWavenum = value
        self.reloadMoleculeDB()

    def updateMaxWvn(self,value : float):
        self.endWavenum = value
        self.reloadMoleculeDB()

    def reloadMoleculeDB(self):
        self.dbThread = threading.Thread(target=self._reloadDBThread)
        self.dbThread.start()

    def _reloadDBThread(self):
        self.dbLoadingIcon.show()
        # self.dbLoadingIcon.start()
        temp = gaas.gen_abs_db(self.mol,self.iso,self.startWavenum,self.endWavenum,self.dbdir,loadFromHITRAN=True)
        self.dbLock.acquire()
        self.absDB = temp
        self.numLinesLabel.setText("Num Lines: " + str(len(self.absDB)))
        self.dbLock.release()
        self.updateSim()
        # self.dbLoadingIcon.stop()
        self.dbLoadingIcon.hide()
    
    def setMolecule(self,molID):
        self.mol = molID
        self.reloadMoleculeDB()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    gallery = PlotWindow()
    gallery.show()
    sys.exit(app.exec())