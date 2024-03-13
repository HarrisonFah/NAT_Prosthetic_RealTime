from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from random import randint
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import math

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, board_shim):
        super().__init__()
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        #self.num_points = self.window_size * self.sampling_rate
        self.num_points = 250

        # Temperature vs time dynamic plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        self.plot_graph.setTitle("Live EEG Channel Data", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.setLabel("left", "Microvolts", **styles)
        self.plot_graph.setLabel("bottom", "Sample", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(-1500, 1500)

        voltage = [0 for _ in range(self.num_points)]
        # Get a line reference
        self.lines = []
        for channel in range(len(self.eeg_channels)):
            pen = pg.mkPen(color = colors[channel % len(colors)], width=3)
            line = self.plot_graph.plot(
                voltage,
                name="Channel #" + str(channel),
                pen=pen,
            )
            self.lines.append(line)
        # Add a timer to simulate new temperature measurements
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_speed_ms)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for channel in range(len(self.eeg_channels)):
            voltage = data[channel].tolist()
            self.lines[channel].setData(voltage)