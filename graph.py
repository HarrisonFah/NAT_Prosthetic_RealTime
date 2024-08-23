from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from random import randint
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import math

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']

class MainGraphWindow(QtWidgets.QMainWindow):
    
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

    # board_shim: The EEG board object that is read onto the plot
    # update_speed_ms: How often the plot is updated in milliseconds
    # window_size: The width of the plot window
    # num_points: How many EEG samples are shownn on the plot at once
    def __init__(self, board_shim, update_speed_ms=50, window_size=4, num_points=250):
        super().__init__()
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = update_speed_ms
        self.window_size = window_size
        self.num_points = num_points

        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        self.plot_graph.setTitle("Live EEG Channel Data", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.setLabel("left", "Microvolts", **styles)
        self.plot_graph.setLabel("bottom", "Sample", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(-5, 5)

        voltage = [0 for _ in range(self.num_points)]
        self.lines = []
        for channel in range(len(self.eeg_channels)):
            pen = pg.mkPen(color = colors[channel % len(colors)], width=3)
            line = self.plot_graph.plot(
                voltage,
                name="Channel #" + str(channel),
                pen=pen,
            )
            self.lines.append(line)

        #Set up a timer to update the plot every update_speed_ms
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_speed_ms)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    #Updates the plot with the latest num_points samples every update_speed_ms
    def update_plot(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        data /= 1000
        for channel in range(len(self.eeg_channels)):
            voltage = data[channel].tolist()
            self.lines[channel].setData(voltage)