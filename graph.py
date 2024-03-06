# from PyQt5 import QtWidgets  
# import pyqtgraph as pg
# from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
# from pyqtgraph.Qt import QtGui, QtCore

# class Graph:
#     def __init__(self, board_shim):
#         self.board_id = board_shim.get_board_id()
#         self.board_shim = board_shim
#         self.exg_channels = BoardShim.get_exg_channels(self.board_id)
#         self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
#         self.update_speed_ms = 50
#         self.window_size = 4
#         self.num_points = self.window_size * self.sampling_rate

#         self.app = QtWidgets.QApplication([])
#         self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot', size=(800, 600))

#         self._init_timeseries()

#         timer = QtCore.QTimer()
#         timer.timeout.connect(self.update)
#         timer.start(self.update_speed_ms)
#         QtWidgets.QApplication.instance().exec_()

#     def _init_timeseries(self):
#         self.plots = list()
#         self.curves = list()
#         for i in range(len(self.exg_channels)):
#             p = self.win.addPlot(row=i, col=0)
#             p.showAxis('left', False)
#             p.setMenuEnabled('left', False)
#             p.showAxis('bottom', False)
#             p.setMenuEnabled('bottom', False)
#             if i == 0:
#                 p.setTitle('TimeSeries Plot')
#             self.plots.append(p)
#             curve = p.plot()
#             self.curves.append(curve)

#     def update(self):
#         data = self.board_shim.get_current_board_data(self.num_points)
#         for count, channel in enumerate(self.exg_channels):
#             # plot timeseries
#             DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
#             DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
#                                         FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
#             DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
#                                         FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
#             DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
#                                         FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
#             self.curves[count].setData(data[channel].tolist())

#         self.app.processEvents()

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from random import randint
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, board_shim):
        super().__init__()
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        # Temperature vs time dynamic plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        pen = pg.mkPen(color=(255, 0, 0))
        self.plot_graph.setTitle("Temperature vs Time", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.setLabel("left", "Temperature (°C)", **styles)
        self.plot_graph.setLabel("bottom", "Time (min)", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(20, 40)
        self.time = list(range(self.num_points))
        self.temperature = [0 for _ in range(self.num_points)]
        # Get a line reference
        self.line = self.plot_graph.plot(
            self.time,
            self.temperature,
            name="Temperature Sensor",
            pen=pen,
            symbol="+",
            symbolSize=15,
            symbolBrush="b",
        )
        # Add a timer to simulate new temperature measurements
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_speed_ms)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        # self.time = self.time[1:]
        # self.time.append(self.time[-1] + 1)
        # self.temperature = self.temperature[1:]
        # self.temperature.append(randint(20, 40))
        data = self.board_shim.get_current_board_data(self.num_points)
        self.temperature = data[0].tolist()
        self.time = list(range(len(data[0].tolist())))
        self.line.setData(self.time, self.temperature)

    # def update_plot(self):
    #     data = self.board_shim.get_current_board_data(self.num_points)
    #     for count, channel in enumerate(self.eeg_channels):
    #         print(data[channel].tolist())
    #         # plot timeseries
    #         # DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
    #         # DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
    #         #                             1, 0)
    #         # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
    #         #                             1, 0)
    #         # DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
    #         #                             1, 0)
    #         self.curves[count].setData(data[channel].tolist())

    #     #self.app.processEvents()
    #     # self.time = self.time[1:]
    #     # self.time.append(self.time[-1] + 1)
    #     # self.temperature = self.temperature[1:]
    #     # self.temperature.append(randint(20, 40))
    #     # self.line.setData(self.time, self.temperature)