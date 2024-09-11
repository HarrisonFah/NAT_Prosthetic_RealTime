from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from random import randint
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import math
import torch
import numpy as np
from model import QLearner

class MainLiveRLWindow(QtWidgets.QMainWindow):
    
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

    # board_shim: The EEG board object that is read onto the plot
    # num_actions: The total number of discrete actions that the policy chooses from
    # action_functions: Lists of functions such that function with index i is called when action i is selected by the model
    # num_samples: The number of samples used in each state
    # num_baseline_samples: The number of samples used to calculate the baseline mean that is subtracted from the state samples
    # update_speed_ms: How often the plot is updated in milliseconds
    # window_size: The width of the plot window
    # num_points: How many EEG samples are shownn on the plot at once
    # reference_channels: Channels to ignore when calculating the state
    # one_hot_value: Value of selected action in one hot vector
    # filename: Prefix of load file name
    def __init__(self, board_shim, action_functions, filename, num_actions=2, num_samples=125, num_baseline_samples=50, update_speed_ms=50, window_size=4, num_points=250, reference_channels=[], one_hot_value=1e3):
        super().__init__()
        #Sets all paramaters
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = update_speed_ms
        self.window_size = window_size
        self.num_points = num_points
        self.reference_channels = reference_channels
        self.num_samples = num_samples
        self.num_baseline_samples = num_baseline_samples
        self.num_actions = num_actions
        self.action_functions = action_functions
        assert len(action_functions) >= num_actions, "Length of action functions is shorter than number of actions."
        #Initializes FFT values
        self.fft_min = torch.full((len(self.eeg_channels)-len(self.reference_channels), 1), float("Inf")) #Initializes the rolling minimum for normalizing FFT
        self.fft_max = torch.full((len(self.eeg_channels)-len(self.reference_channels), 1), float("-Inf")) #Initializes the rolling maximum for normalizing FFT
        num_fft_features = torch.flatten(torch.fft.fft(torch.zeros((len(self.eeg_channels)-len(self.reference_channels), self.num_samples)))).shape[0] #Calculates the total number of values in an FFT (the state features)
        self.learner = QLearner(self.num_actions, num_fft_features, one_hot_value=one_hot_value, filename=filename) #Initializes reinforcement learning model
        assert filename is not None, "filename is None."
        self.learner.load(filename)

        #Sets up the plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        self.plot_graph.setTitle("Using RL Agent", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(-num_actions, num_actions)

        #Sets up line for plotting predictions for each action
        self.all_predictions = []
        for action in range(self.num_actions):
            predictions = [0 for _ in range(self.num_points)]
            predict_pen = pg.mkPen(color = ['b', 'c', 'm'][action], width=3)
            predict_line = self.plot_graph.plot(
                predictions,
                name=f"Action {action} Prediction",
                pen=predict_pen,
            )
            self.all_predictions.append((predictions, predict_line))

        #Sets up line for plotting selected actions
        self.actions = [0 for _ in range(self.num_points)]
        action_pen = pg.mkPen(color = 'g', width=3)
        self.action_line = self.plot_graph.plot(
            self.actions,
            name="Action",
            pen=action_pen,
        )

        #Updates the plot with the latest num_points samples every update_speed_ms
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_speed_ms)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        data = self.board_shim.get_current_board_data(self.num_baseline_samples + self.num_samples) #Gets the latest num_baseline_samples + num_samples from board
        baseline_data = data[:,:self.num_baseline_samples] #Isolates the baseline data
        state_data = (data[:,self.num_baseline_samples:].transpose() - np.mean(baseline_data, axis=1)).transpose() #Subtracts mean of each channel's baseline samples from all samples in each channel
        eeg_data = torch.tensor(state_data[[channel for channel_idx, channel in enumerate(self.eeg_channels) if channel_idx not in self.reference_channels]]) #Gets the data from all eeg channels except the reference channels
        #Gets the magnitude of FFT and normalizes it using rolling min/max
        fft_eeg_data = torch.fft.fft(eeg_data)
        fft_magnitude = torch.abs(fft_eeg_data)
        self.fft_min = torch.minimum(self.fft_min, torch.min(fft_magnitude, 1)[0].unsqueeze(1))
        self.fft_max = torch.maximum(self.fft_max, torch.max(fft_magnitude, 1)[0].unsqueeze(1))
        fft_magnitude /= self.num_samples
        flat_fft_mag = torch.flatten(fft_magnitude)
        #Performs training step and gets the selected action and predicted rewards from the RL model
        selected_action, predicted_rewards = self.learner.get_action(flat_fft_mag)
        print("Predicted rewards:", [reward.item() for reward in predicted_rewards])
        try:
            self.action_functions[selected_action]()
        except Exception as e:
            print(e)
            print("Could not call function for action:", selected_action)

        #Updates the predictions for each action on the plot
        for action in range(self.num_actions):
            predictions, predict_line = self.all_predictions[action]
            predictions.pop(0)
            predictions.append(predicted_rewards[action].item())
            predict_line.setData(predictions)

        #Updates the selected action on the plot
        self.actions.pop(0)
        self.actions.append(selected_action)
        self.action_line.setData(self.actions)