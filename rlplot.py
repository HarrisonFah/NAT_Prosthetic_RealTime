from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from random import randint
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import math
import torch
import numpy as np
from model import QLearner

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']
actions = {0:0, 1:1, 2:-1} #translates action as represented in rl model to -1, 0, +1

class MainRLWindow(QtWidgets.QMainWindow):
    
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

    def __init__(self, board_shim, num_samples=125, num_baseline_samples = 50):
        super().__init__()
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 100
        self.window_size = 4
        self.num_points = 100
        self.num_samples = num_samples
        self.num_baseline_samples = num_baseline_samples
        #self.learner = QLearner(3, self.num_samples*len(self.eeg_channels))
        num_fft_features = torch.flatten(torch.fft.fft(torch.zeros((len(self.eeg_channels)-1, self.num_samples)))).shape[0]
        self.num_actions = 2
        self.learner = QLearner(self.num_actions, num_fft_features)
        self.queued_reward = 0
        #moving max and min values for fft magnitudes
        self.fft_min = torch.full((len(self.eeg_channels)-1, 1), float("Inf"))
        self.fft_max = torch.full((len(self.eeg_channels)-1, 1), float("-Inf"))

        # Temperature vs time dynamic plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        self.plot_graph.setTitle("RL Agent Stats", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(-2, 2)

        self.all_predictions = []
        for action in range(self.num_actions):
            predictions = [0 for _ in range(self.num_points)]
            # Get a line reference
            predict_pen = pg.mkPen(color = ['b', 'c', 'm'][action], width=3)
            predict_line = self.plot_graph.plot(
                predictions,
                name=f"Action {action} Prediction",
                pen=predict_pen,
            )
            self.all_predictions.append((predictions, predict_line))
    
        self.rewards = [0 for _ in range(self.num_points)]
        reward_pen = pg.mkPen(color = 'r', width=3)
        self.reward_line = self.plot_graph.plot(
            self.rewards,
            name="Reward",
            pen=reward_pen,
        )
        self.actions = [0 for _ in range(self.num_points)]
        action_pen = pg.mkPen(color = 'g', width=3)
        self.action_line = self.plot_graph.plot(
            self.actions,
            name="Action",
            pen=action_pen,
        )
        # Add a timer to simulate new temperature measurements
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_speed_ms)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        data = self.board_shim.get_current_board_data(self.num_baseline_samples + self.num_samples)
        baseline_data = data[:,:self.num_baseline_samples]
        #subtracts mean of each channel from all samples in each channel
        state_data = (data[:,self.num_baseline_samples:].transpose() - np.mean(baseline_data, axis=1)).transpose()
        eeg_data = torch.tensor(state_data[self.eeg_channels[1:]]) #gets the data from all eeg channels except the first one (reference channel)
        #https://dsp.stackexchange.com/questions/25456/conversion-of-fft-to-psd#:~:text=To%20get%20the%20PSD%20from,spacing%20on%20your%20x%20axis.&text=If%20you%20want%20to%20check,variance%20of%20the%20original%20signal.
        #https://www.mathworks.com/help/signal/ug/practical-introduction-to-frequency-domain-analysis.html
        fft_eeg_data = torch.fft.fft(eeg_data)
        fft_magnitude = torch.abs(fft_eeg_data)
        #psd_data = torch.abs(fft_eeg_data)**2
        self.fft_min = torch.minimum(self.fft_min, torch.min(fft_magnitude, 1)[0].unsqueeze(1))
        print(self.fft_min)
        self.fft_max = torch.maximum(self.fft_max, torch.max(fft_magnitude, 1)[0].unsqueeze(1))
        print(self.fft_max)
        #fft_magnitude = (fft_magnitude - self.fft_min.expand(-1, fft_magnitude.shape[1])) / self.fft_max.expand(-1, fft_magnitude.shape[1])
        fft_magnitude /= self.num_baseline_samples + self.num_samples
        flat_fft_mag = torch.flatten(fft_magnitude)
        selected_action, predicted_rewards = self.learner.step(flat_fft_mag, self.queued_reward)
        print("Selected action:", actions[selected_action], ", predicted rewards:", predicted_rewards)

        for action in range(self.num_actions):
            predictions, predict_line = self.all_predictions[action]
            predictions.pop(0)
            predictions.append(predicted_rewards[action].item())
            predict_line.setData(predictions)

        self.actions.pop(0)
        self.actions.append(actions[selected_action])
        self.action_line.setData(self.actions)

        self.rewards.pop(0)
        self.rewards.append(self.queued_reward)
        self.reward_line.setData(self.rewards)
        self.queued_reward = 0

    def keyPressEvent(self, event):
        super(MainRLWindow, self).keyPressEvent(event)
        if event.key() == 16777235:
            self.queued_reward = 1
        elif event.key() == 16777237:
            self.queued_reward = -1
        print('pressed from MainWindow: ', event.key())
        self.keyPressed.emit(event)