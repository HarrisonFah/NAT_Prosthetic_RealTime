import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from PyQt6 import QtCore, QtWidgets
from graph import MainGraphWindow
from rlplot import MainTrainRLWindow
from rlplot_live import MainLiveRLWindow
from model import QLearner
import random
import time
import torch

SYNTHETIC_BOARD_ID = -1 #Board ID for simulation board
SYNTHETIC_BOARD_PORT = 'COM0'
BOARD_IDS = [0, 1, 2, 3, 4, 5, 6, 21, 22, 38, 39, 41, 42] #Only includes OpenBCI and Muse board IDs
SERIAL_PORT_PREFIXES = ['COM', '/dev/ttyUSB']
SERIAL_PORT_SUFFIXES = list(range(0,11))

def _action_0_default():
    print("Selected action 0.")

def _action_1_default():
    print("Selected action 1.")

# Searches through all combinations of serial port and board ID until a valid board is found
#
# serial_prefixes: List of all serial port prefixes to scan through
# serial_suffixes: List of all serial port suffixes to scan through
# board_ids: List of all board IDs to scan through
# debug: Enable debug mode
def find_port_and_id(serial_prefixes=SERIAL_PORT_PREFIXES, serial_suffixes=SERIAL_PORT_SUFFIXES, board_ids=BOARD_IDS, debug=False):
    for port_prefix in serial_prefixes:
        for port_suffix in serial_suffixes:
            for board_id in board_ids:

                print(f"Testing: SERIAL_PORT:{port_prefix + str(port_suffix)}, BOARD_ID={board_id}")
        
                serial_port = port_prefix + str(port_suffix)

                params = BrainFlowInputParams()
                params.ip_port = 0
                params.serial_port = serial_port
                params.mac_address = ''
                params.other_info = ''
                params.serial_number = ''
                params.ip_address = ''
                params.ip_protocol = 0
                params.timeout = 0
                params.file = ''

                if (debug):
                    BoardShim.enable_dev_board_logger()
                else:
                    BoardShim.disable_board_logger()
                board = BoardShim(board_id, params)

                try:
                    board.prepare_session()
                    board.release_session()
                    return board_id, serial_port
                except Exception as e:
                    pass

    return None, None

# Creates the board object and prepares a session (verifying that it connects properly)
#
# serial_port: Serial port of EEG device
# board_id: Board ID of EEG device
# debug: Enable debug mode
def init_board(serial_port=None, board_id=None, debug=False):
    if serial_port is None:
        serial_port = SYNTHETIC_BOARD_PORT
    if board_id is None:
        board_id = SYNTHETIC_BOARD_ID
        print("Warning: Using synethtic board due to board_id=None.")
    params = BrainFlowInputParams()
    params.ip_port = 0
    params.serial_port = serial_port
    params.mac_address = ''
    params.other_info = ''
    params.serial_number = ''
    params.ip_address = ''
    params.ip_protocol = 0
    params.timeout = 0
    params.file = ''

    if (debug):
        BoardShim.enable_dev_board_logger()
    else:
        BoardShim.disable_board_logger()

    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
    except Exception as e:
        print("\nError: " + str(e))
        print("This issue can be caused by the following reasons:")
        print("\t-Board is not connected to computer.")
        print("\t-Board is being used by another program.")
        raise(e)

    return board

# Starts the plot of the EEG signals without any RL model
#
# board: The EEG board object that is read onto the plot
# update_speed_ms: How often the plot is updated in milliseconds
# window_size: The width of the plot window
# num_points: How many EEG samples are shownn on the plot at once
def start_eeg_plot(board, update_speed_ms=50, window_size=4, num_points=250):
    try:
        board.start_stream(450000)
        app = QtWidgets.QApplication([])
        main = MainGraphWindow(board, update_speed_ms, window_size, num_points)
        main.show()
        app.exec()
    except BaseException as e:
        print(e)
    finally:
        board.stop_stream()

# Starts the training plot of the RL agent and trains it until the window is closed
#
# board: The EEG board object that is read onto the plot
# num_actions: The total number of discrete actions that the policy chooses from
# num_samples: The number of samples used in each state
# num_baseline_samples: The number of samples used to calculate the baseline mean that is subtracted from the state samples
# update_speed_ms: How often the plot is updated in milliseconds
# window_size: The width of the plot window
# num_points: How many EEG samples are shownn on the plot at once
# reference_channels: Channels to ignore when calculating the state
# epsilon: Value of epsilon used in an epsilon-greedy policy (takes a random action with probability of epsilon)
# alpha: Learning rate of model
# eta: How much the average reward is updated by each step
# one_hot_value: Value of selected action in one hot vector
# filename: Prefix of save file name
# save_freq: Saves the model every save_freq timesteps
def run_training_session(board, num_actions=2, num_samples=125, num_baseline_samples=50, update_speed_ms=50, window_size=4, num_points=250, reference_channels=[], epsilon=5e-2, alpha=1e-4, eta=1e-2, one_hot_value=1e3, filename="default_model", save_freq=100):
    try:
        board.start_stream(450000)
        time.sleep(5)
        app = QtWidgets.QApplication([])
        main = MainTrainRLWindow(board, num_actions, num_samples, num_baseline_samples, update_speed_ms, window_size, num_points, reference_channels, epsilon, alpha, eta, one_hot_value, filename, save_freq)
        main.show()
        app.exec()
    except Exception as e:
        print(e)
    finally:
        board.stop_stream()

# Start the plot of the trained RL agent with the actions connected to functions
#
# board: The EEG board object that is read onto the plot
# action_functions: Lists of functions such that function with index i is called when action i is selected by the model
# filename: Prefix of save file name
# num_actions: The total number of discrete actions that the policy chooses from
# num_samples: The number of samples used in each state
# num_baseline_samples: The number of samples used to calculate the baseline mean that is subtracted from the state samples
# update_speed_ms: How often the plot is updated in milliseconds
# window_size: The width of the plot window
# num_points: How many EEG samples are shownn on the plot at once
# reference_channels: Channels to ignore when calculating the state
# one_hot_value: Value of selected action in one hot vector
def run_live_session(board, action_functions=[_action_0_default, _action_1_default], filename="default_model", num_actions=2, num_samples=125, num_baseline_samples=50, update_speed_ms=50, window_size=4, num_points=250, reference_channels=[], one_hot_value=1e3):
    try:
        board.start_stream(450000)
        time.sleep(5)
        app = QtWidgets.QApplication([])
        main = MainLiveRLWindow(board, action_functions, filename, num_actions, num_samples, num_baseline_samples, update_speed_ms, window_size, num_points, reference_channels, one_hot_value)
        main.show()
        app.exec()
    except Exception as e:
        print(e)
    finally:
        board.stop_stream()