import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from PyQt6 import QtCore, QtWidgets
from graph import MainGraphWindow
from rlplot import MainRLWindow
from model import QLearner
import random
import time
import torch

SYNTHETIC_BOARD_ID = -1 #Board ID for simulation board
SYNTHETIC_BOARD_PORT = 'COM0'
BOARD_IDS = [0, 1, 2, 3, 4, 5, 6, 21, 22, 38, 39, 41, 42] #Only includes OpenBCI and Muse board IDs
SERIAL_PORT_PREFIXES = ['COM', '/dev/ttyUSB']
SERIAL_PORT_SUFFIXES = list(range(0,11))

# Searches through all combinations of serial port and board ID until a valid board is found
def find_port_and_id(debug=False):
    for port_prefix in SERIAL_PORT_PREFIXES:
        for port_suffix in SERIAL_PORT_SUFFIXES:
            for board_id in BOARD_IDS:

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
def run_training_session(board, num_actions=2, num_samples=125, num_baseline_samples=50, update_speed_ms=50, window_size=4, num_points=250, reference_channels=[]):
    try:
        board.start_stream(450000)
        time.sleep(5)
        app = QtWidgets.QApplication([])
        main = MainRLWindow(board, num_actions, num_samples, num_baseline_samples, update_speed_ms, window_size, num_points, reference_channels)
        main.show()
        app.exec()
    except Exception as e:
        print(e)
    finally:
        board.stop_stream()

# Start the plot of the trained RL agent with the actions connected to functions
def run_live_session():
    pass