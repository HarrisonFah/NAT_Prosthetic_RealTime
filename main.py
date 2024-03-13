import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from PyQt6 import QtCore, QtWidgets
from graph import MainWindow
from model import QLearner
import random
import time
import torch

BOARD_ID = 39 #board id for cyton daisy
SYNTHETIC_BOARD_ID = -1 #board id for simulation board
SERIAL_PORT = 'COM4' #com6 for simulated board maybe?
debug = False
NUM_POINTS = 125

def init_board(serialPort, boardID, debug):
	params = BrainFlowInputParams()
	params.ip_port = 0
	params.serial_port = serialPort
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
	board = BoardShim(boardID, params)

	return board

def prep_session(board):
	try:
		board.prepare_session()
		return True
	except Exception as e:
		print("\nError: " + str(e))
		print("This issue can be caused by the following reasons:")
		print("\t-Board is not connected to computer.")
		print("\t-Board is being used by another program.")
		return False

def setup_graph(board):
	try:
		board.start_stream(450000)
		app = QtWidgets.QApplication([])
		main = MainWindow(board)
		main.show()
		app.exec()
	except BaseException as e:
		print(e)
	finally:
		print("Ending session...")
		board.stop_stream()

def run_session(board):
	#try:
	eeg_channels = board.get_eeg_channels(BOARD_ID)
	learner = QLearner(3, NUM_POINTS*len(eeg_channels))
	board.start_stream(450000)
	print("Warming up...")
	time.sleep(5)
	print("Started...")
	total_time = 0
	while total_time < 30:
		time.sleep(0.1)
		data = board.get_current_board_data(NUM_POINTS)
		eeg_data = data[eeg_channels]
		flat_eeg_data = torch.flatten(torch.tensor(eeg_data)).double()
		reward = random.choice([-1, 0, 1])
		selected_action, predicted_reward = learner.step(flat_eeg_data, reward)
		print("Selected action:", selected_action, ", predicted reward:", predicted_reward)
		total_time += 0.1
	# except Exception as e:
	# 	print(e)
	# finally:
	# 	board.stop_stream()
	# 	board.release_session()

def main():
	board = init_board(SERIAL_PORT, BOARD_ID, debug)
	if not prep_session(board):
		response = ""
		while response.lower() not in ["y", "n"]:
			response = input("Did you want to use a simulated board? (y/n): ")
		if response == "y":
			board = init_board(SERIAL_PORT, SYNTHETIC_BOARD_ID, debug)
			prep_session(board)
		else:
			return
	setup_graph(board)
	run_session(board)

main()