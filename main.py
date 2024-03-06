import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from PyQt6 import QtCore, QtWidgets
from graph import MainWindow

BOARD_ID = 2 #board id for cyton daisy
SYNTHETIC_BOARD_ID = -1 #board id for simulation board
SERIAL_PORT = 'COM5' #com6 for simulated board maybe?
debug = False

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

def run_session(board):
	try:
		board.start_stream(450000)
		app = QtWidgets.QApplication([])
		main = MainWindow(board)
		main.show()
		app.exec()
	except BaseException as e:
		print(e)
		return
	finally:
		print("Ending session...")
		board.release_session()
		return

	for count, stim in enumerate(stimuli):
		print("\tSample " + str(count) + "/" + str(len(stimuli)))
		print("\t\t" + class_actions[stim-1][0])
		board.insert_marker(stim)
		time.sleep(SAMPLE_SECONDS)
		print("\t\t" + class_actions[stim-1][1])
		#wait SAMPLE_SECONDS for user to perform action
		time.sleep(SAMPLE_SECONDS)

	board.stop_stream()
	data = board.get_data()
	eeg_channels = board.get_eeg_channels()
	sfreq = board.get_sfreq()
	ch_names = board.get_eeg_names()
	event_channel = board.get_event_channel()
	board.release_session()
	TestPlotter(data, eeg_channels, sfreq, ch_names, stimuli, event_channel)

	return data

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
	run_session(board)

main()