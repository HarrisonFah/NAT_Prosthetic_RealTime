import hecatron

# This example is set up to work with the Muse 2 headset
SERIAL_PORT = 'COM4' 
BOARD_ID = 38

board = hecatron.init_board(SERIAL_PORT, BOARD_ID, debug=True)

hecatron.start_eeg_plot(board)

hecatron.run_training_session(board, num_actions=2, reference_channels=[0])

#hecatron.run_live_session(board)