import hecatron

# This example is set up to work with the synthetic board
SERIAL_PORT = 'COM0' 
BOARD_ID = -1

board = hecatron.init_board(SERIAL_PORT, BOARD_ID, debug=True)

hecatron.start_eeg_plot(board)

hecatron.run_training_session(board, num_actions=2)

#hecatron.run_live_session(board)