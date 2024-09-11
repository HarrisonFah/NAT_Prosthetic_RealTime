import hecatron

# This example first tries to find an OpenBCI or Muse headset to use.
# If it cannot find one, it uses a synthetic board.

response = input("Do you want to use a connected board? (y/n)")
if response.lower() == 'y':
    response = input("Search for board? (y/n)")
    if response.lower() == 'y':
        board_id, serial_port = hecatron.find_port_and_id()
    else:
        serial_port = input("Serial Port: ")
        board_id = int(input("Board ID: "))
else:
    board_id = None
    serial_port = None
print("board_id:", board_id)
print("serial_port:", serial_port)

board = hecatron.init_board(serial_port, board_id)

hecatron.start_eeg_plot(board)

hecatron.run_training_session(board, num_actions=2, reference_channels=[0], filename="example_model")

hecatron.run_live_session(board, num_actions=2, reference_channels=[0], filename="example_model")