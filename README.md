# Hecatron
With Hecatron, we have developed a framework for training a deep reinforcement learning model to associate electroencephalography (EEG) signals with any number of disjoint actions. The original intention of this project was to to control a prosthetic hand purely by using an EEG brain-computer interface, but our work can be generalized to associate any EEG signal with a desired action.

# Set-Up
1. Clone git repo
2. Create virtual environment with ```python3 -m venv venv```
3. Activate virtual environment (mac: ```source ./venv/bin/activate```; windows: ```venv\Scripts\activate``` from command prompt)
4. Install needed requirements with ```pip3 install -r requirements.txt```
5. Either import via "import hecatron" or run the example with ```python3 example.py```

# Examples 
Examples for using the Hecatron functions with Muse, OpenBCI or the synthetic board can be found in 'example.py'.

# Documentation
## hecatron.find_port_and_id(serial_prefixes=SERIAL_PORT_PREFIXES, serial_suffixes=SERIAL_PORT_SUFFIXES, board_ids=BOARD_IDS, debug=False)
Searches through all combinations of serial ports and board IDs until a board is successfully connected, after which the board ID and serial port are returned. By default the serial ports for Windows and Linux/MacOS and board IDs for Muse and OpenBCI devices are used. It is suggested to use your own specific to your OS and device or the function may take a while to search through everything.
### Returns
A tuple of (board_id, serial_port) where board_id is an integer and serial_port is a string.
### Parameters
serial_prefixes (list[str]): A list of serial port prefixes to use (e.g. 'COM' for Windows, '/dev/ttyUSB' for Linux/MacOS)
serial_suffixes (list[str] or list[int]): A list of serial port suffixes to use (e.g. 'COM' for Windows, '/dev/ttyUSB' for Linux/MacOS)
board_ids (list[int]): A list of board IDs to use. Go to https://brainflow.readthedocs.io/en/stable/UserAPI.html to find the board ID for your EEG device.
debug (bool): Set to True if you want Brainflow debug outputs. 

run_training_session(...): 
Use up arrow key to give a reward of +1, down arrow key to give a reward of -1.

# Acknowledgements
This project is partly inspired by Khurram Javed and Abhishek Naik and their project in natHACKS 2023 (https://www.youtube.com/watch?v=NZ9JDahaU70&list=PL3jjdunHUqxzI6Hm05xex8BNJFxMtB2vV&index=3&t=361s)

The reinforcement learning algorithm we used is based on pseudocode from the paper "Learning and Planning in Average-Reward Markov Decision Processes" by Yi Wan, Abhishek Naik, and Richard S. Sutton (https://arxiv.org/abs/2006.16318)

