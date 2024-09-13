# Hecatron
With Hecatron, we have developed a framework for training a deep reinforcement learning model to associate electroencephalography (EEG) signals with any number of disjoint actions. The original intention of this project was to to control a prosthetic hand purely by using an EEG brain-computer interface, but our work can be generalized to associate any EEG signal with a desired action.

# Set-Up
1. Clone git repo
2. Create virtual environment with ```python3 -m venv venv```
3. Activate virtual environment (mac: ```source ./venv/bin/activate```; windows: ```venv\Scripts\activate``` from command prompt)
4. Install needed requirements with ```pip3 install -r requirements.txt```
5. Either import via ```import hecatron``` or run the example with ```python3 example.py```

# Examples 
Examples for using the Hecatron functions with Muse, OpenBCI or the synthetic board can be found in 'example.py'.

# Documentation
## hecatron.find_port_and_id(serial_prefixes=SERIAL_PORT_PREFIXES, serial_suffixes=SERIAL_PORT_SUFFIXES, board_ids=BOARD_IDS, debug=False)
Searches through all combinations of serial ports and board IDs until a board is successfully connected, after which the board ID and serial port are returned. By default the serial ports for Windows and Linux/MacOS and board IDs for Muse and OpenBCI devices are used. It is suggested to use your own specific to your OS and device or the function may take a while to search through everything.

### Returns
A tuple of (board_id, serial_port) where board_id is an integer and serial_port is a string.

### Parameters
**serial_prefixes** (list[str]): A list of serial port prefixes to use (e.g. 'COM' for Windows, '/dev/ttyUSB' for Linux/MacOS)

**serial_suffixes** (list[str] or list[int]): A list of serial port suffixes to use (e.g. 'COM' for Windows, '/dev/ttyUSB' for Linux/MacOS)

**board_ids** (list[int]): A list of board IDs to use. Go to https://brainflow.readthedocs.io/en/stable/UserAPI.html to find the board ID for your EEG device.

**debug** (bool): Set to True if you want Brainflow debug outputs. 

## hecatron.init_board(serial_port=None, board_id=None, debug=False)
Initializes a brainflow.BoardShim object with serial_port and board_id, verifies that it connects properly, and then returns the board object.

### Returns
A brainflow.BoardShim object 

### Parameters
**serial_port** (str): The serial port the EEG device is connected to. If set to None then the synthetic board serial_port will be used.

**board_id** (int): The board ID of the EEG device. This is found either through the hecatron.find_port_and_id(...) function or found on https://brainflow.readthedocs.io/en/stable/UserAPI.html. If set to None the synthetic board board_id will be used.

**debug** (bool): Set to True if you want Brainflow debug outputs. 

## hecatron.start_eeg_plot(board, update_speed_ms=50, window_size=4, num_points=250)
Starts a live plot of the EEG channels sent from the board object.

### Returns 
None

### Parameters
**board** (brainflow.BoardShim object): The EEG device being read from.

**update_speed_ms** (int): How often the plot is updated, in milliseconds.

**window_size** (int): The width of the plot window.

**num_points** (int): The number of EEG samples to plot at each timepoint.

## hecatron.run_training_session(board, num_actions=2, num_samples=125, num_baseline_samples=50, update_speed_ms=50, window_size=4, num_points=250, reference_channels=[], epsilon=5e-2, alpha=1e-4, eta=1e-2, one_hot_value=1e3, filename="default_model", save_freq=100)
Starts training the reinforcement learning model while displaying a live plot with the model predictions and selected actions as well as the rewards given by the user (up arrow key for +1 reward, down arrow key for -1 reward). By default the model is saved every 100 training steps as "default_model_network" and "default_model_optimizer".

### Returns
None

### Parameters
**board** (brainflow.BoardShim object): The EEG device being read from.

**num_actions** (int): The number of actions that the RL model can select from.

**num_samples** (int): The number of samples used to describe a state at each timestep.

**num_baseline_samples** (int): The number of samples before the state sample that are averaged and subtracted from it.

**update_speed_ms** (int): How often the plot is updated, in milliseconds.

**window_size** (int): The width of the plot window.

**num_points** (int): The number of EEG samples to plot at each timepoint.

**reference_channels** (list[int]): A list of the indices of EEG channels to ignore (should be used for reference channels). The index should be of the EEG channels and not the total channels (e.g. if we want to ignore the first EEG channel which brainflow has at channel index 16, we would just pass in an index of 0.)

**epsilon*** (float): Value of epsilon used in an epsilon-greedy policy (takes a random action with probability of epsilon).

**alpha** (float): Learning rate of model. Rate at which the model weights are updated in each training step.

**eta** (float): Rate at which the average reward is updated in each training step.

**one_hot_value** (float): Value of selected action in one_hot_vector

**filename** (str): Prefix of save file name.

**save_freq** (int): Saves the model every save_freq timesteps.

# Acknowledgements
This project is partly inspired by Khurram Javed and Abhishek Naik and their project in natHACKS 2023 (https://www.youtube.com/watch?v=NZ9JDahaU70&list=PL3jjdunHUqxzI6Hm05xex8BNJFxMtB2vV&index=3&t=361s)

The reinforcement learning algorithm we used is based on pseudocode from the paper "Learning and Planning in Average-Reward Markov Decision Processes" by Yi Wan, Abhishek Naik, and Richard S. Sutton (https://arxiv.org/abs/2006.16318)

