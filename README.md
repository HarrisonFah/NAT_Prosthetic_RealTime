# Hecatron
With Hecatron, we have developed a framework for training a deep reinforcement learning model to associate electroencephalography (EEG) signals with any number of disjoint actions. The original intention of this project was to to control a prosthetic hand purely by using an EEG brain-computer interface, but our work can be generalized to associate any EEG signal with a desired action.

# Set-Up
1. Clone git repo
2. Create virtual environment with ```python3 -m venv venv```
3. Activate virtual environment (mac: ```source ./venv/bin/activate```; windows: ```venv\Scripts\activate``` from command prompt)
4. Install needed requirements with ```pip3 install -r requirements.txt```
5. Run the app with ```python3 main.py```

# Examples 
Examples for using the Hecatron functions with Muse, OpenBCI or the synthetic board can be found in 'example.py'.

# Documentation

run_training_session(...): 
Use up arrow key to give a reward of +1, down arrow key to give a reward of -1.

# Acknowledgements
This project is partly inspired by Khurram Javed and Abhishek Naik and their project in natHACKS 2023 (https://www.youtube.com/watch?v=NZ9JDahaU70&list=PL3jjdunHUqxzI6Hm05xex8BNJFxMtB2vV&index=3&t=361s)

The reinforcement learning algorithm we used is based on pseudocode from the paper "Learning and Planning in Average-Reward Markov Decision Processes" by Yi Wan, Abhishek Naik, and Richard S. Sutton (https://arxiv.org/abs/2006.16318)

