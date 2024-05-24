# ME5418 Machine Learning in Robotics Project

![OptimizedRacingCar-ezgif com-video-to-gif-converter](https://github.com/KAN201197/Machine-Learning-Robotics/assets/128454220/288b988e-8dda-4484-aa9a-657e6f120709)

This project aims to develop the deep reinforcement learning algorithm which will be implemented into an **Autonomous Racing Car**. The objective of this project is for the agent to reach a desired goal position while minimizing time lap as well as preventing the agent from moving outside of the track. The reinforcement learning algorithm that used in this project is **PPO (Proximal Policy Optimization)**. In this project, there is three different part that should be integrated as **gym environment, the neural network model, and the learning agent**. 

Inside the gym environment, the observation space includes the position of the car, heading direction, the car’s velocity, and lidar sensor data. The action space consists of continuous action in steering angle and acceleration ranging from -1 to 1. The reward structure is defined based on various factors such as the agent’s velocity (velocity-based reward), time spent (time-based reward), velocity penalty, lap completion reward, and off-track penalty. 

The neural Network Model is defined as a neural network with a single stream input network for both of actor and critic network which will be used later to train the agent. The last one in the learning agent defines all computations related to the gym environment and neural network such as computation of action and value as well as loss computation to update the weight on the neural network.

## List of Executable Files

In this repo, it contain all of following files and folder showing below:

1. PPO_Agent.py
2. Gym_new.py
3. ACNet.py
4. Utilities.py
5. trainer.ipynb
6. requirements.txt
7. car.png
8. map1a.png
9. map3.png
10. models
11. new_models

## Instruction to Run the Code

You may begin by creating a conda environment according to the requirements listed in “requirements.txt” file inside of Codes folder.

"PPO_Agent.py" contains the code for the agent. The agent interacts with environment in "Gym_new.py", gets the observation and passes it to Neural-Network in "ACNet.py" to generate actions. The "Utilities.py" consist of helper function to help trained the agent. 

The "trainer.ipynb" is used to trained the agent and using trained model to test the agent in different maps. If you want to run this code to train the agent, you can run it in the cell 3 (three) and change the argument in Trainer class for train to "True" and rewrite map to "map1a" or "map3". If you want to test the trained model in this code, you can run it in the cell 3 (three) and change the argument in Trainer class for train to "False" and rewrite map to "map1a" or "map3". The new trained model will be saved inside "new_models" folder. The "models" folder is the best saved trained model that we have.

## List of Required Dependencies and Libraries

Listed below are the list of required dependencies and libraries required to be installed prior to running the code for Neural Network and PPO_Agent code.

  	1. Numpy (pip install numpy==1.22.1)
  	2. TensorFlow  (pip install tensorflow==2.13.1)
  	3. tensorflow-probability  (pip install tensorflow-probability)

## Additional Note

You only need to run the code in the "trainer.ipynb". In this code, no need to change any parameters inside of other cells except cell 3 (three)
