import gym
from gym import spaces
import numpy as np
import pygame
import math


class ImprovedLTYEnv(gym.Env):
    def __init__(self, map='map3'):
        super(ImprovedLTYEnv, self).__init__()
        
        self.rend1 = True
        
        # Constants
        self.WIDTH = 1920
        self.HEIGHT = 1080

        # Car properties
        self.car_position_x = 980
        self.car_position_y = 970
        car_speed = 0
        car_angle = 0
        steering_angle = 0
        self.max_steering_angle = 20  # Maximum steering angle in degrees
        self.max_speed = 30
        self.min_speed = 5
        self.max_acc = 1
        
        # Color
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.yellow = (255, 255, 0)

        # Define the observation space: x position, y position, angle, speed, time, lidar sensors
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, -180.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), high=np.array([100.0, 100.0, 180.0, 10.0, 300.0, 300, 300.0, 300.0, 300.0]), dtype=np.float32)

        # Define action and observation spaces (steer, acc)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
    

        # Initialize agent's state with car-related properties
        self.agent_state = [self.car_position_x, self.car_position_y, 0, 0, 0, 0, 0, 0, 0]  # Car's initial position, angle, speed, lidar sensors 1 to 5

        # Car-related parameters
        self.car_size_x = 40
        self.car_size_y = 40
        self.car_speed = 0
        self.car_angle = 0
        self.car_speed_set = False
        self.car_centre = [self.agent_state[0] + self.car_size_x / 2, self.agent_state[1] + self.car_size_y / 2]
        self.car_corners = []  # Car's corner points

        # Initialize car's radars
        self.car_radars = []

        # Initialize Pygame window (optional for visualization)
        pygame.init()
        #self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.HIDDEN)
        pygame.display.set_caption('group35carracinggym-v0')
            
        # Load car sprite and rotate
        self.car_sprite = pygame.image.load('car.png').convert_alpha()
        self.car_sprite = pygame.transform.scale(self.car_sprite, (self.car_size_x, self.car_size_y))
        self.car_rotated_sprite = self.car_sprite
        
        # Load the race track image
        #track_map = pygame.image.load('map3.png').convert()
        track_map = pygame.image.load(map+'.png').convert()
        self.track_map = track_map
        
        # Episode-specific variables
        self.current_step = 0
        self.last_checkpoint_time_step = 0
        self.current_lap_time = 0
        self.best_lap_time = float('inf')
        self.previous_laptime = 0

        # Reward structure
        self.checkpoint_reward = 1
        self.time_penalty_factor = 0.5
        self.lap_completion_reward = 50000
        self.off_track_penalty = -5
        self.stationary_penalty = -1
        self.aggressive_input_penalty = -1

    def reset(self):
        # Reset the environment to an initial state
        self.current_step = 0
        self.agent_state = [self.car_position_x, self.car_position_y, 0, 0, 0, 0, 0, 0, 0]
        self.last_checkpoint_time_step = 0
        self.current_lap_time = 0
        self.car_speed = 0
        self.car_angle = 0
        self.car_speed_set = False
        self.car_centre = [self.agent_state[0] + self.car_size_x / 2, self.agent_state[1] + self.car_size_y / 2]
        self.car_corners = []
        self.car_radars = []
        self.dist = []

        return self.agent_state

    def step(self, action):
        # Implement the step function to update the environment and calculate rewards

        # Extract the agent's position (modify based on your state space)
        agent_x, agent_y, agent_orientation, agent_velocity, lidar1, lidar2, lidar3, lidar4, lidar5 = self.agent_state

        agent_velocity = agent_velocity + action[1]*self.max_acc

        # setting max and min velocity
        if agent_velocity > self.max_speed:
            agent_velocity = self.max_speed
        if agent_velocity < self.min_speed:
            agent_velocity = self.min_speed
            
        # setting max and min steering
        if action[0] > 1:
            action[0] = 1
        if action[0] < -1:
            action[0] = -1
        
        agent_orientation = agent_orientation + action[0] * np.pi *self.max_steering_angle / 180.0  # Adjust the scaling factor as needed
        agent_x = agent_x + agent_velocity * np.cos(agent_orientation)
        agent_y = agent_y + agent_velocity * np.sin(agent_orientation)
        self.car_position = [agent_x, agent_y]
        self.car_centre = [agent_x + self.car_size_x/2, agent_y + self.car_size_y/2]
        self.agent_orientation_deg = agent_orientation*180/np.pi

        # Update agent's state based on action (modify as needed)
        self.car_sensors()
        lidar1 = self.dist[0]
        lidar2 = self.dist[1]
        lidar3 = self.dist[2]
        lidar4 = self.dist[3]
        lidar5 = self.dist[4]

        # Calculate rewards based on your specified reward structure
        checkpoint_reached = False  # Implement a condition to check if checkpoint is reached
        off_track = False  # Implement a condition to check if agent is off track
        stationary = False  # Implement a condition to check if agent is stationary
        aggressive_input = False  # Implement a condition to check for aggressive input

        # Calculate lap time
        #self.current_lap_time = self.current_step - self.last_checkpoint_time_step

        # Calculate reward
        #reward = 0
        #if checkpoint_reached:
        #    reward += self.checkpoint_reward
        #    self.last_checkpoint_time_step = self.current_step
        #    if self.current_lap_time < self.best_lap_time:
        #        reward += self.lap_completion_reward  # Reward for completing a lap
        #        self.best_lap_time = self.current_lap_time
        #else:
        #    reward += self.time_penalty_factor * (self.current_step - self.last_checkpoint_time_step)

        #if off_track:
        #    reward += self.off_track_penalty

        #if stationary:
        #    reward += self.stationary_penalty

        #if aggressive_input:
        #    reward += self.aggressive_input_penalty

        #self.current_step += 1

        # Define termination condition (modify as needed)

        # Define a new reward structure

        # This measures the cumulative time in simulation
        self.current_step = self.current_step + 0.01

        reward = 0

        reward = 0.1 * self.current_step + 0.1 * agent_velocity
        # reward =  self.current_step

        if agent_velocity <= 0:
            reward = reward - 10

        # If the agent spends 50 time steps, i.e current_step == is a mutiple of 5, the agent recieves a reward of +50
        if self.current_step % 0.5 == 0:
            reward += 50

        done = False
        goal = False
        
        if (self.car_centre[0]<1919) and (self.car_centre[1]<1079) and (self.car_centre[0]>1) and (self.car_centre[1]>1) :
            color = self.track_map.get_at((int(self.car_centre[0]), int(self.car_centre[1])))
            if (color[0], color[1], color[2]) == (255, 255, 255):
                reward -= 100
                done = True
            
            # Define termination condtion of reaching goal (indicated by yellow colour)
            if (color[0], color[1], color[2]) == self.yellow:
                if self.current_step > self.previous_laptime:
                    reward = reward + 100
                    self.previous_laptime = self.current_step
                else:
                    reward = reward - 200
                    self.previous_laptime = self.current_step
                reward += 500
                print("We have reached goal.")
                done = True
                goal = True
        else:
            done = True
        
        # Update the agent's state (modify based on your state space)
        self.agent_state = [agent_x, agent_y, agent_orientation, agent_velocity, lidar1, lidar2, lidar3, lidar4, lidar5]

        return self.agent_state, reward, done, goal, {}

    def render(self, mode='human'):
        # Render the car and track (optional)
        if mode == 'human':
            rend = True
            if self.rend1 and rend:
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                self.rend1 = False
                
            
            self.screen.blit(self.track_map, (0, 0))
            self.car_rotated_sprite = self.rotate_center(self.car_sprite, -self.agent_state[2]*180/np.pi)
            self.screen.blit(self.car_rotated_sprite, self.agent_state[:2])
            
            #render centre point
            rec1 = pygame.Rect(0, 0, 10, 10)
            rec1.center = (int(self.car_centre[0]), int(self.car_centre[1]))
            pygame.draw.rect(self.screen, self.red, rec1)
            
            #render radar lines
            for j in self.car_radars:
                pygame.draw.line(self.screen, self.green, self.car_centre, j, width = 3)
                
            pygame.display.update()

    def rotate_center(self, image, angle):
        # Rotate the car image
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        
        return rotated_image

    def generate_random_track(self):
        # Implement a method to generate random waypoints that define the racetrack
        track_points = [(0.0, 0.0)]
        for _ in range(self.num_checkpoints):
            x = np.random.uniform(0.0, self.track_width)
            y = np.random.uniform(0.0, 100.0)  # Adjust the range as needed
            track_points.append((x, y))
        return track_points
    
    def check_radar(self, degree, game_map):
        length = 0
        radars = []
        x = int(self.car_centre[0] + math.cos(math.radians(360 - (-self.agent_orientation_deg + degree))) * length)
        y = int(self.car_centre[1] + math.sin(math.radians(360 - (-self.agent_orientation_deg + degree))) * length)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        if (x<1919) and (y<1079) and (x>1) and (y>1):
            while not game_map.get_at((x, y)) == (255, 255, 255, 255) and length < 300: 
                length = length + 1
                x = int(self.car_centre[0] + math.cos(math.radians(360 - (-self.agent_orientation_deg + degree))) * length)
                y = int(self.car_centre[1] + math.sin(math.radians(360 - (-self.agent_orientation_deg + degree))) * length)


        # Calculate Distance To Border
        dist = int(math.sqrt(math.pow(x - self.car_centre[0], 2) + math.pow(y - self.car_centre[1], 2)))
        
        return (x, y), dist
    
    def car_sensors(self):
        temp = []
        temp3 = []
        for i in range(-90, 100, 45):
            temp1, temp2 = self.check_radar(i, self.track_map)
            temp.append(temp1)
            temp3.append(temp2)
        
        self.car_radars = temp
        self.dist = temp3
        #print(self.dist)
        pass
        

    def close(self):
        # Implement any cleanup if necessary
        pygame.quit()

    # Register the custom environment with Gym
gym.envs.register(id='group35carracinggym-v0', entry_point=__name__ + ':ImprovedLTYEnv')

print("completed test gym with car")
