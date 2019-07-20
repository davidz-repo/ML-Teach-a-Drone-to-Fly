
import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        if init_pose is not None:
            self.distance = self.get_euclidean_distance(init_pose, self.target_pos)
        else:
            self.distance = 0

    def get_euclidean_distance(self, start, finish):
        """Return the euclidean distance between two points in (x, y, z) space"""
        x = start[0] - finish[0]
        y = start[1] - finish[1]
        z = start[2] - finish[2]
        d = np.sqrt(x**2 + y**2 + z**2)
        return d

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        # instead of using average, euclidean distance seems to work better
        current_distance = self.get_euclidean_distance(self.sim.pose[:3], self.target_pos)

        # add angular velocity should be 0 for taking off (going straight up)
        distance_angular = self.get_euclidean_distance(self.sim.angular_v, [0, 0, 0])

        # clip the reward within (-1, 1) using tanh function
        reward = np.tanh(2. + 1./current_distance + 1./distance_angular)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            if done and abs(self.distance) > 2:
                reward -= 10
            if done and abs(self.distance) < 2:
                reward += 1000
                done = True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
