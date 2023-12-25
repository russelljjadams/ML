# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Game environment for Tic-Tac-Toe
class TicTacToe:
    def __init__(self):
        self.reset()

    def render(self):
        # Map the internal representation to X and O for rendering
        symbols = {1: 'X', -1: 'O', 0: '-'}
        # Print the board
        print('\n'.join([' '.join([symbols[cell] for cell in row]) for row in self.board]))
        print()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.flatten()

    def step(self, action):
        x, y = divmod(action, 3)
        if self.board[x, y] != 0:
            return self.board.flatten(), -1000, True  # Illegal move
        self.board[x, y] = self.current_player
        self.done, self.winner = self.check_winner()
        reward = self.winner * 10 if self.winner else 0
        self.current_player *= -1
        return self.board.flatten(), reward, self.done

    def check_winner(self):
        for p in [1, -1]:
            # Check rows and columns
            if any(np.all(self.board == p, axis=0)) or any(np.all(self.board == p, axis=1)):
                return True, p
            # Check diagonals
            if np.all(np.diag(self.board) == p) or np.all(np.diag(np.fliplr(self.board)) == p):
                return True, p
        if not np.any(self.board == 0):
            return True, 0  # Tie
        return False, None

    def legal_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

# Define the Q-Network using PyTorch
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

def train_dqn(env, trained_policy_net=None):
    # Hyperparameters
    gamma = 0.99
    epsilon = 0.999
    epsilon_decay = 0.995
    epsilon_min = 0.01
    learning_rate = 0.01
    batch_size = 64
    memory_size = 256
    num_episodes = 10000
    update_target_frequency = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Two networks for stability
    policy_net = DQN(9,9).to(device)
    if trained_policy_net:
        policy_net.load_state_dict(trained_policy_net.state_dict())
        
    target_net = DQN(9, 9).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Set target net to evaluation mode

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)
    loss_fn = nn.MSELoss()

    def choose_action(state, legal_actions):
        if np.random.rand() < epsilon:
            return np.random.choice(legal_actions)
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) # Ensure 2D tensor
            action_values = policy_net(state_tensor).squeeze(0) # Unsqueeze (batch dimension), then squeeze it out after prediction
        legal_action_values = action_values[legal_actions]
        return legal_actions[torch.argmax(legal_action_values).item()]

    def replay():
        if len(memory) < batch_size:
            return
        minibatch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        # Convert to PyTorch tensors (ensuring states and next_states as float32 and reshaping actions and dones if necessary)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).view(-1, 1).to(device)  # Ensure actions are the right shape with batch_dim
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Get current Q values from policy net
        curr_Q = policy_net(states).gather(1, actions)

        # Compute value of next states using target net
        next_Q = target_net(next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_Q = rewards + gamma * next_Q * (1 - dones)

        # Compute loss between current Q values and expected Q values
        loss = loss_fn(curr_Q.squeeze(), expected_Q)

        # Gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    update_frequency = 500  # Set the update frequency for performance evaluation
    eval_games = 50  # Number of games to evaluate performance

    # Track the performance
    performance = []
    total_iterations = 0

    # Main training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            legal_actions = env.legal_actions()
            action = choose_action(state, legal_actions)
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            replay()

        # Update target network
        if episode % update_target_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay


        total_iterations += 1
        # Periodically evaluate the model's performance
        if total_iterations % update_frequency == 0:
            wins = 0
            draws = 0
            losses = 0

            for _ in range(eval_games):
                result = play(env, policy_net, random_agent=True)
                if result == 1:
                    losses += 1
                elif result == 0:
                    draws += 1
                else:
                    wins += 1

            # Calculate win rate and add to performance list
            win_rate = (wins + int((draws/2))) / eval_games
            performance.append(win_rate)
            print(wins, draws, losses)
            print(f'Performance at iteration {total_iterations}: {win_rate}')
            print()

    return policy_net



def play(env, policy_net, random_agent=False):
    state = env.reset()
    done = False
    if not random_agent:
        env.render()
        
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The human player will be 1 ('X'), and the agent will be -1 ('O').
    human_player = 1
    agent_player = -1
    random_player = 0
    if random_agent:
        random_player = 1
        human_player = 0
    current_player = 1  # Human goes first

    while not done:
        if current_player == random_player:
            move = np.random.choice(env.legal_actions())
            state, reward, done = env.step(move)
        elif current_player == human_player:
            # Human's turn
            print("Your turn! Current board (1-9):")
            move = int(input("Choose your move (1-9): ")) - 1
            while move not in env.legal_actions():
                print("Invalid move. Moves are 1-indexed, make sure to pick an empty spot.")
                move = int(input("Choose your move (1-9): ")) - 1

            state, reward, done = env.step(move)
        else:
            # Agent's turn
            if not random_agent:
                print("Agent's turn!")
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()  # The agent selects the action with the highest Q-value

                # Checking if action is valid. If not, choose the next best action
                legal_actions = env.legal_actions()
                if action not in legal_actions:
                    illegal_action_mask = torch.ones(9, device=device, dtype=torch.float32) * float('-inf')
                    illegal_action_mask[legal_actions] = 0
                    q_values[0] += illegal_action_mask
                    action = q_values.max(1)[1].item()

            state, reward, done = env.step(action)

        if not random_agent:
            env.render()

        if done:
            if not random_agent:
                if env.winner == human_player:
                    print("Congratulations, you've won!")
                elif env.winner == agent_player:
                    print("The agent has won!")
                else:
                    print("It's a draw!")
            return env.winner

        current_player *= -1  # Switch players
        

# Filtered Net Functions to generate Random Weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# takes in a module and applies the specified weight initialization
def init_weights_general(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def filtered_nets(env, n_games=100):
    b_model = DQN(9,9).to(device)
    b_model_losses = 100
    
    model = DQN(9, 9).to(device)
    model.eval()
    while True:
        model.apply(init_weights_general)
        
        wins = 0
        draws = 0
        losses = 0
        for i in range(n_games):
            game_outcome = play(env, model, True)
            if game_outcome == 1:
                losses += 1
            elif game_outcome == 0:
                draws += 1
            else:
                wins += 1
        if losses < b_model_losses:
            print("Losses: {}   Previous Losses: {}".format(losses, b_model_losses))
            b_model.load_state_dict(model.state_dict())
            b_model_losses = losses
        if losses <= 21:
            break
    return b_model
        
        
    



# Train the DQN
env = TicTacToe()

# Check if the saved model exists
if os.path.isfile('tictactoe_dqn.pth'):
    # Load the saved model
    trained_policy_net = DQN(9, 9)
    trained_policy_net.load_state_dict(torch.load('tictactoe_dqn.pth'))
    #trained_policy_net.eval()
    trained_policy_net_new = train_dqn(env, trained_policy_net)
    # Save the trained model
    torch.save(trained_policy_net_new.state_dict(), 'tictactoe_dqn.pth')
else:
    filtered = filtered_nets(env)
    # Train the DQN
    trained_policy_net = train_dqn(env,filtered)
    # Save the trained model
    torch.save(trained_policy_net.state_dict(), 'tictactoe_dqn.pth')

trained_policy_net.eval()
play(env, trained_policy_net)