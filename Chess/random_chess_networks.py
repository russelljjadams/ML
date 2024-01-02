# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import chess
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import copy

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.fc(x)

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

class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()

    def get_board_representation(self):
        # Convert the board to a fixed-size numerical representation
        return self.board_to_numeric(self.board)

    def is_game_over(self):
        return self.board.is_game_over()

    def play_move(self, move):
        # Here, move should be a legal move in UCI (Universal Chess Interface) format.
        try:
            self.board.push_uci(move)
            return True
        except ValueError:
            return False  # Illegal move

    def get_possible_moves(self):
        # Returns all legal moves in UCI format
        return [move.uci() for move in self.board.legal_moves]

    def get_game_result(self):
        # Determine the result of the game, e.g., 'win', 'loss', 'draw'.
        if self.board.is_checkmate():
            return 'win' if self.board.turn else 'loss' 
        elif self.board.is_stalemate() or self.board.can_claim_draw():
            return 'draw'
        return 'undetermined'

    @staticmethod
    def board_to_numeric(board):
        # You'll need to convert the chess.Board object into a fixed-size numerical array.
        # There are many ways to represent a chess board for a neural network.
        # Here's a simplistic example where we simply use the piece type at each square.
        pieces = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                  'p':-1, 'n':-2, 'b':-3, 'r':-4, 'q':-5, 'k':-6}
        board_squares = np.zeros(64)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                board_squares[i] = pieces[piece.symbol()]
        return board_squares

# Create a method to play a game with a given network and board environment
def play_game_with_network(env, net):
    env.reset()

    while not env.is_game_over():
        board_state = env.get_board_representation()
        # Here we should convert board_state to tensor with appropriate shape (add batch dimension)
        board_tensor = torch.tensor([board_state], dtype=torch.float32)
        legal_move_uci = env.get_possible_moves()
        
        # Get network's predictions for each legal move
        predictions = net(board_tensor).detach().squeeze(0).numpy()

        # Assign high negative value to illegal moves
        move_scores = np.full(env.board.legal_moves.count(), -np.inf)

        # Loop through legal moves and assign predictions to these moves only
        for idx, move in enumerate(legal_move_uci):
            move_index = chess.SQUARE_NAMES.index(move[2:4])  # Assuming output neuron corresponds to end square
            move_scores[idx] = predictions[move_index]
        
        # Select move with highest score as our candidate move which will always be a legal move
        best_move_uci = legal_move_uci[np.argmax(move_scores)]

        # Now play the best move
        env.play_move(best_move_uci)

    return env.get_game_result()

def play_game_with_network_random(env, net):
    env.reset()

    while not env.is_game_over():
        board_state = env.get_board_representation()
        board_state = np.array(board_state)
        
        if env.board.turn == chess.WHITE:
            # It's the network's (white) turn
            board_tensor = torch.tensor([board_state], dtype=torch.float32)
            predictions = net(board_tensor).detach().squeeze(0).numpy()
            legal_move_scores = np.full(env.board.legal_moves.count(), -np.inf)
            
            # Assuming the network outputs a value for every square on the board
            for idx, move in enumerate(env.get_possible_moves()):
                move_index = chess.SQUARE_NAMES.index(move[2:4])
                legal_move_scores[idx] = predictions[move_index]
                
            best_move_uci = env.get_possible_moves()[np.argmax(legal_move_scores)]
            env.play_move(best_move_uci)
        else:
            # It's the random agent's (black) turn
            legal_moves = list(env.board.legal_moves)
            random_move = random.choice(legal_moves).uci()
            env.play_move(random_move)

    return env.get_game_result() if env.get_game_result() != 'undetermined' else 'draw'

def mutate_weights(model, mutation_strength=0.01):
    # Copy the original model to ensure that we do not modify it directly
    new_model = copy.deepcopy(model)
    
    # Apply mutation
    with torch.no_grad():
        for param in new_model.parameters():
            # Ensure we only mutate weights and not biases, etc.
            if len(param.size()) > 0:  
                mutation = torch.randn_like(param) * mutation_strength
                param.add_(mutation)
    return new_model

# Assuming you have an 8x8 board representation with 1 number per square
input_size = 64
output_size = 64  # Assuming network output corresponds to move destination square (simplistic approach)

# Instantiate your ChessEnvironment
env = ChessEnvironment()

# Create the network with random weights
net = DQN(input_size, output_size)

counter = 49
iterations = 15
score = 0
main_model_score = 0
mutate = False

for i in range(10000):
    #net.apply(init_weights_general)  # Apply your random weight initializer
    if score/100.0 >= .6 or mutate == True:
        print("mutating", score)
        mutate = True
        
        if score > main_model_score:
            main_model = copy.deepcopy(net)
            main_model_score = score
        else:
            net = mutate_weights(main_model)
        
    else:
        print("new net")
        net.apply(init_weights_general)
        main_model = copy.deepcopy(net)
        
    # Play a game with this network
    #game_result = play_game_with_network(env, net)

    win, loss, draw = 0, 0, 0
    for i in range(iterations):
        env.reset()
        game_result = play_game_with_network_random(env, net)
        if game_result == 'win':
            win += 1
        elif game_result == 'loss':
            loss += 1
        else:
            draw += 1
    score = ((win+(draw/2)) / iterations) * 100
    if (win+(draw/2)) / iterations >= .7:
            torch.save(net.state_dict(), f'chess_{counter}_{score}.pth')
            counter += 1
    #print((win+(draw/2)) / iterations)  
    print(main_model_score)
                           
    #torch.save(net.state_dict(), f'chess_{counter}.pth')
