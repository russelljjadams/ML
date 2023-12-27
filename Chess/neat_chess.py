# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:39:01 2023

@author: rjadams
"""
import neat
import torch
import chess
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import pickle

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
    
def play_game_with_network_random(env, net):
    env.reset()

    while not env.is_game_over():
        board_state = env.get_board_representation()
        
        if env.board.turn == chess.WHITE:
            # It's the NEAT network's (white) turn
            # NEAT network accepts a list of inputs and returns a list of outputs
            predictions = net.activate(board_state)
            legal_moves = env.get_possible_moves()
            
            # Initialize a low score for each move
            legal_move_scores = [-np.inf] * len(legal_moves)
            for idx, move in enumerate(legal_moves):
                # Assuming the network's output corresponds to the end square
                move_index = chess.SQUARE_NAMES.index(move[2:4])
                legal_move_scores[idx] = predictions[move_index]
            
            # Choose the move with the highest score
            best_move = legal_moves[np.argmax(legal_move_scores)]
            env.play_move(best_move)
        else:
            # It's the random agent's (black) turn
            random_move = random.choice(list(env.board.legal_moves)).uci()
            env.play_move(random_move)

    # Determine game result
    return env.get_game_result() if env.get_game_result() != 'undetermined' else 'draw'



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Create network from genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        win, loss, draw = 0, 0, 0
        iterations = 10
        # Instantiate your ChessEnvironment
        env = ChessEnvironment()
        for _ in range(iterations):
            env.reset()
            game_result = play_game_with_network_random(env, net)
            if game_result == 'win':
                win += 1
            elif game_result == 'loss':
                loss += 1
            else:
                draw += 1

        # Fitness function
        fitness = (win + (draw / 2)) / iterations
        genome.fitness = fitness  # Assign fitness to the genome

# Read configuration file
config_path = 'config-feedforward.txt'
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

# Run for a maximum number of generations.
winner = p.run(eval_genomes, 10)

# Save the winning genome to file (optional).
with open('winner.pkl', 'wb') as f:
    pickle.dump(winner, f)

# Convert the winner genome into a neural network (to use it or analyze it further).
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

# Use winner_net for further tasks, such as playing games or analysis.