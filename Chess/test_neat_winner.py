# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:28:42 2023

@author: rjadams
"""

import neat
import pickle
import chess
import numpy as np
import random

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

# Load the configuration.
config_path = 'config-feedforward.txt'  # Update the path to your NEAT configuration file.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Load the saved winner.
with open('winner.pkl', 'rb') as f:
    winner_genome = pickle.load(f)

# Convert the winner genome into a neural network.
winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

# Test the loaded genome. You can use the ChessEnvironment.
env = ChessEnvironment()  # This would be the class you created for the chess environment.

# Play a game with the winner network.
# If you have a function to play a game given a network and environment, you can call it here.
# For example:
game_result = play_game_with_network_random(env, winner_net)

# Print the result of the game.
print(f"The winner network's game result: {game_result}")