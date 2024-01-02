# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 17:08:51 2024

@author: rjadams
"""

import numpy as np
import chess


def index_to_coordinate(index):
    file = index % 8  # Get the file (column)
    rank = 8 - (index // 8)  # Get the rank (row)
    return f"{chr(file + ord('a'))}{rank}"

def move_to_notation(from_index, to_index):
    return f"{index_to_coordinate(from_index)}{index_to_coordinate(to_index)}"

def mirror_table(pst):
    pst_np = {}
    for p, table in pst.items():
        # Convert to a 2D numpy array
        pst_table = np.array(table).reshape(8, 8)
        # Flip the array horizontally to mirror the file for black
        pst_table_flipped = np.flip(pst_table, axis=0)
        pst_table_flipped = np.flip(pst_table_flipped, axis=0)
        # Store the flipped tables in a new dictionary
        pst_np[p] = pst_table_flipped
    return pst_np

piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
# Add your piece-square tables (PSTs) for the 'P', 'N', 'B', 'R', 'Q', 'K' pieces. 
# The example given below is a representation of PSTs for each piece (these tables need to reflect your desired values).
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}

pst_np = mirror_table(pst)


class Piece:
    def __init__(self, piece_type, position, color, pst_value):
        self.piece_type = piece_type
        self.position = position
        self.color = color
        self.base_value = piece[self.piece_type.upper()]
        self.pst_value = pst_value  # Use the pst_value passed in from the Board class

    def get_pst_value(self):
        rank, file = divmod(self.position, 8)
        if self.color == 'white':
            return pst[self.piece_type][(7 - rank) * 8 + file]  # Access value for white from pst
        else:
            return pst_np[self.piece_type][rank, file]  # Access value for black from mirrored pst

    def possible_moves(self):
        directions = {
            'P': [(-1, 0)],  # Only capture moves, no promotion or double moves.
            'N': [(-2, -1), (-1, -2), (-2, 1), (-1, 2), (1, -2), (2, -1), (1, 2), (2, 1)],
            'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            'R': [(0, -1), (0, 1), (-1, 0), (1, 0)],
            'Q': [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)],
            'K': [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)],
        }
        rank, file = divmod(self.position, 8)
        moves = []
        for dr, df in directions[self.piece_type]:
            r, f = rank + dr, file + df
            if 0 <= r < 8 and 0 <= f < 8:
                moves.append(r * 8 + f)
        return moves
    
    def value(self):
        # Return the combined base value and pst value
        return self.base_value + self.get_pst_value()


class Board:
    def __init__(self, fen):
        self.pieces = []
        self.board = chess.Board(fen)  # Initialize python-chess Board

        self.load_fen(fen)

    def get_pst_value(self, piece, square, color):
        piece_type = piece.symbol().upper()
        if color == chess.WHITE:
            return pst[piece_type][(7 - square // 8) * 8 + square % 8]
        else:
            return pst_np[piece_type][square // 8, 7 - (square % 8)]

    def load_fen(self, fen):
        # Load the positions from the python-chess board instead
        self.pieces = []  # Reset pieces list
        self.board.set_fen(fen)  # Update the internal python-chess board
        piece_map = self.board.piece_map()  # Get a dictionary of pieces by square index
        for square, piece in piece_map.items():
            color = 'white' if piece.color else 'black'
            piece_type = piece.symbol().upper()
            position = square
            pst_value = self.get_pst_value(piece, square, piece.color)
            self.pieces.append(Piece(piece_type, position, color, pst_value))

    def evaluate(self):
        white_score = sum(piece.value() for piece in self.pieces if piece.color == 'white')
        black_score = sum(piece.value() for piece in self.pieces if piece.color == 'black')
        # for piece in self.pieces:
        #     print(f"Piece: {piece.piece_type}, Color: {piece.color}, Position: {piece.position}, " +
        #           f"PST_Value: {piece.get_pst_value()}, Base_Value: {piece.base_value}")
        score = white_score - black_score
        return score

    def generate_moves(self):
        moves = []
        for move in self.board.generate_legal_moves():
            moves.append(move)
        return moves

    def top_moves(self, n=5):
        moves = self.generate_moves()
        
        # Evaluate the board after each move
        scored_moves = []
        fen = self.board.fen()
        for move in moves:
            # Simulate the move
            
            # Push the move to the board
            self.board.push(move)
            
            # Update the piece list to reflect the new board state
            self.load_fen(self.board.fen())
            
            # Evaluate the board state after the move
            score = self.evaluate()
            scored_moves.append((move, score))
            
            # Undo the move to restore the board to its initial state
            #self.board.pop()
            
            # Reset the board state to the initial FEN (the list of pieces is also updated here)
            self.load_fen(fen)
        
        # Sort the moves by evaluation score
        scored_moves.sort(key=lambda x: x[1], reverse=True)  # Assuming white's perspective
        return scored_moves[:n]

# Example usage with a FEN
fen = "r1bqk1nr/ppp3pp/2np4/8/4P3/2Q2N2/P4PPP/RNB2RK1 w - - 0 11"

board = Board(fen)
top_moves = board.top_moves(5)

print(f'The board evaluation is {board.evaluate()}')
print("Top 5 moves:")
for move, score in top_moves[:5]:  # Limit to the top 5 moves
    print(f"Move {move} with score {score}")
