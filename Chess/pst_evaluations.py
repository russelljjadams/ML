# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 14:03:12 2023

@author: rjadams
"""
import numpy as np
import random


def mutate_table(table, mutation_rate):
    for key in table:
        for i in range(len(table[key])):
            # Generate a random number between 0 and 1
            mutation_prob = random.random()
            # If the mutation probability is less than the mutation rate, mutate the value
            if mutation_prob < mutation_rate:
                # Generate a random number between -3 and 3
                mutation = random.randint(-3, 3)
                # Mutate the value by adding the mutation
                table[key][i] += mutation
    return table


# Let's assume pst is your original piece-square table for white, as you provided it.
# Convert all of the PSTs to NumPy arrays and flip them for black.
def mirror_table(pst):
    pst_np = {}
    for p, table in pst.items():
        # Convert to a 2D numpy array
        pst_table = np.array(table).reshape(8, 8)
        # Flip the array horizontally to mirror the file for black
        pst_table_flipped = np.flip(pst_table, axis=1)
        # Store the flipped tables in a new dictionary
        pst_np[p] = pst_table_flipped
    return pst_np


piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
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

# Define the standard values of the pieces and the piece square tables...
# (reuse the values from the code you provided earlier)

class Piece:
    def __init__(self, piece_type, position, color):
        self.piece_type = piece_type
        self.position = position
        self.color = color
        self.base_value = piece[self.piece_type.upper()]  # Ensure 'piece' is a dictionary
        self.pst_value = self.get_pst_value()

    def get_pst_value(self):
        rank, file = divmod(self.position, 8)
        
        if self.color == 'black':
            # Use the original PST, and invert the rank to lookup for white.
            return pst[self.piece_type][(7 - rank) * 8 + file]
        else:
            # Use the flipped PST. Rank stays the same because pst_np is already mirrored/flipped for files.
            return pst_np[self.piece_type][rank, (7 - file)]

    def value(self):
        # Return the combined base value and pst value
        return self.base_value + self.pst_value

# And modify our load_fen method:

class Board:
    def __init__(self, fen):
        self.pieces = []
        self.load_fen(fen)

    def load_fen(self, fen):
        # Parse FEN and create Piece objects
        rows = fen.split()[0].split('/')
        for rank, row in enumerate(rows):
            file = 0
            for c in row:
                if c.isdigit():  # Skip empty squares
                    file += int(c)
                else:
                    color = 'white' if c.isupper() else 'black'
                    piece_type = c.upper()  # Use upper-case character for the piece type
                    position = rank * 8 + file  # 0-based index
                    self.pieces.append(Piece(piece_type, position, color))
                    file += 1
    
    # Evaluate the board by summing piece values and subtracting black's from white's
    def evaluate(self):
        score = 0
        for piece in self.pieces:
            piece_value = piece.value()
            if piece.color == 'white':
                score += piece_value
                print(f"White {piece.piece_type} at {piece.position} value: {piece_value}")
            else:
                score -= piece_value
                print(f"Black {piece.piece_type} at {piece.position} value: {piece_value}")
        return score

# Example usage with a FEN
fen = fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
board = Board(fen)

evaluation = board.evaluate()
print(f'The board evaluation is {evaluation}')
