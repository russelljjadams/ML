# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:04:30 2024

@author: rjadams
"""

import numpy as np
import chess
import sys
import random

import cProfile
import pstats

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # The king's value is not important in this context
}


def index_to_coordinate(index):
    file = index % 8  # Get the file (column)
    rank = 8 - (index // 8)  # Get the rank (row)
    return f"{chr(file + ord('a'))}{rank}"

def move_to_notation(from_index, to_index):
    return f"{index_to_coordinate(from_index)}{index_to_coordinate(to_index)}"

def order_moves(board, moves):
    def mvv_lva(move):
        # Most Valuable Victim - Least Valuable Aggressor
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if victim is not None:
                victim_value = PIECE_VALUE[victim.piece_type]
                attacker_value = PIECE_VALUE[board.piece_at(move.from_square).piece_type]
                return victim_value - attacker_value
        return 0

    # Sort the moves in descending order, so high value captures are first
    # Since we want the least valuable aggressor, we subtract the attacker_value
    sorted_moves = sorted(moves, key=mvv_lva, reverse=True)
    return sorted_moves

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

pst_np = {}
for p, table in pst.items():
    # Convert to a 2D numpy array
    pst_table = np.array(table).reshape(8, 8)
    pst_np[p] = pst_table

#pst_np = mirror_table(pst)


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
            # Access value for white from pst. Rank is inverted because the array starts from 0 at the top (rank 8).
            return pst[self.piece_type][(7 - rank) * 8 + file]
        else:
            # Access value for black from the same pst as white.
            # Here, we invert the file accessing since the left-to-right order from black's perspective is mirrored.
            return pst[self.piece_type][rank * 8 + (7 - file)]
        
    def value(self):
        # Return the combined base value and pst value
        return self.base_value + self.get_pst_value()

class Board:
    zobrist_table = {}
    def __init__(self, fen):
        self.pieces = []
        self.board = chess.Board(fen)  # Initialize python-chess Board
        self.load_fen(fen)
        
    def move(self, move):
        self.board.push_uci(move)
        
    def to_move(self):
        if 'w' in self.board.fen():
            return True
        else:
            return False
        
    def generate_checking_moves(self):
        # Generate moves that give check but are not captures
        # This list could potentially be very long, so it might be necessary to
        # use some additional heuristics to prune out unlikely checks
        checking_moves = []
        for move in self.board.legal_moves:
            if self.board.gives_check(move) and not self.board.is_capture(move):
                checking_moves.append(move)
        return checking_moves
    

    
    def alpha_beta_search(self, alpha, beta, depth):   
        if depth <= 0:
            return self.quiescence_search(alpha, beta)
    
        if self.board.is_game_over(claim_draw=False):
            return self.evaluate()
    
        moves = list(self.board.legal_moves)
        moves = order_moves(self.board, moves)
    
        first_move = True
        for move in moves:
            self.board.push(move)
    
            if first_move:
                score = -self.alpha_beta_search(-beta, -alpha, depth - 1)
                first_move = False
            else:
                score = -self.alpha_beta_search(-alpha - 1, -alpha, depth - 1)
                if alpha < score < beta:
                    score = -self.alpha_beta_search(-beta, -alpha, depth - 1)
            
            self.board.pop()
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta-cutoff.
    
        return alpha
        
    def quiescence_search(self, alpha, beta, depth=0, max_depth=100):
        stand_pat = self.evaluate()
    
        # If the game is over (checkmate or draw), return the evaluation immediately
        if self.board.is_game_over(claim_draw=False):
            return stand_pat
    
        if depth > max_depth:
            return stand_pat
    
        if stand_pat >= beta:
            return beta
    
        if alpha < stand_pat:
            alpha = stand_pat
    
        moves = list(self.board.generate_legal_captures())
        # checks = self.generate_checking_moves()
        # moves.extend(checks)
        # Order moves by the MVV-LVA heuristic
        moves = order_moves(self.board, moves)
    
        for move in moves:
            self.board.push(move)
            score = -self.quiescence_search(-beta, -alpha, depth + 1)
            self.board.pop()
    
            if score >= beta:
                return beta
    
            if score > alpha:
                alpha = score
    
        return alpha

    def get_pst_value(self, piece, square, color):
        piece_type = piece.symbol().upper()
        if color == chess.WHITE:
            return pst[piece_type][(7 - square // 8) * 8 + square % 8]
        else:
            return pst_np[piece_type][square // 8, square % 8]


    def load_fen(self, fen):
        self.pieces = []  # Reset pieces list
        self.board.set_fen(fen)  # Update the internal python-chess board
        piece_map = self.board.piece_map()  # Get a dictionary of pieces by square index
        for square, piece in piece_map.items():
            color = 'white' if piece.color else 'black'
            piece_type = piece.symbol().upper()
            pst_value = self.get_pst_value(piece, square, piece.color)
            self.pieces.append(Piece(piece_type, square, color, pst_value))

    def piece_value(self, piece, square, color):
        # Base values for the pieces, defined outside of the function
        base_piece_values = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
        piece_type = piece.symbol().upper()
    
        # Get the base value of the piece
        base_value = base_piece_values[piece_type]
    
        # Get the piece-square table value
        if color == chess.WHITE:
            pst_value = pst[piece_type][(7 - square // 8) * 8 + square % 8]
        else:
            # Use the mirrored table for black
            pst_value = pst_np[piece_type][square // 8, square % 8]
    
        # The total value is the base value plus the position value
        total_value = base_value + pst_value
        return total_value
    
    # Example usage within the evaluation function
    def evaluate(self):
        # Check if the game is over
        if self.board.is_game_over(claim_draw=False):
            # If the current player's king is in check, it's checkmate
            is_checkmate = self.board.is_checkmate()
            if is_checkmate:
                # Return -infinity if it's the side to move's turn, and infinity if it's opponent's turn
                return -float('inf')
            else:
                # It's a stalemate or draw by insufficient material, etc.
                return 0  # You may handle different types of draws differently depending on your engine design
        
        evaluation = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_value = self.piece_value(piece, square, piece.color)
                if piece.color == chess.WHITE:
                    evaluation += piece_value
                else:
                    evaluation -= piece_value
        
        # Return the evaluation score from the perspective of the side whose turn it is
        return evaluation if self.board.turn == chess.WHITE else -evaluation
    
    def generate_moves(self, quiescence=False):
        # Generate only moves that are captures if quiescence is True
        if quiescence:
            return list(self.board.generate_legal_captures())
        else:
            return list(self.board.generate_legal_moves())

    def top_moves(self, depth=4):
        if self.board.is_game_over(claim_draw=False):
            return []  # No further exploration is needed for the final game state.
    
        moves = self.board.legal_moves
        # Order moves by the MVV-LVA heuristic
        moves = order_moves(self.board, moves)
        scored_moves = []
        
        for move in moves:
            self.board.push(move)
            
            # Use the alpha-beta search instead of the quiescence search, with the provided search depth.
            score = -self.alpha_beta_search(-float('inf'), float('inf'), (depth - 1))
            scored_moves.append((move, score))
            self.board.pop()
    
        # Sort and return the moves by evaluation score
        return sorted(scored_moves, key=lambda x: x[1], reverse=True)
    
    def top_five(self):
        t = self.top_moves()
        for move, score in t[:5]:
            print(f"Move {move} with score {score}")


# Example usage with a FEN
#fen = "r1bqk1nr/ppp3pp/2np4/8/4P3/2Q2N2/P4PPP/RNB2RK1 w - - 0 11"
#fen = "r1bqk1nr/ppp3pp/2np4/8/3QP3/5N2/P4PPP/RNB2RK1 b - - 1 11"
#fen = 'rnbqkbnr/p1pppppp/8/8/1pPPP3/8/PP3PPP/RNBQKBNR b KQkq c3 0 3'
#fen = 'rnbqkb1r/p1pppp1p/5np1/1B6/1p1PP3/5N2/PPP2PPP/RNBQK2R w KQkq - 2 5' #problem fen
#fen = 'rnbqkb1r/p1pppp1p/5np1/1B6/1p1PP3/2N2N2/PPP2PPP/R1BQK2R b KQkq - 3 5' #problem2 fen
#fen = '6Q1/5R2/4B2k/8/3P2P1/8/PB6/6K1 w - - 1 52' #mate in 1
fen = '8/p3R3/1p6/6p1/5b2/2B2k2/PPr5/6K1 w - - 5 38' #mate inc
#fen = 'r1bq1rk1/pp2bppp/2n1pn2/8/8/PBN5/1P2NPPP/R1BQ1RK1 w - - 2 13' #wtf

# profiler = cProfile.Profile()
# profiler.enable()

# board = Board(fen)
# top_moves = board.top_moves()

# profiler.disable()

# print(board.board)
# print(f'The board evaluation is {board.evaluate()}')
# print("Top 5 moves:")
# for move, score in top_moves[:5]:  # Limit to the top 5 moves
    # print(f"Move {move} with score {score}")
    
# stats = pstats.Stats(profiler).sort_stats('cumulative')
# stats.print_stats()
    
# Example UCI loop
def main():
    board = Board(fen=chess.STARTING_FEN)
    waiting_for = ""  # This is used to process multi-part commands like "position" and "go"
    
    print("id name SimpleChessBro")
    print("id author Russell J. Adams")
    print("uciok")
    sys.stdout.flush()
    
    while True:
        raw_input = input()
        if raw_input == "uci":
            print("id name SimpleChessEngine")
            print("id author Russell J. Adams")
            print("uciok")
            sys.stdout.flush()
        elif raw_input == "isready":
            print("readyok")
            sys.stdout.flush()
        elif raw_input == "ucinewgame":
            board = Board(fen=chess.STARTING_FEN)
        elif raw_input.startswith("position"):
            _, position_cmd, *position_data = raw_input.split(" ")
            if position_cmd == "fen":
                fen = " ".join(position_data)
                board = Board(fen)
            elif position_cmd == "startpos":
                board = Board(fen=chess.STARTING_FEN)
                fen = chess.STARTING_FEN
            if "moves" in raw_input:
                moves_str = raw_input.split("moves ")[1]
                moves = moves_str.strip().split()
                for move in moves:
                    board.move(move)
        elif raw_input.startswith("go"):
            # Handle the rest of the 'go' command parameters here
            # Sample output during search (depth, score in centipawns, best move so far as a string)
            depth = 1
            top_moves = board.top_moves()
            if top_moves:
                best_move = top_moves[0][0]
                score = top_moves[0][1]
                print(f"info depth {depth} score cp {score} pv {best_move}")
                sys.stdout.flush()
                print(f"bestmove {best_move.uci()}")
                sys.stdout.flush()
        elif raw_input == "quit":
            break

#Entry point
if __name__ == "__main__":
    #pass
    main()