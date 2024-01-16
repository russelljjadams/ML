import torch
import chess
import numpy as np
import torch.nn as nn
import random
import copy

class ValueNet(nn.Module):
    def __init__(self, input_size):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
          nn.Linear(input_size, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32,16),
          nn.ReLU(),
          nn.Linear(16, 1),
          nn.Tanh()  # Output a value between -1 and 1
        )

    def forward(self, x):
        return self.fc(x)

class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()

    def get_board_representation(self):
        return self.board_to_numeric(self.board)

    def is_game_over(self):
        return self.board.is_game_over()

    def play_move(self, move):
        try:
            self.board.push_uci(move)
            return True
        except ValueError:
            return False  # Illegal move

    def get_possible_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def get_game_result(self):
        if self.board.is_checkmate():
            return 'win' if self.board.turn else 'loss'
        elif self.board.is_stalemate() or self.board.can_claim_draw() or self.board.is_insufficient_material():
            return 'draw'
        print(self.board)
        return 'undetermined'

    @staticmethod
    def board_to_numeric(board):
        pieces = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                  'p':-1, 'n':-2, 'b':-3, 'r':-4, 'q':-5, 'k':-6}
        board_squares = np.zeros(64)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                board_squares[i] = pieces[piece.symbol()]
        return board_squares

def init_weights_general(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

def play_game_with_value_network(env, net):
    env.reset()

    while not env.is_game_over():
        if env.board.turn == chess.WHITE:
            # Neural network plays for White
            legal_moves = list(env.board.legal_moves)
            best_move_eval = -np.inf
            best_move = None

            for move in legal_moves:
                env.board.push(move)
                board_state = env.get_board_representation()
                board_tensor = torch.tensor([board_state], dtype=torch.float32)
                move_eval = net(board_tensor).item()
                env.board.pop()

                if move_eval > best_move_eval:
                    best_move_eval = move_eval
                    best_move = move.uci()

            env.play_move(best_move)

        else:
            # Random agent plays for Black
            legal_moves = list(env.board.legal_moves)
            random_move = random.choice(legal_moves).uci()
            env.play_move(random_move)
    result = env.get_game_result()
    #print(result)
    return result

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

# Instantiate your ChessEnvironment
env = ChessEnvironment()

counter = 49
iterations = 10
score = 0
main_model_score = 95
mutate = False

#net = ValueNet(input_size)

# Define the path to your saved model
saved_model_path = '/content/drive/MyDrive/chess_value_net_049_95.00.pth'

# Create an instance of your network
main_model = ValueNet(input_size)

# Load the saved model state_dict
state_dict = torch.load(saved_model_path)

# Apply the loaded state_dict to your model instance
main_model.load_state_dict(state_dict)

# Mutate if we loaded from file
mutate = True

# Loop to test different networks
for game_idx in range(10000):
    if score/100.0 >= .4 or mutate == True:
        print("mutating", main_model_score)
        mutate = True

        if score > main_model_score:
            print("New model")
            main_model = copy.deepcopy(net)
            main_model_score = score
        else:
            w = np.random.randint(1,88)/100
            net = mutate_weights(main_model,w)

    else:
        print("new net")
        net.apply(init_weights_general)
        main_model = copy.deepcopy(net)

    win, loss, draw = 0, 0, 0
    for _ in range(iterations):
        game_result = play_game_with_value_network(env, net)
        if game_result == 'win':
            win += 1
        elif game_result == 'loss':
            loss += 1
        else:  # Draw or undetermined result count as a draw here
            draw += 1

    score = ((win + (draw / 2)) / iterations) * 100
    if score >= 95:
        torch.save(net.state_dict(), f'/content/drive/MyDrive/chess_value_net_{counter}_{score:.2f}.pth')
        counter += 1

    print(f"Game {game_idx}: Win rate: {win / iterations:.2f}, Loss rate: {loss / iterations:.2f}, Draw rate: {draw / iterations:.2f}, Score: {score:.2f}")