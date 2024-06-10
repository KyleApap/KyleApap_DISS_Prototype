import random
import math
import time
import pandas as pd
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self):
        self.board = [' ']*9
        self.player = 'X'
    
    def print_board(self):
        for i in range(0, 9, 3):
            print('|'.join(self.board[i:i+3]))
    
    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']
    
    def make_move(self, move):
        self.board[move] = self.player
        self.player = 'O' if self.player == 'X' else 'X'
    
    def is_winner(self, player):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
            [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]
        ]
        for combo in winning_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    
    def is_draw(self):
        return ' ' not in self.board
    
    def game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

class MiniMaxPlayer:
    def __init__(self, symbol, depth=1):
        self.symbol = symbol
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.depth = depth  # Initial depth
        self.simulations = 200

    def minimax(self, game, depth, maximizing_player):
        if game.game_over() or depth == 0:
            return self.evaluate(game)

        if maximizing_player:
            max_eval = -math.inf
            for move in game.available_moves():
                game.make_move(move)
                eval = self.minimax(game, depth - 1, False)
                game.board[move] = ' '
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = math.inf
            for move in game.available_moves():
                game.make_move(move)
                eval = self.minimax(game, depth - 1, True)
                game.board[move] = ' '
                min_eval = min(min_eval, eval)
            return min_eval

    def evaluate(self, game):
        if game.is_winner(self.symbol):
            return 1
        elif game.is_winner('O' if self.symbol == 'X' else 'X'):
            return -1
        else:
            return 0
        
    def update_stats(self, result):
        self.total_games += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1

    def adjust_difficulty(self):
      
        win_rate = self.wins / self.total_games
        loss_rate = self.losses / self.total_games
        draw_rate = self.draws / self.total_games

        # decrease depth if win rate is low, keep stable if balanced
        if win_rate < 0.50:
            self.depth += 1  # Increase depth
        elif win_rate > 0.50:
            self.depth -= 1  # Decrease depth


        self.depth = max(min(self.depth, 3), 1) ##########################################################################

    def get_best_move(self, game):  
        best_move = None
        best_eval = -math.inf
        for move in game.available_moves():
            game.make_move(move)
            eval = self.minimax(game, self.depth - 1, False)  # Use dynamic depth
            game.board[move] = ' '
            if eval > best_eval:
                best_eval = eval
                best_move = move
        return best_move



class MCTSPlayer:
    def __init__(self, symbol, simulations=5):
        self.symbol = symbol
        self.simulations = simulations
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def mcts(self, game):
        root = Node(None, None, game.board.copy())

        for _ in range(self.simulations):
            node = root
            temp_game = TicTacToe()
            temp_game.board = game.board.copy()

           
            while node.children:
                node = node.select_child()
                temp_game.make_move(node.move)

            
            if not temp_game.game_over():
                untried_moves = temp_game.available_moves()
                move = random.choice(untried_moves)
                temp_game.make_move(move)
                node = node.add_child(move, temp_game.board.copy())

            
            while not temp_game.game_over():
                move = random.choice(temp_game.available_moves())
                temp_game.make_move(move)
        
            while node:
                node.update(temp_game)
                node = node.parent

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move
    
    def update_stats(self, result):
        self.total_games += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        else:
            self.draws += 1

    def adjust_difficulty(self):

        # Adjust simulation count based on win/loss/draw rates
        win_rate = self.wins / self.total_games
        loss_rate = self.losses / self.total_games
        draw_rate = self.draws / self.total_games


        if win_rate < 0.50:
            self.simulations += 1  # Add a fixed amount
        elif win_rate > 0.50:
            self.simulations -= 1  # Subtract a fixed amount

        self.simulations = max(self.simulations, 1)

class Node:
    def __init__(self, parent, move, board):
        self.parent = parent
        self.move = move
        self.board = board
        self.visits = 0
        self.wins = 0
        self.children = []

    def select_child(self):
        return max(self.children, key=lambda c: c.ucb())

    def add_child(self, move, board):
        child = Node(self, move, board)
        self.children.append(child)
        return child

    def update(self, game):
        self.visits += 1
        if game.is_winner('O'):
            self.wins += 1

    def ucb(self):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + math.sqrt(2 * math.log(self.parent.visits) / self.visits)

def play_game(player1, player2, num_games=10):
    player1_win_rates = []
    player2_win_rates = []
    draw_rates = []
    avg_game_times = []
    simulation_sizes = []
    depth_searches = []

    for game_num in range(1, num_games + 1):
        game = TicTacToe()
        players = {'X': player1, 'O': player2}
        start_time = time.time()

        while not game.game_over():
            current_player = players[game.player]
            if isinstance(current_player, MiniMaxPlayer):
                move = current_player.get_best_move(game)
            else:
                move = current_player.mcts(game)
            game.make_move(move)

            if game.is_winner('X'):
                break
            elif game.is_winner('O'):
                break
            elif game.is_draw():
                break

        total_game_time = time.time() - start_time

        # Update player statistics based on game outcome

        if game.is_winner(player1.symbol):
            player1.update_stats('win')
        elif game.is_draw():
            player1.update_stats('draw')
        else:
            player1.update_stats('loss')

        player1.adjust_difficulty()

        if game.is_winner(player2.symbol):
            player2.update_stats('win')
        elif game.is_draw():
            player2.update_stats('draw')
        else:
            player2.update_stats('loss')

        
        player2.adjust_difficulty()

        player1_win_rates.append(player1.wins / player1.total_games if player1.total_games != 0 else 0)
        player2_win_rates.append(player2.wins / player2.total_games if player2.total_games != 0 else 0)
        draw_rates.append(player2.draws / player2.total_games if player2.total_games != 0 else 0)
        avg_game_times.append(total_game_time)
        simulation_sizes.append(player2.simulations)
        depth_searches.append(player1.depth)

        # Print game results in the terminal
        print(f"Game {game_num}:")
        print("Player 1 (MiniMax) Win Rate:", player1.wins / player1.total_games if player1.total_games != 0 else 0)
        print("Player 2 (MCTS) Win Rate:", player2.wins / player2.total_games if player2.total_games != 0 else 0)
        print("Draw Rate:", player2.draws / player2.total_games if player2.total_games != 0 else 0)
        print("Game Time:", total_game_time)
        print()

    # Save results to Excel
    df = pd.DataFrame({
        'Game': list(range(1, num_games + 1)),
        'Player 1 (MiniMax) Win Rate': player1_win_rates,
        'Player 2 (MCTS) Win Rate': player2_win_rates,
        'Draw Rate': draw_rates,
        'Average Game Time (seconds)': avg_game_times,
        'Simulation Size': simulation_sizes,
        'Depth Search': depth_searches
    })
    df.to_excel('game_results.xlsx', index=False)

    # Plotting
    plt.figure(figsize=(15, 5))

    # Plot win rates and draw rates
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_games + 1), player1_win_rates, label='Player 1 (MiniMax)')
    plt.plot(range(1, num_games + 1), player2_win_rates, label='Player 2 (MCTS)')
    plt.plot(range(1, num_games + 1), draw_rates, label='Draw')
    plt.xlabel('Games')
    plt.ylabel('Win Rate / Draw Rate')
    plt.title('Win Rate and Draw Rate Over Games')
    plt.legend()

    # Plot average game time
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_games + 1), avg_game_times, label='Average Game Time', color='orange')
    plt.xlabel('Games')
    plt.ylabel('Average Game Time (seconds)')
    plt.title('Average Game Time Over Games')

    # Plot simulation size and depth search
    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_games + 1), simulation_sizes, label='Simulation Size')
    plt.plot(range(1, num_games + 1), depth_searches, label='Depth Search')
    plt.xlabel('Games')
    plt.ylabel('Value')
    plt.title('Simulation Size and Depth Search Over Games')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    player1 = MiniMaxPlayer('X', depth=3)
    player2 = MCTSPlayer('O', simulations=5)
    play_game(player1, player2, num_games=100)  # Play 100 games