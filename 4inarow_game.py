import numpy as np
import copy

#Board class which stores an instance of the game:
class Board(object):

    #Creates a board with the inputted grid, defaults to empty/starting position:
    def __init__(self, grid = [[(False, False) for i in range(6)] for j in range(7)]):
        self.grid = grid
        #Ensures functions are only called once:
        self.current_player = self.get_current_player()
        self.valid_moves = self.get_valid_moves()
        self.game_status = self.get_game_status()

    #Returns the board to the empty/starting position
    def restart(self):
        self.grid = [[(False, False) for i in range(6)] for j in range(7)]
        self.current_player = self.get_current_player()
        self.valid_moves = self.get_valid_moves()
        self.game_status = self.get_game_status()

    #Self explanatory:    
    def get_board_dimensions(self):
        return (6, 7)

    #Returns # of possible moves, i.e. size of policy head:
    def get_action_space_size(self):
        return 7

    #Returns the current player's number (Yellow = 1, Red = 2, Yellow goes first):
    def get_current_player(self):
        count = 0
        for col in self.grid:
            for element in col:
                if element[0]:
                    count += 1
        return (count%2)+1
        
    #Returns a Board object of the game instance if the inputted move were to be played:
    #Move is an integer from 0 to 6
    def make_move(self, move):
        col = self.grid[move]
        i = 0
        while i < len(col) and col[i][0]:
            i += 1
        #Have to copy state so that new grid doesn't change current one:
        newGrid = copy.deepcopy(self.grid)
        newGrid[move][i] = (True, bool(self.current_player-1))
        return Board(newGrid)

    #Returns all moves the current player can make, as a list of integers:
    def get_valid_moves(self):
        moves = []
        for move in range(0, 7):
            if not self.grid[move][-1][0]:
                moves.append(move)
        return moves
    
    #Checks if player with given token has won:
    def check_win(self, player_token):
            #Horizontal check:
            for r in range(6):
                for c in range(4):
                    if all(self.grid[c + i][r] == player_token for i in range(4)):
                        return True

            #Vertical check:
            for c in range(7):
                for r in range(3):
                    if all(self.grid[c][r + i] == player_token for i in range(4)):
                        return True

            #Bottom left to top right check:
            for c in range(4):
                for r in range(3):
                    if all(self.grid[c + i][r + i] == player_token for i in range(4)):
                        return True

            #Top left to bottom right checkL
            for c in range(4):
                for r in range(3, 6):
                    if all(self.grid[c + i][r - i] == player_token for i in range(4)):
                        return True
            return False

    #Returns number of winner if there is one, -1 for a draw, and 0 if ongoing
    def get_game_status(self):
        yellow_token = (True, False)
        red_token = (True, True)

        # Check for player wins
        if self.check_win(yellow_token):
            return 1
        if self.check_win(red_token):
            return 2

        if not self.valid_moves:
            return -1

        # If no win and no draw, the game is ongoing
        return 0
    
    #Prints out readable version of board: (X for red counter, O for yellow counter)
    def print_board(self):
        yellow_token = (True, False)
        red_token = (True, True)
        empty_token = (False, False)

        #Dictionary to map token values to display characters
        char_map = {
            red_token: 'X',
            yellow_token: 'O',
            empty_token: '-'
        }

        print("------------------------")

        #Iterate through rows:
        for r in range(5, -1, -1):
            row_str = "|"
            #Iterate through columns:
            for c in range(7):
                row_str += f" {char_map.get(self.grid[c][r], '?')} "
            row_str += "|"
            print(row_str)

        print("------------------------")
        print("| 1  2  3  4  5  6  7 |")
        print("------------------------\n")

    #Transforms state for neural network:
    def transform_state(self):
        #Keras Conv2D expects (height, width, channels), so we use (6, 7, 3)
        state = np.zeros((6, 7, 3), dtype=np.float32)
        
        yellow_token = (True, False)
        red_token = (True, True)

        #Iterate through columns and rows:
        for c in range(7):
            for r in range(6):
                if self.grid[c][r] == yellow_token:
                    state[r, c, 0] = 1
                elif self.grid[c][r] == red_token:
                    state[r, c, 1] = 1
        
        #Third channel indicates player to move, with plane of 0s for yellow, 1s for red
        if self.current_player == 2: # Red's turn
            state[:, :, 2] = 1
            
        return state