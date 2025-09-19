import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import gc

tf.config.run_functions_eagerly(True) 
#CONSTANTS:
C_PUCT = 1.41 #Exploration constant used for MCTS

#Used in the MCTS, a single node on the tree of game states:
class Node(object):
    
    def __init__(self, board, parent, prior_probability):
        self.board = board
        self.parent = parent
        self.player_to_move = self.board.current_player
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior_probability = prior_probability
    
    def is_leaf(self):
        if self.children == {}:
            return True
        else:
            return False
        
    def is_terminal_node(self):
        if self.board.game_status == 0:
            return False
        else:
            return True
        
    def calculate_ucb(self, child):
        global C_PUCT
        UCB = child.value_sum/child.visit_count + C_PUCT * child.prior_probability * np.sqrt(self.visit_count)/(1+child.visit_count)
        return UCB
    
    #Selects the best child based on UCB score, while prioritizing unvisited nodes:
    def select_child(self):
        #Ensure all children get visited first:
        for child in self.children.values():
            if child.visit_count == 0:
                return child

        #Once all children visited, use UCB to decide which child to visit
        best_child = None
        best_score = -float('inf')

        for child in self.children.values():
            score = self.calculate_ucb(child)
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    #Adds all possible children to children dictionary:
    def expand(self, policy_output): #policy_output must be a list/tuple
        moves = self.board.valid_moves
        for move in moves:
            new_child = Node(self.board.make_move(move), self, policy_output[move])
            self.children[move] = new_child

    #Backpropagates value, used in MCTS search UCB formula
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value

'''
#Creates the neural network model:
def create_connect4_model(input_shape=(6, 7, 3), l2_reg=1e-4):
    """
    Args:
        input_shape: The shape of the input data (height, width, channels).
        l2_reg: The L2 regularization factor to prevent overfitting.
    Returns:
        A compiled Keras Model.
    """
    #Input layer:
    inputs = Input(shape=input_shape)

    #Convolutional Body

    #Initial Convolution
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
               kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #Residual Block 1
    shortcut = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)

    #Flatten the output of the convolutional layers to feed into the dense layers
    x_flat = Flatten()(x)

    #Policy Head ---
    #Predicts the best move to make, outputs a probability for each of the 7 columns.
    policy = Dense(32, kernel_regularizer=l2(l2_reg), activation='relu')(x_flat)
    #Final layer uses softmax to create a probability distribution over the 7 moves.
    policy_output = Dense(7, activation='softmax', name='policy_output')(policy)

    #Value Head
    #Evaluates the board position, outputs a single number between -1 (loss) and 1 (win).
    value = Dense(32, kernel_regularizer=l2(l2_reg), activation='relu')(x_flat)
    #Final layer uses tanh, which naturally outputs in the [-1, 1] range.
    value_output = Dense(1, activation='tanh', name='value_output')(value)

    #Create and compile:
    model = Model(inputs=inputs, outputs=[policy_output, value_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'policy_output': 'categorical_crossentropy', #For comparing probability distributions
            'value_output': 'mean_squared_error'      #For comparing the scalar value
        },
        metrics={
            'policy_output': 'accuracy',
            'value_output': 'mean_absolute_error'
        }
    )
    
    return model
'''

#Creates a dynamic, two-headed neural network model for a (generalized) board game between 2 players
def create_dynamic_model(board_dimensions, action_space_size, num_piece_types=1, l2_reg=1e-4):
    """
    Args:
        board_dimensions (tuple): The (height, width) of the game board.
        action_space_size (int): The total number of possible moves in the game.
        num_piece_types (int): The number of unique piece types for a single player.
        l2_reg (float): The L2 regularization factor.

    Returns:
        A compiled Keras Model.
    """
    height, width = board_dimensions
    
    # --- Dynamically Calculate Input Channels ---
    # We need a channel for each piece type for each player, plus one for the turn indicator.
    num_channels = num_piece_types * 2 + 1 
    input_shape = (height, width, num_channels)

    # --- Input Layer ---
    inputs = Input(shape=input_shape)

    # --- Convolutional Body (remains the same) ---
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
               kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual Block
    shortcut = x
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', 
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)

    x_flat = Flatten()(x)

    # --- Policy Head ---
    policy = Dense(32, kernel_regularizer=l2(l2_reg), activation='relu')(x_flat)
    policy_output = Dense(action_space_size, activation='softmax', name='policy_output')(policy)

    # --- Value Head ---
    value = Dense(32, kernel_regularizer=l2(l2_reg), activation='relu')(x_flat)
    value_output = Dense(1, activation='tanh', name='value_output')(value)

    # --- Create and Compile the Model ---
    model = Model(inputs=inputs, outputs=[policy_output, value_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'policy_output': 'categorical_crossentropy',
            'value_output': 'mean_squared_error'
        }
    )
    
    return model

#Used in search function, converts status to value appropriate for UCB formula:
#1 is for win, -1 for loss, 0 if ongoing/draw
def convert_status_to_value(game_status, player_num):
    if game_status == player_num:
        return 1
    elif game_status == -1:
        return 0
    else:
        return -1    

#Main function, running MCTS simulations for a number of times:
def search(root_board, model, num_simulations):

    #Create root node from root_state:
    root = Node(root_board, parent=None, prior_probability=0)

    #Get initial policy/value from network:
    transformed_state = root.board.transform_state()
    input_data = np.expand_dims(transformed_state, axis=0)
    policy, value = model.predict(input_data, verbose = 0)

    # Expand the root node and backpropagate the initial value
    root.expand(policy[0]) 
    root.backpropagate(value[0][0])

    #Run all simulations:
    for i in range(num_simulations):
        node = root
        
        #MCTS 1: Selection
        while not node.is_leaf():
            node = node.select_child()

        #MCTS 2/3: Expansion & Simulation
        # Check if the game is over at this leaf node
        game_status = node.board.game_status
        if game_status != 0: # Game is over (win/loss/draw)
            value = convert_status_to_value(game_status, node.player_to_move)
        else:
            # If game is not over, use the network to get value and expand
            transformed_state = node.board.transform_state()
            input_data = np.expand_dims(transformed_state, axis=0)
            policy, value = model.predict(input_data, verbose = 0)
            value = value[0][0] # Extract scalar value
            node.expand(policy[0])

        #MCTS 4: Backpropagation
        temp_node = node
        while temp_node is not None:
            temp_node.backpropagate(value)
            value *= -1 #Must flip value because players switched
            temp_node = temp_node.parent
    
    #Having run MCTS, use info to return array of win probabilities for moves
        
    valid_moves = root.board.valid_moves
    #Only include children that are valid moves:
    valid_children = {move: child for move, child in root.children.items() if move in valid_moves}

    #If not children are found, give error and return zeros (has never happened)
    if not valid_children:
        print("Warning: MCTS search resulted in no valid moves. Will choose randomly.")
        return np.zeros(root_board.get_action_space_size()) # Return zeroed probs
    
    move_probs = np.zeros(root_board.get_action_space_size(), dtype=np.float32)
    total_visits = sum(child.visit_count for child in root.children.values())
    if total_visits > 0:
        for move, child_node in root.children.items():
            move_probs[move] = child_node.visit_count / total_visits
    else:
        #Fallback for rare cases (has never happened):
        for move in valid_moves:
            move_probs[move] = 1/len(valid_moves)

    return move_probs

#Proceses game history for train function:
def process_game_history(game_history, game_outcome):
    training_samples = []
    for state, move_probs in game_history:
        value = 0
        if game_outcome != -1: # Not a draw
            value = 1 if game_outcome == state.current_player else -1
        
        # Add value into tuple:
        training_samples.append((state, move_probs, value))
    return training_samples

#Main function for training the network:
def train(model_path, exploration_moves, num_iterations=100, num_games_per_iteration=50, mcts_simulations=100):
    """
    Args:
        model_path: Where the file containing the keras model is stored
        exploration_moves: Number of moves to do using temperature based exploration
        num_iterations: How many cycles training to run
        num_games_per_iteration: How many games to play to generate data per cycle
        mcts_simulations: How many simulations for MCTS to run per move
        model_path: Path to save/load the model file.
    """
    state = Board() #Creates empty board
    #Load/create model
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)

        #Compile model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mean_squared_error'
            }
        )
    else:
        print("Creating a new model...")
        model = create_dynamic_model(state.get_board_dimensions(), state.get_action_space_size()) 
        model.summary()

    #Training cycle:
    for i in range(num_iterations):
        print(f"\nITERATION {i+1}/{num_iterations}---------------------------------------------------")
        
        all_training_samples = [] #Stores (state, policy, value) tuples from all games

        #Create training data by playing self:
        for g in range(num_games_per_iteration):
            print(f"  Playing game {g+1}/{num_games_per_iteration}...")
            
            game_history = [] #Stores (state, move_probs) tuples for each game
            state.restart() #restarts game
            player_turn = 1

            move_count = 0

            while True:
                #Use MCTS to find the move probabilities:
                move_probabilities = search(state, model, mcts_simulations)
                move_probabilities = move_probabilities / np.sum(move_probabilities)

                if move_count < exploration_moves:
                    move = np.random.choice(len(move_probabilities), p=move_probabilities)
                else:
                    move = np.argmax(move_probabilities)

                # Store move:
                game_history.append((state, move_probabilities))
                #Make move
                state = state.make_move(move)
                move_count += 1

                #Check game ended:
                game_status = state.game_status

                print(f"Move made by Player {player_turn}. Game status is now: {game_status}")
                state.print_board()

                if game_status != 0: # Game is over
                    #Process the game history now it is over, and add to training samples:
                    game_outcome = game_status
                    training_samples = process_game_history(game_history, game_outcome)
                    all_training_samples.extend(training_samples)
                    break

                # Switch player
                player_turn = 3-player_turn

        #Train model:
        print("\n  Training the model...")
        
        states, policies, values = zip(*all_training_samples)
        #Transform states into the format the network expects
        transformed_states = np.array([s.transform_state() for s in states])
        policies = np.array(policies)
        values = np.array(values)

        #Train the model on the generated data:
        model.fit(
            transformed_states,
            {'policy_output': policies, 'value_output': values},
            epochs=1,
            batch_size=64,
            verbose=1
        )
        #Save model:
        print("  Saving model...")
        model.save(model_path)

        #Clear memory (I had problems with RAM filling up)
        tf.keras.backend.clear_session()
        gc.collect()

#Used to play against AI. human_turn=1 to go first, 2 to go second
#Only difference of machine decisions is it will always choose best move it finds, not 'temperature based'
def play(model_path, human_turn=1):

    state = Board()
    #Load/create model
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)

        #Compile model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mean_squared_error'
            }
        )
    else:
        print("Creating a new model...")
        model = create_dynamic_model(state.get_board_dimensions(), state.get_action_space_size()) 
        model.summary()
            
    game_history = [] #Stores (state, move_probs) tuples for each game
    state.restart() #restarts game
    player_turn = 1

    move_count = 0

    print('---NEW GAME AGAINST COMPUTER---')
    current_player_num = human_turn

    while True:        
        if current_player_num == 1:
            #Get network's idea of what best moves are (for diagnostics):
            transformed_state = state.transform_state()
            input_data = np.expand_dims(transformed_state, axis=0)
            policy, value = model.predict(input_data, verbose = 0)
            print('Computer analysis:')
            print("Moves:", policy[0])
            print("Win probability", value[0])


            print("The current possible moves are:", state.valid_moves)
            move = int(input('Input player move: ')) #COULD ADD SYSTEM SO WORKS IF TEXT ETC INSERTED
        else:
            move_probabilities = search(state, model, 150)
            move = np.argmax(move_probabilities)

        #Make move:
        state = state.make_move(move)
        move_count += 1

        #Check game ended:
        game_status = state.game_status

        print(f"Move made by Player {current_player_num}. Game status is now: {game_status}")
        state.print_board()

        if game_status != 0: # Game is over
            # Process the game history with the final outcome
            game_outcome = game_status
            if game_outcome == -1:
                print('The game ended in a draw.')
            elif game_outcome == 1:
                print('You won!')
            else:
                print('You lost :(')
            break

        # Switch player
        current_player_num = 3-current_player_num


'''
#Example implementation for tic-tac-toe:
from tictactoe_game import Board
train('mtictactoe_model.h5', 5, num_iterations=100, num_games_per_iteration =40, mcts_simulations=60)
play('my_tictactoe_model.h5')
'''

#Example implementation of 4-in-a-row
from connect4_game import Board
train('4inarow_model.h5', 30, num_iterations=100, num_games_per_iteration = 40, mcts_simulations=100)
play(model_path='4inarow_model.h5')