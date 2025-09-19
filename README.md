# AlphaZero-Inspired-Simple-Game-Agent
An AI model inspired by Google’s AlphaZero to master perfect information games using a Tensorflow neural network and a Monte Carlo Tree Search.

## Summary
To deepen my understanding of machine learning and neural networks, I created this program which replicates the principles of DeepMind's AlphaZero. The system is designed to be game-agnostic, meaning with sufficient training it is capable of mastering perfect information (generalised) board games - this encompasses a large class of common games, such as tic-tact-toe, four-in-a-row, chess, checkers, and go. It achieves this by self-play and a Monte Carlo tree search to generate data and iteratively improve the neural network.

## How it works
The learning process is a continuous cycle:
1.  **Self-Play:** The model plays thousands of games against itself, guided by a Monte Carlo tree search to efficiently evaluate moves.
2.  **Data Generation:** The outcomes and search probabilities from these games are stored as training data.
3.  **Training:** The model is trained on this data. Its goal is to better predict game outcomes (the value head) and the MCTS search results (the policy head).
4.  **Evaluation:** This becomes the "best model" for the next generation of self-play, constantly improving the quality of the data and its own performance.

## Tech Stack
1. Python 3.10 - programming language for the entire project
2. TensorFlow 2.12 - the software library used to create and train the neural network
3. NumPy - for handling the data
4. WSL2 (Ubuntu) - for the development environment

## How to run
Python 3.10, NumPy, and TensorFlow 2.12 are required to run the file main.py. I also used WSL2, but you might want to run on Linux, MacOS, or Windows natively.
1. Download the file main.py, along with any specific game files you want to try running
2. If you would like to try training the model on your own game, you will have to create and import a file with the Board class shown in my example game files, with all the functions that are included there. Note: you will have to consider how the neural network's output is mapped to a move in the game. This is quite simple in games like tic-tac-toe or four-in-a-row, but could get quite complicated with a game like chess where there are thousands of possible moves.
3. At the end of the file main.py, call the functions play() or train() with the parameters you require.

## Future improvements
1. Implement the framework for a more complex game like chess, which is what this sort of solution is specialised for. I haven't done this yet because I don't have the enormous computing power necessary to train such a model.
2. There are huge opportunities for optimization: for example the MCTS could use batching when getting model predictions; previously explored nodes currently get deleted due to the program's structure, but when a move is made to this node this data should be kept; the program can be optimized for specific games.
3. Create a better GUI using Pygame or Tkinter.
