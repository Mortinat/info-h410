from bot import Bot
import copy
from common import MINIMAX, EMPTY, ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH
import random
import math
import inspect


def negamax(board, depth, alpha, beta):

    valid_locations = board.get_valid_locations()
    is_terminal = board.is_terminal_node()

    if depth == 0 or is_terminal:
        return (None, 0)

    for col in valid_locations:
        row = board.get_next_open_row(col)
        b_copy = copy.deepcopy(board)
        b_copy.drop_piece(row, col)
        if b_copy.winning_move(-b_copy.turn):
            return col, (ROW_COUNT * COLUMN_COUNT + 1 - b_copy.rounds)/2
    column = valid_locations[0]
    max_score = (ROW_COUNT * COLUMN_COUNT - 1 - b_copy.rounds)/2
    if (beta > max_score):
        beta = max_score
        if alpha >= beta:
            return column, beta

    for col in valid_locations:
        row = board.get_next_open_row(col)
        b_copy = copy.deepcopy(board)
        b_copy.drop_piece(row, col)
        score = -negamax(b_copy, depth - 1, -beta, -alpha)[1]
        if score >= beta:
            return col, score
        if score > alpha:
            alpha = score
    return column, alpha


class BoardMinimax:
    def __init__(self, board, turn, rounds):
        self.board = board
        self.turn = turn
        self.rounds = rounds

    def get_valid_locations(self):
        """
        Returns all the valid columns where the player can play, aka the columns
        that are not full

        :return: list of all valid column indices
        """
        free_cols = [i for i, x in enumerate(self.board) if x[-1] == 0]
        if len(free_cols) == 0:
            return None
        return free_cols

    def drop_piece(self, row, col):
        """
        Drop a piece in the board at the specified position
        :param row: one of the row of the board
        :param col: one of the column of the board
        """
        self.board[col][row] = self.turn
        self.rounds += 1
        self.turn = -self.turn

    def get_next_open_row(self, col):
        """
        Return the first row which does not have a piece in the specified column (col)
        :param col: one of the column of the board
        :return: row number
        """
        for r in range(ROW_COUNT):
            if self.board[col][r] == 0:
                return r

    def winning_move(self, piece):
        """
        Check if the game has been won
        :param board: board with all the pieces that have been placed
        :param piece: 1 or -1 depending on whose turn it is
        """
        # Check horizontal locations for win
        # print(self.board)
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (
                    self.board[c][r] == piece
                    and self.board[c + 1][r] == piece
                    and self.board[c + 2][r] == piece
                    and self.board[c + 3][r] == piece
                ):
                    return True

        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (
                    self.board[c][r] == piece
                    and self.board[c][r + 1] == piece
                    and self.board[c][r + 2] == piece
                    and self.board[c][r + 3] == piece
                ):
                    return True

        # Check positively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (
                    self.board[c][r] == piece
                    and self.board[c + 1][r + 1] == piece
                    and self.board[c + 2][r + 2] == piece
                    and self.board[c + 3][r + 3] == piece
                ):
                    return True

        # Check negatively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (
                    self.board[c][r] == piece
                    and self.board[c + 1][r - 1] == piece
                    and self.board[c + 2][r - 2] == piece
                    and self.board[c + 3][r - 3] == piece
                ):
                    return True
        return False

    def is_terminal_node(self):
        """
        Determines wheter the game is finished or not
        :param board: board with all the pieces that have been placed
        :return: boolean that determines wheter the game is finish or not
        """
        return (
            self.winning_move(self.turn * -1)
            or self.winning_move(self.turn)
            or self.get_valid_locations() is None
        )


class MiniMax(Bot):
    """
    This class is responsible for the Minimax algorithm.
    At each depth, the algorithm will simulate up to 7 boards, each having a piece that has been dropped in a free column. So with depth 1, we will have 7 boards to analyse, with depth 2 : 49 ,...
    Through a system of reward each board will be attributed a score. The Minimax will then either try to minimise or maximise the rewards depending on the depth (odd or even). Indeed, because we are using multiple
    depth, the minimax algorithm will simulate in alternance the possible moves of the current player and the ones of the adversary (creating Min nodes and max nodes). The player that needs to decide where to
    drop a piece on the current board is considered as the maximising player, hence trying to maximise the reward when a max nodes is encountered. The algorithm will also consider that the adversary plays as good as possible (with
    the information available with the depth chosen) and hence try to minimise the reward when possible (minimizing player).
    So after creating all the boards of the tree, at each depth, a board will be selected based on the reward and on the type of nodes (min or max node) starting from the bottom of the tree.
    The final choice is made based on the 7 boards possible with the score updated through the reward procedure describe above.
    Note that the larger the depth, the slower the execution.
    In order to avoid unnecessary exploration of boards, an alpha beta pruning has 
        print(self._game._turn, self._game._round)been implemented.
    """

    def __init__(self, game, depth, pruning=True):
        super().__init__(game, bot_type=MINIMAX, depth=depth, pruning=pruning)

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, pruning):
        """
        Main function of minimax, called whenever a move is needed.
        Recursive function, depth of the recursion being determined by the parameter depth.
        :param depth: number of iterations the Minimax algorith will run for
            (the larger the depth the longer the algorithm takes)
        :alpha: used for the pruning, correspond to the lowest value of the range values of the node
        :beta: used for the pruning, correspond to the hihest value of the range values of the node
        :maximizingPlayer: boolean to specify if the algorithm should maximize or minimize the reward
        :pruning: boolean to specify if the algorithm uses the pruning
        :return: column where to place the piece
        """
        board = BoardMinimax(self._game._board, -self._game._turn, self._game._round)
        return negamax(board, depth, -math.inf, math.inf)
