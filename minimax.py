from bot import Bot
from common import ROW_COUNT, COLUMN_COUNT, MINIMAX
import math


COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]
PRECOMPUTED_TOP_MASKS = [(1 << (ROW_COUNT - 1)) << col * (ROW_COUNT + 1) for col in range(COLUMN_COUNT)]
BOTTOM_MASK = [1 << col * (ROW_COUNT + 1) for col in range(COLUMN_COUNT)]
COLUMN_MASK = [((1 << ROW_COUNT) - 1) << col * (ROW_COUNT + 1) for col in range(COLUMN_COUNT)]
# use this to change the order of the columns
# for i in range(COLUMN_COUNT):
#     self.columnOrder[i] = COLUMN_COUNT//2 + (1 - 2*(i % 2))*(i+1)//2


def negamax(board, depth, alpha, beta):
    is_terminal = board.is_terminal_node()

    if is_terminal:
        return (None, 0)

    column = None
    for col in range(COLUMN_COUNT):
        col = COLUMN_ORDER[col]
        if board.can_play(col):
            column = col
            b_copy = board.copy()
            if b_copy.winning_move(col):
                return col, ((ROW_COUNT * COLUMN_COUNT) + 1 - b_copy.rounds)/2

    max_score = ((ROW_COUNT * COLUMN_COUNT) - 1 - board.rounds)/2
    if (beta > max_score):
        beta = max_score
        if alpha >= beta:
            return column, beta

    for col in range(COLUMN_COUNT):
        if board.can_play(col):
            b_copy = board.copy()
            b_copy.play(col)
            score = -negamax(b_copy, depth - 1, -beta, -alpha)[1]
            if score >= beta:
                return col, score
            if score > alpha:
                alpha = score
    return column, alpha


class BoardMinimax:
    def __init__(self, board, turn, rounds):
        self.rounds = rounds
        if not board:
            self.position = 0
            self.mask = 0
            return

        self.position = []
        for col in range(COLUMN_COUNT):
            for row in range(ROW_COUNT):
                if board[col][row] == 1:
                    self.position.append('1')
                elif board[col][row] == -1:
                    self.position.append('-1')
                else:
                    self.position.append('0')
            self.position.append('0')
        self.position = self.position[::-1]
        self.mask = int(''.join(self.position).replace('-1', '1'), 2)
        self.position = int(''.join(self.position).replace('-1', '0'), 2)

        # PRINT BOARD
        # test = list(str(bin(self.position))[2:])[::-1]
        # for i in range(49-len(test)):
        #     test.append('0')
        # for start in range(5, -1, -1):
        #     indices = [start + 7 * i for i in range(7)]
        #     values = [test[index] for index in indices]
        #     print(''.join(values))
        # print()

    def copy(self):
        new_board = BoardMinimax([], 0, 0)
        new_board.position = self.position
        new_board.mask = self.mask
        new_board.rounds = self.rounds
        return new_board

    def can_play(self, col):
        return (self.mask & PRECOMPUTED_TOP_MASKS[col]) == 0

    def play(self, col):
        self.position ^= self.mask
        self.mask |= self.mask + BOTTOM_MASK[col]
        self.rounds += 1

    def winning_move(self, col):
        pos = self.position
        pos |= (self.mask + BOTTOM_MASK[col]) & COLUMN_MASK[col]
        return self.alignment(pos)

    def alignment(self, pos):
        # Horizontal check
        m = pos & (pos >> (ROW_COUNT + 1))
        if m & (m >> (2 * (ROW_COUNT + 1))):
            return True

        # Diagonal check (bottom-left to top-right)
        m = pos & (pos >> ROW_COUNT)
        if m & (m >> (2 * ROW_COUNT)):
            return True

        # Diagonal check (top-left to bottom-right)
        m = pos & (pos >> (ROW_COUNT + 2))
        if m & (m >> (2 * (ROW_COUNT + 2))):
            return True

        # Vertical check
        m = pos & (pos >> 1)
        if m & (m >> 2):
            return True

        return False

    def is_terminal_node(self):
        return self.rounds == ROW_COUNT * COLUMN_COUNT or self.alignment(self.position)


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
