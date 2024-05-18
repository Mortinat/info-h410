import minimax as mm
from common import MINIMAX, EMPTY, ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH
import copy
import math


with open("Test_L2_R1", "r") as f:
    print(ROW_COUNT, COLUMN_COUNT)
    empty_board = [[EMPTY for _ in range(ROW_COUNT)] for _ in range(COLUMN_COUNT)]
    for line in f.readlines():
        print(line)
        board = mm.BoardMinimax(board=copy.deepcopy(empty_board), turn=1, rounds=0)
        split_line = line.split(" ")
        for i in split_line[0]:
            row = board.get_next_open_row(int(i)-1)
            board.drop_piece(row, int(i)-1)
        print(mm.negamax(board, 10000, -math.inf, math.inf))
