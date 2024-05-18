import minimax as mm
from common import MINIMAX, EMPTY, ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH
import copy
import math
import time

time_list = []
with open("Test_L2_R1", "r") as f:
    empty_board = [[EMPTY for _ in range(ROW_COUNT)] for _ in range(COLUMN_COUNT)]
    for line in f.readlines():
        board = mm.BoardMinimax(board=copy.deepcopy(empty_board), turn=1, rounds=0)
        split_line = line.split(" ")
        for i in split_line[0]:
            row = board.get_next_open_row(int(i)-1)
            board.drop_piece(row, int(i)-1)
        start = time.time()
        result = mm.negamax(board, 10000, -math.inf, math.inf)
        stop = time.time()
        time_list.append(stop-start)
        print(sum(time_list)/len(time_list))
