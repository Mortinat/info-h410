import minimax as mm
from common import MINIMAX, EMPTY, ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH
import copy
import math
import time

time_list = []
with open("Test_L1_R1", "r") as f:
    empty_board = [[EMPTY for _ in range(ROW_COUNT)] for _ in range(COLUMN_COUNT)]
    for index, line in enumerate(f.readlines()):
        board = mm.BoardMinimax(board=copy.deepcopy(empty_board), turn=1, rounds=0)
        split_line = line.split(" ")
        for i in split_line[0]:
            board.play(int(i)-1)
        start = time.time()
        result = mm.solve(board, 10000)
        stop = time.time()
        time_list.append(stop-start)
        if index > 50:
            break
    print(sum(time_list)/len(time_list))
