from enum import Enum
from typing import List, Dict, Any, Set

import numpy as np


class Player(Enum):
    PLAYER_CPU: int = 1
    PLAYER_HUMAN: int = 2


players: Set[int] = set(player.value for player in Player)


class Board:
    def __init__(self, width: int, height: int, sequence_length: int = 4) -> None:
        self.WIDTH: int = width
        self.HEIGHT: int = height
        self.SEQUENCE_LENGTH: int = sequence_length
        self.data: np.ndarray = self.init_data()
        self.moves: List[Dict[str, Any]] = list()

        if (self.WIDTH < self.SEQUENCE_LENGTH) or (self.HEIGHT < self.SEQUENCE_LENGTH):
            raise ValueError(f"Sequence length has to be less than width and height of the board!")

    def init_data(self):
        return np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int16)

    def print_board(self) -> None:
        print(f"{self.data}\n", flush=True)

    def move_legal(self, col: int) -> bool:
        return self.get_next_open_row(col) != -1

    def move(self, col: int, player: int) -> None:
        if not self.move_legal(col):
            raise ValueError(f"The given column ({col}) is full!")

        if player not in players:
            raise ValueError(f"The given player ({player}) is not supported!")

        row: int = self.get_next_open_row(col)

        self.data[row][col] = player
        self.moves.append({'player': player, 'position': (row, col)})

    def undo_move(self) -> None:
        row: int
        col: int

        row, col = self.moves.pop()['position']
        self.data[row][col] = 0

    def get_next_open_row(self, col: int) -> int:
        for row in range(self.HEIGHT - 1, -1, -1):
            if self.data[row][col] == 0:
                return row

        return -1

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.HEIGHT and 0 <= col < self.WIDTH

    def game_end(self) -> bool:
        if len(self.moves) < self.SEQUENCE_LENGTH:
            return False

        player: int = self.moves[-1]['player']

        last_row: int
        last_col: int

        last_row, last_col = self.moves[-1]['position']

        directions: List = [[[-1, 0], 0, True],
                            [[1, 0], 0, True],
                            [[0, -1], 0, True],
                            [[0, 1], 0, True],
                            [[-1, -1], 0, True],
                            [[1, 1], 0, True],
                            [[-1, 1], 0, True],
                            [[1, -1], 0, True]]

        for x in range(self.SEQUENCE_LENGTH):
            for direction in directions:
                row: int = last_row + (direction[0][0] * (x + 1))
                col: int = last_col + (direction[0][1] * (x + 1))

                if direction[2] and self.in_bounds(row, col) and self.data[row][col] == player:
                    direction[1] += 1
                else:
                    direction[2] = False

        for i in range(0, 2 * self.SEQUENCE_LENGTH - 1, 2):
            if directions[i][1] + directions[i + 1][1] >= self.SEQUENCE_LENGTH - 1:
                return True

        return False
