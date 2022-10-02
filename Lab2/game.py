import sys
import time
import copy
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from mpi4py import MPI
from board import Board

# Board constants
BOARD_WIDTH: int = 7
BOARD_HEIGHT: int = 6
SEQUENCE_LENGTH: int = 4

# Processing constants
DEPTH: int = 6 if len(sys.argv) < 2 else int(sys.argv[1])
DEPTH_MASTER: int = 2
DEPTH_WORKER: int = DEPTH - DEPTH_MASTER

# Player constants
# CAUTION - board only supports {1, 2} as indices
PLAYER_HUMAN: int = 1
PLAYER_CPU: int = 2

# Debugging
DEBUG: int = 0

# MPI constants
RANK_MASTER: int = 0

# MPI tags
TAG_STOP: int = 0
TAG_START: int = 1
TAG_REQUEST: int = 3
TAG_TASK: int = 4
TAG_WAIT: int = 5
TAG_RESULT: int = 6

# MPI
comm = MPI.COMM_WORLD
status = MPI.Status()
rank: int = comm.rank
size: int = comm.size

# Variables
tasks: List = []
task_id: int = 0


def get_next_player(player: int) -> int:
    return PLAYER_HUMAN if player == PLAYER_CPU else PLAYER_CPU


def evaluate(current: Board, player: int, depth: int,
             moves_result_list: Optional[List[Dict[str, Any]]] = None) -> float:
    if current.game_end():
        return 1 if player == PLAYER_CPU else -1

    if depth <= 0:
        return 0

    next_player: int = get_next_player(player)

    n_moves: int = 0
    total: float = 0

    all_win: bool = True
    all_lose: bool = True

    for col in range(current.WIDTH):
        if current.move_legal(col):
            n_moves += 1

            current_moves: List[Dict[str, Any]] = current.moves.copy()
            current.move(col, next_player)

            result: float

            if moves_result_list is not None and depth == DEPTH_WORKER:
                result = get_result_for_moves(moves_result_list, current_moves)['result']
            else:
                result = evaluate(current, next_player, depth - 1, moves_result_list=moves_result_list)

            current.undo_move()

            if result < 1:
                all_win = False

            if result > -1:
                all_lose = False

            if result == 1 and next_player == PLAYER_CPU:
                return 1

            if result == -1 and next_player == PLAYER_HUMAN:
                return -1

            total += result

    if all_win:
        return 1

    if all_lose:
        return -1

    return round(total / n_moves, 8)


def get_input(current: Board, player: int) -> int:
    while True:
        try:
            col: int = int(input(f"Player: {player}, make your selection: "))
        except ValueError:
            print(f"Invalid input, please try again!")
            continue
        else:
            if col < 0 or col > current.WIDTH - 1:
                print(f"Index out of bounds, please try again!")
                continue
            if not current.move_legal(col):
                print(f"Invalid location, please try again!")
                continue
            return col


def notify_workers(tag: int) -> None:
    for dest in range(1, size):
        comm.send(0, dest=dest, tag=tag)


def create_tasks(board: Board, depth: int, player: int, path: Optional[List[int]] = None,
                 tasks: Optional[List[Dict[str, Any]]] = None) -> None:
    if depth == 0:
        current_board: Board = copy.deepcopy(board)
        current_player: int = player

        for col in path:
            if current_board.move_legal(col):
                current_board.move(col, current_player)
                current_player = get_next_player(current_player)

        tasks.append({'board': current_board, 'player': player, 'moves': current_board.moves.copy(), 'result': None,
                      'active': False})

    else:
        if path is None:
            path = []

        for col in range(BOARD_WIDTH):
            new_path: List[int] = path.copy()
            new_path.append(col)

            create_tasks(board, depth - 1, player, new_path, tasks)


def get_task(tasks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for task in tasks:
        if not task['active']:
            task['active'] = True
            return task

    return None


def update_task(tasks: List[Dict[str, Any]], task: Dict[str, Any], moves_result_list: List[Dict[str, Any]]) -> None:
    for index in range(len(tasks)):
        tmp: Dict[str, Any] = tasks[index]

        if tmp['moves'] == task['moves']:
            tmp['result'] = task['result']
            moves_result_list.append({'moves': task['moves'], 'result': task['result']})
            break


def get_result_for_moves(moves_result_list: List[Dict[str, Any]], moves: List[Dict[str, Any]]) -> \
        Optional[Dict[str, Any]]:
    for item in moves_result_list:
        if item['moves'] == moves:
            return item

    return None


def do_job(depth: int, player: int, moves_result_list: Optional[List[Dict[str, float]]] = None) -> Tuple[float, int]:
    current_depth: int = depth
    best_result: float = -1
    best_col: int = 0

    while best_result == -1 and current_depth > 0:
        best_result = -1
        best_col = 0

        for col in range(board.WIDTH):
            if board.move_legal(col):
                board.move(col, player)
                result: float = evaluate(board, player, current_depth - 1, moves_result_list=moves_result_list)
                board.undo_move()

                if (result > best_result) or (
                        result == best_result and np.random.random_sample() < 0.5):
                    best_result = result
                    best_col = col

        current_depth = int(current_depth / 2)

    return best_result, best_col


if __name__ == "__main__":
    # Master
    if rank == RANK_MASTER:
        board: Board = Board(BOARD_WIDTH, BOARD_HEIGHT, SEQUENCE_LENGTH)
        depth: int = DEPTH

        # Until the game is over
        while True:
            # Human turn
            col: int = get_input(board, PLAYER_HUMAN)
            board.move(col, PLAYER_HUMAN)
            board.print_board()

            if board.game_end():
                notify_workers(TAG_STOP)
                print(f"Player {PLAYER_HUMAN} has won the game!", flush=True)
                break

            # CPU turn
            print(f"CPU turn", flush=True)

            start_time: float = time.time()

            # Start the workers
            notify_workers(TAG_START)

            # Create tasks (tasks variable is defined at the start of the program)
            create_tasks(board, DEPTH_MASTER, PLAYER_CPU, tasks=tasks)

            # Create moves results data structure
            moves_result_list: Optional[List[Dict[str, float]]] = [] if size > 1 else None

            active_workers: int = size - 1

            # If tasks exist, delegate them
            while active_workers > 0:
                message: Union[int, Dict[str, Any]] = comm.recv(status=status)

                if DEBUG:
                    print(f"Master has received a message: {message}, from process {status.source}", flush=True)

                # If a worker is requesting a task
                if status.tag == TAG_REQUEST:
                    task: Dict[str, Any] = get_task(tasks)

                    # If an inactive task exists
                    if task is not None:
                        if DEBUG:
                            print(f"Master is sending process {status.source} a task: {task}", flush=True)

                        comm.send(task, dest=status.source, tag=TAG_TASK)

                    # If there are no more inactive tasks
                    else:
                        active_workers -= 1
                        comm.send(0, dest=status.source, tag=TAG_WAIT)

                        if DEBUG:
                            print(f"Master is telling process {status.source} to wait", flush=True)

                # If a worker is sending back a result
                elif status.tag == TAG_RESULT:
                    update_task(tasks, message, moves_result_list)

                else:
                    raise ValueError(f"Encountered unexpected tag ({status.tag})!")

            # Process results and make a move
            best_result: float
            best_col: int

            best_result, best_col = do_job(DEPTH, PLAYER_CPU, moves_result_list)

            print(f"Time elapsed: {(time.time() - start_time):.4f} seconds\n", flush=True)

            board.move(best_col, PLAYER_CPU)
            board.print_board()

            if board.game_end():
                notify_workers(TAG_STOP)
                print(f"Player {PLAYER_CPU} has won the game!", flush=True)
                break

    # Worker
    else:
        # Work until the end of the game
        while True:
            # Wait for message
            message: int = comm.recv(source=RANK_MASTER, status=status)

            # Stop worker if game is over
            if status.tag == TAG_STOP:
                if DEBUG:
                    print(f"Process {rank} has stopped", flush=True)

                break

            # Start processing tasks
            elif status.tag == TAG_START:
                if DEBUG:
                    print(f"Process {rank} has started", flush=True)

            else:
                raise ValueError(f"Encountered unexpected tag ({status.tag})!")

            # While available tasks exist, process them
            while True:
                # Request a task
                comm.send(0, dest=RANK_MASTER, tag=TAG_REQUEST)

                if DEBUG:
                    print(f"Process {rank} has requested a task", flush=True)

                # Receive a task
                message: Dict[str, Any] = comm.recv(source=RANK_MASTER, status=status)

                if DEBUG:
                    print(f"Process {rank} received a message: {message}", flush=True)

                # There are no more available tasks to process at this moment
                if status.tag == TAG_WAIT:
                    if DEBUG:
                        print(f"Process {rank} is waiting", flush=True)

                    break

                elif status.tag == TAG_TASK:
                    board: Board = message['board']
                    player: int = message['player']

                    # Update message with result
                    best_result: float
                    best_col: int

                    best_result, best_col = do_job(DEPTH_WORKER, player)

                    message['result'] = best_result
                    message['moves'] = board.moves

                    # Return message
                    comm.send(message, dest=RANK_MASTER, tag=TAG_RESULT)
                    if DEBUG:
                        print(f"Process {rank} is sending a message: {message}", flush=True)

                else:
                    raise ValueError(f"Encountered unexpected tag ({status.tag})!")
