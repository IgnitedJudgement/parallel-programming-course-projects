# Terminal command
# mpiexec -n [n] python .\main.py [n] where n represents the number of processes

import sys
import time
from typing import List, Optional, Tuple, Any

import numpy as np
from mpi4py import MPI

# Global variables
N_PHILOSOPHERS: int = int(sys.argv[1])
MAX_IDLE_TIME_IN_SECONDS: int = 3

# MPI Tags
TAG_SEND: int = 0
TAG_RECEIVE: int = 1


class Philosopher:
    def __init__(self, rank: int) -> None:
        self.rank: int = rank
        self.left_fork: Optional[Fork] = None
        self.right_fork: Optional[Fork] = None
        self.requests: List[Tuple[int, int]] = []

        if self.rank == 0:
            self.left_fork = Fork(id=0)
            self.right_fork = Fork(id=1)
        elif self.rank < N_PHILOSOPHERS - 1:
            self.right_fork = Fork(id=rank + 1)

    def eat(self) -> None:
        if self.left_fork is not None:
            self.left_fork.dirty = True

        if self.right_fork is not None:
            self.right_fork.dirty = True

    def get_fork_with_id(self, id: int) -> Optional["Fork"]:
        if self.left_fork is not None and self.left_fork.id == id:
            return self.left_fork

        if self.right_fork is not None and self.right_fork.id == id:
            return self.right_fork

        return None

    def is_fork_dirty_by_id(self, id) -> bool:
        fork = self.get_fork_with_id(id)

        return fork is not None and fork.dirty

    def add_fork(self, fork) -> None:
        if self.left_fork is None:
            self.left_fork = fork
        else:
            self.right_fork = fork

    def remove_fork_with_id(self, id) -> None:
        if self.left_fork is not None and self.left_fork.id == id:
            self.left_fork = None

        if self.right_fork is not None and self.right_fork.id == id:
            self.right_fork = None


class Fork:
    def __init__(self, id: int, dirty: bool = True):
        self.id: int = id
        self.dirty: bool = dirty

    def __str__(self) -> str:
        return f"id: {self.id}, dirty: {self.dirty}"


# MPI variables
comm: Any = MPI.COMM_WORLD
status: Any = MPI.Status()
rank: int = comm.rank
size: int = comm.size

# Neighbour indices
LEFT_NEIGHBOUR: int = (rank - 1) % N_PHILOSOPHERS
RIGHT_NEIGHBOUR: int = (rank + 1) % N_PHILOSOPHERS

philosopher: Philosopher = Philosopher(rank)

if size != N_PHILOSOPHERS:
    if rank == 0:
        print(f"Number of processes ({size}) does not match the number of philosophers ({N_PHILOSOPHERS})!")
    exit(1)


def custom_print(text, rank):
    print("{}{}".format("\t" * rank * 2, text), flush=True)


def fork_status(rank) -> None:
    custom_print(f"P{rank}, forks: ({philosopher.left_fork} | {philosopher.right_fork})", rank)


if __name__ == "__main__":
    while True:
        # Think
        custom_print(f"P{rank} is thinking", rank)
        sleep_time: float = np.random.random_sample() * MAX_IDLE_TIME_IN_SECONDS

        while sleep_time > 0:
            time.sleep(1)
            sleep_time -= 1

            if comm.iprobe(tag=TAG_RECEIVE, status=status):
                send_fork_id = comm.recv(tag=TAG_RECEIVE, status=status)

                send_to_dest: int = status.Get_source()

                comm.send(send_fork_id, dest=send_to_dest, tag=TAG_SEND)
                philosopher.remove_fork_with_id(send_fork_id)

                custom_print(f"P{rank} sending fork {send_fork_id} to P{send_to_dest}", rank)

        # Wait for both forks to become available
        while philosopher.left_fork is None or philosopher.right_fork is None:
            # Request a required fork (priority: [left, right])
            receive_from_dest: int = LEFT_NEIGHBOUR if philosopher.left_fork is None else RIGHT_NEIGHBOUR
            receive_fork_id: int = rank if philosopher.left_fork is None else RIGHT_NEIGHBOUR
            comm.send(receive_fork_id, dest=receive_from_dest, tag=TAG_RECEIVE)

            # Repeat until the required fork is received
            while True:
                send_fork_id = comm.recv(status=status)
                send_to_dest = status.Get_source()

                if status.tag == TAG_SEND:
                    custom_print(f"P{rank} received fork {receive_fork_id} from P{receive_from_dest}", rank)

                    # Received fork
                    philosopher.add_fork(Fork(receive_fork_id, dirty=False))
                    break
                else:

                    # If a fork is requested
                    if philosopher.is_fork_dirty_by_id(send_fork_id):
                        # If fork is dirty
                        comm.send(send_fork_id, dest=send_to_dest, tag=TAG_SEND)
                        philosopher.remove_fork_with_id(send_fork_id)

                        custom_print(f"P{rank} sending fork {send_fork_id} to P{send_to_dest}", rank)

                    else:
                        # If fork is clean
                        philosopher.requests.append((send_to_dest, send_fork_id))

        # Eat
        custom_print(f"P{rank} is eating", rank)

        eat_time: float = np.random.random_sample() * MAX_IDLE_TIME_IN_SECONDS
        time.sleep(eat_time)

        # Update forks (clean -> dirty)
        philosopher.eat()

        # Respond to existing requests
        for dest, fork_id in philosopher.requests:
            comm.send(fork_id, dest=dest, tag=TAG_SEND)
            philosopher.remove_fork_with_id(fork_id)

            custom_print(f"P{rank} sending fork {fork_id} to P{dest}", rank)
