from enum import Enum
import numpy as np


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def prepare_board():
    board = [[-1 for j in range(12)] for i in range(4)]
    board[3][1:-1] = [-100] * 10
    board[3][0] = 0
    board[3][-1] = 0

    for i in range(len(board)):
        print(board[i])
    return board


def prepare_actions(board):
    actions_states = {}
    for i in range(len(board)):
        for j in range(len(board[0])):
            actions = []
            if i != 0:
                actions.append(Actions.UP)
            if i != len(board) - 1:
                actions.append(Actions.DOWN)

            if j != 0:
                actions.append(Actions.LEFT)
            if j != len(board[0]) - 1:
                actions.append(Actions.RIGHT)

            # print(i, j, actions)
            actions_states.update({(i, j): actions})

    return actions_states


def move(i, j, action):
    if action == Actions.UP:
        return i - 1, j
    if action == Actions.DOWN:
        return i + 1, j

    if action == Actions.LEFT:
        return i, j - 1
    if action == Actions.RIGHT:
        return i, j + 1


def calculate_max_reward(board, actions_states, state):
    max_state = ()
    max_reward = -9999999999

    for a in actions_states[state]:
        next_state = tuple(move(state[0], state[1], a))
        reward = board[next_state[0]][next_state[1]]
        if reward > max_reward:
            max_reward = reward
            max_state = next_state
    return max_reward


def chose_A(use_max, actions_states, current_state, board):
    A = None
    max_reward = -9999999999
    if use_max == 1:
        for a in actions_states[current_state]:
            next_state = tuple(move(current_state[0], current_state[1], a))
            reward = board[next_state[0]][next_state[1]]
            if reward > max_reward:
                max_reward = reward
                A = a

    else:
        A = np.random.choice(actions_states[current_state])

    return A


def sarsa():
    board = prepare_board()
    actions_states = prepare_actions(board)

    e = 0.1
    q_s_a = [[1 for j in range(12)] for i in range(4)]
    # q_s_a = board.copy()
    q_s_a[3][11] = 0

    for episode in range(1000):
        if episode % 100 == 0:
            print("episode: ", episode)

        current_state = (3, 0)
        use_max = np.random.choice([0, 1], p=[e, 1 - e])
        A_1 = chose_A(use_max, actions_states, current_state, board)

        n = 0
        while current_state != (3, 11):
            n += 1
            S_2 = move(current_state[0], current_state[1], A_1)
            R = board[S_2[0]][S_2[1]]

            # if R == -100:
            #     S_2 = (3, 0)

            use_max = np.random.choice([0, 1], p=[e, 1 - e])
            A_2 = chose_A(use_max, actions_states, S_2, board)

            q_s_a_value_2 = q_s_a[S_2[0]][S_2[1]]
            q_s_a_value = q_s_a[current_state[0]][current_state[1]]
            q_s_a[current_state[0]][current_state[1]] = \
                q_s_a_value + 1 / n * (R + 0.9 * q_s_a_value_2 - q_s_a_value)

            current_state = S_2
            A_1 = A_2

    for i in range(len(q_s_a)):
        print(["%.2f" % i for i in q_s_a[i]])


start_position = (3, 0)


def q_learn():
    board = prepare_board()
    actions_states = prepare_actions(board)

    e = 0.1
    q_s_a = [[-999 for _ in range(12)] for _ in range(4)]
    q_s_a[3][11] = 0

    for episode in range(100):
        if episode % 100 == 0:
            print("episode: ", episode)

        current_state = start_position

        alpha = 1.0
        n = 0
        while current_state != (3, 11):
            n += 1
            use_max = np.random.choice([0, 1], p=[e, 1 - e])

            max_state = ()
            max_reward = -9999999999
            if use_max:
                for a in actions_states[current_state]:
                    next_state = tuple(move(current_state[0], current_state[1], a))
                    reward = board[next_state[0]][next_state[1]]
                    if reward > max_reward:
                        max_reward = reward
                        max_state = next_state

            else:
                a = np.random.choice(actions_states[current_state])
                max_state = tuple(move(current_state[0], current_state[1], a))
                max_reward = board[max_state[0]][max_state[1]]

            if max_reward == -100:
                max_state = start_position

            max_next_reward = calculate_max_reward(q_s_a, actions_states, max_state)

            alpha = pow(n, -0.1)
            q_s_a_value = q_s_a[current_state[0]][current_state[1]]
            delta = alpha * (max_reward + 0.9 * max_next_reward - q_s_a_value)
            q_s_a[current_state[0]][current_state[1]] += delta

            current_state = max_state

    for i in range(len(q_s_a)):
        print(["%.2f" % i for i in q_s_a[i]])


if __name__ == "__main__":
    q_learn()
    # sarsa()
