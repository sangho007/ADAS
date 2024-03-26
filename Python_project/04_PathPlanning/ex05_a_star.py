import numpy as np
import math
import matplotlib.pyplot as plt
import random
from map_1 import map

show_animation = True


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.f = 0

    def __eq__(self, other):
        if self.position == other.position:
            return True


def get_action():
    # action = [dx, dy, cost]
    # action_set = [action1, action2, ...]
    action_set = [
        [-1, -1, np.sqrt(2)],
        [1, -1, np.sqrt(2)],
        [-1, 1, np.sqrt(2)],
        [1, 1, np.sqrt(2)],
        [0, -1, 1],
        [0, 1, 1],
        [1, 0, 1],
        [-1, 0, 1]
    ]
    return action_set


def collision_check(omap, node_pos):
    # Check if node position == obstacle position
    col = False
    obstacle = [(x) for x in zip(*omap)]

    if node_pos in obstacle:
        col = True

    return col  # True or False


def Astar(start, goal, map_obstacle):
    start_node = Node(None, start)
    goal_node = Node(None, goal)

    open_list = []
    closed_list = []

    open_list.append(start_node)
    while open_list is not None:
        # Find node with lowest cost
        cur_node = open_list[0]
        cur_index = 0
        for index, node in enumerate(open_list):
            if node.f < cur_node.f:
                cur_node = node
                cur_index = index

        # If goal, return optimal path
        if cur_node.position == goal_node.position:
            opt_path = []
            node = cur_node
            while node is not None:
                opt_path.append(node.position)
                node = node.parent
            return opt_path[::-1]

        # If not goal, move from open list to closed list
        open_list.pop(cur_index)
        closed_list.append(cur_node)

        # Generate child candidate
        action_set = get_action()
        for action in action_set:
            child_candidate_position = (cur_node.position[0] + action[0], cur_node.position[1] + action[1])
            # If collision expected, do nothing
            if collision_check(map_obstacle, child_candidate_position):
                continue
            # If not collision, create child node
            child = Node(cur_node, child_candidate_position)
            # If already in closed list, do nothing
            if child in closed_list:
                continue
            # If not in closed list, update open list
            child.f = (cur_node.f + action[2] +
                       100.0 * (abs(cur_node.position[0] - goal_node.position[0])**2 + abs(cur_node.position[1] - goal_node.position[1])**2)) # 휴리스틱
            if child in open_list:
                if child.f < open_list[open_list.index(child)].f:
                    open_list[open_list.index(child)].parent = child.parent
                    open_list[open_list.index(child)].f = child.f
            else:
                open_list.append(child)

        # show graph
        if show_animation:
            plt.plot(cur_node.position[0], cur_node.position[1], 'yo', alpha=0.5)
            if len(closed_list) % 100 == 0:
                plt.pause(0.1)


def main():
    start, goal, omap = map()

    if show_animation == True:
        plt.figure(figsize=(8, 8))
        plt.plot(start[0], start[1], 'bs', markersize=7)
        plt.text(start[0], start[1] + 0.5, 'start', fontsize=12)
        plt.plot(goal[0], goal[1], 'rs', markersize=7)
        plt.text(goal[0], goal[1] + 0.5, 'goal', fontsize=12)
        plt.plot(omap[0], omap[1], '.k', markersize=10)
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("X [m]"), plt.ylabel("Y [m]")
        plt.title("Astar algorithm", fontsize=20)

    opt_path = Astar(start, goal, omap)
    print("Optimal path found!")
    opt_path = np.array(opt_path)

    if show_animation == True:
        plt.plot(opt_path[:, 0], opt_path[:, 1], "m.-")
        plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    main()
