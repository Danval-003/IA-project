# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    from game import Directions
    from util import Stack

    passStates = [problem.getStartState()]
    actual = [problem.getStartState(), 0]
    stackToPass = Stack()
    stackToPass.push(actual)
    count = -1
    moves = Stack()
    paths = []

    problem.expand(problem.getStartState())

    while len(stackToPass.list) > 0:
        successors = problem.getActions(actual[0])

        if actual[1] < len(successors) and (count == -1 or count < len(passStates)):
            next = problem.getNextState(actual[0], successors[actual[1]])

            # Evitar expandir nodos repetidos
            if next in passStates:
                actual = stackToPass.pop()
                actual[1] = actual[1] + 1
                stackToPass.push(actual)
                continue


            if problem.isGoalState(next):
                actual = stackToPass.pop()
                actual[1] = actual[1] + 1
                stackToPass.push(actual)
                passStates.append(next)
                count = len(passStates) + 1
                paths.append(passStates.copy())
                passStates.pop()
                continue

            
            problem.expand(next)

            actual = [next, 0]

            if next in passStates:
                actual = stackToPass.pop()
                actual[1] = actual[1] + 1
                stackToPass.push(actual)
            else:
                passStates.append(next)
                stackToPass.push(actual)
        else:
            if len(passStates) == 1:
                break
            passStates.pop()
            stackToPass.pop()
            actual = stackToPass.pop()
            actual[1] = actual[1] + 1
            stackToPass.push(actual)


    min_Path = min(paths, key=len)

    for index in range(len(min_Path) - 1):
        next = min_Path[index + 1]
        action = problem.getActions(min_Path[index])

        for i in range(len(action)):
            if problem.getNextState(min_Path[index], action[i]) == next:
                moves.push(action[i])
                break

    return moves.list

def breadthFirstSearch(problem):
    from util import Queue

    initState = problem.getStartState()
    frontier = Queue()
    frontier.push(initState)

    expanded = set()
    memory = {}

    memory[str(initState)] = None

    while not frontier.isEmpty():
        currentState = frontier.pop()

        if problem.isGoalState(currentState):
            actions = []
            while currentState != initState:
                prevState, action = memory[str(currentState)]
                actions.append(action)
                currentState = prevState    
            actions.reverse()
            return actions

        if currentState not in expanded:
            problem.expand(currentState)
            expanded.add(currentState)
            actions = problem.getActions(currentState)

            for action in actions:
                nextState = problem.getNextState(currentState, action)
                if str(nextState) not in memory:
                    memory[str(nextState)] = (currentState, action)
                    frontier.push(nextState)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    openQueue = util.PriorityQueue()
    closed = []
    start = problem.getStartState()

    openQueue.push([start, []], heuristic(start, problem))

    while not openQueue.isEmpty():
        q = openQueue.pop()

        if problem.isGoalState(q[0]):
            return q[1]

        if q[0] not in closed:
            neighbors = problem.expand(q[0])
            for successor in neighbors:
                if successor[0] not in closed:
                    h = heuristic(successor[0], problem)
                    g = problem.getCostOfActionSequence(q[1] + [successor[1]])
                    f = h + g
                    openQueue.push([successor[0], q[1] + [successor[1]]],  f)
            closed.append(q[0])

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
