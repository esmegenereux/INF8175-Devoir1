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

from custom_types import Direction
from pacman import GameState
from typing import Any, Tuple,List
import util


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self)->Any:
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state:Any)->bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state:Any)->List[Tuple[Any,Direction,int]]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions:List[Direction])->int:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def tinyMazeSearch(problem:SearchProblem)->List[Direction]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem:SearchProblem)->List[Direction]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 1 ICI

    '''
    stack = util.Stack()
    visited = set()

    start_position = problem.getStartState()
    stack.push((start_position, [])) # pushes the first state in the fringe, with an empty path

    while not stack.isEmpty():
        current_pos, path = stack.pop() # Pop the current position and the path taken to reach it

        # Verify if the current state is the goal state, in which case we can stop
        if problem.isGoalState(current_pos):
            return path
        else:
            # Verify that the state has not already been visited, in which case we must process it
            if current_pos not in visited:
                visited.add(current_pos)

                # Get successors and push them onto the stack
                for successor_pos, action, cost in problem.getSuccessors(current_pos):
                    if successor_pos not in visited:
                        stack.push((successor_pos, path + [action]))

    return []  # Return an empty path if no solution is found

def breadthFirstSearch(problem:SearchProblem)->List[Direction]:
    """Search the shallowest nodes in the search tree first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 2 ICI
    '''
    queue = util.Queue()
    visited = set()

    start_state = problem.getStartState()
    queue.push((start_state, [])) # push the first state in the fringe, with an empty path

    while not queue.isEmpty():
        current_pos, path = queue.pop() # Pop the current position and the path taken to reach it

        # Verify if the current state is the goal state, in which case we can stop
        if problem.isGoalState(current_pos):
            return path
        else:
            # Verify that the state has not already been visited, in which case we must process it
            if current_pos not in visited:
                visited.add(current_pos)

                # Get successors and push them into the queue
                for successor_pos, action, cost in problem.getSuccessors(current_pos):
                    if successor_pos not in visited:
                        queue.push((successor_pos, path + [action]))

    return []  # Return an empty path if no solution is found


def uniformCostSearch(problem:SearchProblem)->List[Direction]:
    """Search the node of least total cost first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 3 ICI
    '''
    from util import PriorityQueue
    
    priority_queue = PriorityQueue() # Initialize the priority queue
    visited = set() # Initialize the visited set

    # Push the start state with cost 0 and an empty path
    start_state = problem.getStartState()
    priority_queue.push((start_state, [], 0), 0)  # (current state, path, cost), priority = 0

    while not priority_queue.isEmpty():
        # Pop the state with the lowest total cost
        current_state, path, cost = priority_queue.pop()

        # If the current state is the goal, return the path
        if problem.isGoalState(current_state):
            return path

        # If the state has not been visited, process it
        if current_state not in visited:
            visited.add(current_state)

            # Get successors and push them into the priority queue
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    total_cost = cost + step_cost
                    priority_queue.push((successor, path + [action], total_cost), total_cost)

    return []



def nullHeuristic(state:GameState, problem:SearchProblem=None)->List[Direction]:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem:SearchProblem, heuristic=nullHeuristic)->List[Direction]:
    """Search the node that has the lowest combined cost and heuristic first."""
    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 4 ICI
    '''
    
    from util import PriorityQueue

    # Initialize the priority queue, visited set
    priority_queue = PriorityQueue()
    visited = set()

    # Push start state with cost 0 and empty path
    start_state = problem.getStartState()
    priority_queue.push((start_state, [], 0), 0)  # (current state, path, cost), priority => 0

    while not priority_queue.isEmpty():
        # Pop the state with the lowest f(n) value
        current_state, path, cost = priority_queue.pop()

        # If the current state is the goal, return the path
        if problem.isGoalState(current_state):
            return path

        # If the state has not been visited, process it
        if current_state not in visited:
            visited.add(current_state)

            # Get successors and push them into the priority queue.
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    g_cost = cost + step_cost  # g(n)
                    h_cost = heuristic(successor, problem)  # h(n)
                    f_cost = g_cost + h_cost  # f(n)
                    priority_queue.push((successor, path + [action], g_cost), f_cost)

    return []  # Return empty path if no solution is found

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
