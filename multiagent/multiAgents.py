# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        max_d_ghost = max([util.manhattanDistance(newPos, i.getPosition()) for i in newGhostStates])

        min_d_food = 0
        if (newFood.asList()):
            min_d_food = min([util.manhattanDistance(newPos, i) for i in newFood.asList()])

        return successorGameState.getScore() + max_d_ghost - min_d_food

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        """
        In this part of code, we return the action with max value
        """
        actions = list()
        for action in gameState.getLegalActions(0):
            action_value = self.min_v(gameState.generateSuccessor(0, action), 1, 0)
            actions.append((action_value, action))
        return max(actions)[1]
    "max_v and min_v funcs return a value"
    def max_v(self, state, depth):
        if depth == self.depth or len(state.getLegalActions(0)) == 0 or state.isLose() or state.isWin():
            return self.evaluationFunction(state)
        v = float("-inf")
        successors = set()
        for action in state.getLegalActions(0):
            succ = state.generateSuccessor(0, action)
            successors.add(succ)
        v = max([v] + [self.min_v(s, 1, depth) for s in successors])
        return v

    def min_v(self, state, agent, depth):
        if len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state)
        v = float("inf")
        successors = set()
        for action in state.getLegalActions(agent):
            succ = state.generateSuccessor(agent, action)
            successors.add(succ)
        if agent < state.getNumAgents() - 1:
            v = min([v] + [self.min_v(s, agent+1, depth) for s in successors])
        else:
            v = min([v] + [self.max_v(s, depth+1) for s in successors])
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = list()
        for action in gameState.getLegalActions(0):
            action_value = self.min_value(gameState.generateSuccessor(0, action), 1, 0, float("-inf"), float("inf"))
            actions.append((action_value, action))
        return max(actions)[1]

    def max_value(self, state, depth, alpha, beta):
        if len(state.getLegalActions(0)) == 0 or self.depth == depth:
            return self.evaluationFunction(state)
        v = float("-inf")
        successors = set()
        for action in state.getLegalActions(0):
            succ = state.generateSuccessor(0, action)
            successors.add(succ)
        for s in successors:
            v = max(v, self.min_value(s, depth, 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(v, alpha)
        return v

    def min_value(self, state, depth, agent, alpha, beta):
        if len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state)
        v = float("inf")
        successors = set()
        for action in state.getLegalActions(0):
            succ = state.generateSuccessor(0, action)
            successors.add(succ)
        if agent < state.getNumAgents()-1:
            for s in successors:
                v = min(v, self.min_value(s, depth, agent + 1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(v, beta)
        else:
            for s in successors:
                v = min(v, self.max_value(s, depth + 1, alpha, beta))
                if v <= alpha:
                    return v
                beta = min(v, beta)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

