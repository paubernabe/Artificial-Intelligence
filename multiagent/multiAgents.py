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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        """
        I've done the basic implementation that needs Reflex Agent to survive in
        the maze.
        The score that returns this function is basically the original score + the max distance
        of the ghost - the min distance of food.
        """

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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        We call the function with agent 0 (pacman), depth 0 and maximize = True
        """
        return self.minimax(gameState, 0, 0, True)[1]

    """
    If agent does not have any legal moves or we are on our maximum depth, we will
    return the evaluation func of the state
    
    Now let's see the two cases of minimax, maximize or not maximize:
    
    If we have to maximize, that means is pacman's turn. we will look through all legal actions and we'll see
    which one maximize the most.
    
    If we have to minimize, that means it's ghosts turn, sometimes we have more than one ghost and we must control that.
    We will return the movement that minimizes the most. 
    """
    def minimax(self, state, agent, depth, maximize):
        if depth == self.depth or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None

        if maximize:
            v = float('-inf'), None
            for action in state.getLegalActions(agent):
                successor = self.minimax(state.generateSuccessor(agent, action), 1, depth, False)
                candidate_alpha = (successor[0], action)
                v = max(candidate_alpha, v)
            return v
        else:
            v = float('inf'), None
            nextagent = agent + 1
            if agent == state.getNumAgents() - 1: # when ghosts turn is over, nextagent will be pacman
                nextagent = 0
            for action in state.getLegalActions(agent):
                if nextagent > 0:
                    successor = self.minimax(state.generateSuccessor(agent, action), nextagent, depth, False)
                else:
                    successor = self.minimax(state.generateSuccessor(agent, action), nextagent, depth + 1, True)
                candidate_beta = (successor[0], action)
                v = min(candidate_beta, v)
            return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.alphabeta(gameState, 0, 0, (float("-inf"), None), (float("inf"), None), True)[1]

    """
    This function works as the minimax explained before, but we stop exploring states that we think that won't be
    good for us using alpha beta pruning.
    
    """
    def alphabeta(self, state, agent, depth, alpha, beta, maximize):
        if depth == self.depth or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None

        if maximize:
            v = float('-inf'), None
            for action in state.getLegalActions(agent):
                successor = self.alphabeta(state.generateSuccessor(agent, action), 1, depth, alpha, beta, False)
                candidate_alpha = (successor[0], action)
                v = max(candidate_alpha, v)
                if beta[0] < v[0]:
                    return v
                alpha = max(v, alpha)
            return v
        else:
            v = float('inf'), None
            nextagent = agent + 1
            if agent == state.getNumAgents() - 1:
                nextagent = 0
            for action in state.getLegalActions(agent):
                if nextagent > 0:
                    successor = self.alphabeta(state.generateSuccessor(agent, action), nextagent, depth, alpha, beta,
                                               False)
                else:
                    successor = self.alphabeta(state.generateSuccessor(agent, action), nextagent, depth + 1, alpha,
                                               beta, True)
                candidate_beta = (successor[0], action)
                v = min(candidate_beta, v)
                if v[0] < alpha[0]:
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
        return self.expectimax(gameState, 0, 0, True)[1]
    """
    Expectimax also works like minimax, but is quite different.
    
    When we have to maximize, it works like minimax, we get the move with max value, but when we don't have to maximize,
    we have to get the expected action. 
    It's a simple calculus, we store the count the number of actions that we can do in a variable and we sum the value
    of the action in other variable. We will return value/number of actions.
    """
    def expectimax(self, state, agent, depth, maximize):
        if depth == self.depth or len(state.getLegalActions(agent)) == 0:
            return self.evaluationFunction(state), None

        if maximize:
            v = float('-inf'), None
            for action in state.getLegalActions(agent):
                successor = self.expectimax(state.generateSuccessor(agent, action), 1, depth, False)
                candidate_alpha = (successor[0], action)
                v = max(candidate_alpha, v)
            return v
        else:
            v = 0
            num_acciones = 0
            nextagent = agent + 1
            if agent == state.getNumAgents() - 1:
                nextagent = 0
            for action in state.getLegalActions(agent):
                num_acciones += 1
                if nextagent > 0:
                    successor = self.expectimax(state.generateSuccessor(agent, action), nextagent, depth, False)
                    v += successor[0]
                else:
                    successor = self.expectimax(state.generateSuccessor(agent, action), nextagent, depth + 1, True)
                    v += successor[0]
            return v / num_acciones, action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I used the code of the Reflex Agent seen before, but instead focusing on
      he future states, I've changed to focus only on current states. It works quite good.
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    max_d_ghost = max([util.manhattanDistance(newPos, i.getPosition()) for i in newGhostStates])
    min_d_food = 0
    if newFood.asList():
        min_d_food = min([util.manhattanDistance(newPos, i) for i in newFood.asList()])

    return currentGameState.getScore() + max_d_ghost - min_d_food


# Abbreviation
better = betterEvaluationFunction
