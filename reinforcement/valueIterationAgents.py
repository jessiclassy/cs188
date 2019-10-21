# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
            Update the .values attribute of SELF. Make a copy of SELF.values,
            and while SELF.iterations is non-zero, loop through computing the
            best Q-value and its corresponding action. The best Q-value is
            assigned as the value of SELF.values[state]
        """

        count = self.iterations
        while (count):
            # print("runValueIteration will be looped", count, 'times')
            updatedValues = self.values.copy() #copy of old values to do 'batch' value iteration
            states = self.mdp.getStates() #states are (x, y) coordinates

            for s in states: #iterate thru states
                if self.mdp.isTerminal(s):
                    continue
                maxVal = float('-inf') #initialize max variable that updates only when sum is greater than previous
                for a in self.mdp.getPossibleActions(s):
                    tempSum = self.computeQValueFromValues(s, a)
                    if tempSum > maxVal:
                        maxVal = tempSum
                updatedValues[s] = maxVal #max sum assign to respective state in copied values
            self.values = updatedValues
            count -= 1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        successors = self.mdp.getTransitionStatesAndProbs(state, action) #transitions are  tuples of (successor state, probability)
        sum = 0
        for i in range(len(successors)):
            successorState = successors[i][0]
            successorTransition = successors[i][1]
            reward = self.mdp.getReward(state, action, successorState) #rewards is a float for the reward received when transitioning from s to nextS

            oldSuccessorVal = self.getValue(successorState)
            successorUtility = successorTransition * (reward + (self.discount * oldSuccessorVal))
            sum += successorUtility
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        else:
            qVals = {}
            for a in self.mdp.getPossibleActions(state):
                sum = self.computeQValueFromValues(state, a)
                qVals[sum] = a
            bestAction = qVals[max(qVals.keys())]
            return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # count = self.iterations
        # while (count):
        #     # print("runValueIteration will be looped", count, 'times')
        #     updatedValues = self.values.copy() #copy of old values to do 'batch' value iteration
        #     states = self.mdp.getStates() #states are (x, y) coordinates

        #     for s in states: #iterate thru states
        #         if self.mdp.isTerminal(s):
        #             continue
        #         maxVal = float('-inf') #initialize max variable that updates only when sum is greater than previous
        #         for a in self.mdp.getPossibleActions(s):
        #             tempSum = self.computeQValueFromValues(s, a)
        #             if tempSum > maxVal:
        #                 maxVal = tempSum
        #         updatedValues[s] = maxVal #max sum assign to respective state in copied values
        #     self.values = updatedValues
        #     count -= 1

        count = self.iterations
        states = self.mdp.getStates() #states are (x, y) coordinates
        stateIndex = 0
        while (count):
            if stateIndex > len(states) - 1:
                stateIndex = 0
            currState = states[stateIndex]
            if not self.mdp.isTerminal(currState):
                maxVal = float('-inf') #initialize max variable that updates only when sum is greater than previous
                for a in self.mdp.getPossibleActions(currState):
                    tempSum = self.computeQValueFromValues(currState, a)
                    if tempSum > maxVal:
                        maxVal = tempSum
                self.values[currState] = maxVal #max sum assign to respective state in copied values
            count -= 1
            stateIndex += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
