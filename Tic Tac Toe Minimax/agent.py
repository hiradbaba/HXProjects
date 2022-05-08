from copy import deepcopy
from time import time
class TicTacAgent:
    
    def __init__(self,playerNode,opNode,playerFirst):
        self.playerNode = playerNode
        self.opNode = opNode
        self.win = 1
        self.lose = -1
        self.draw = 0
        self.expandedNodes = -1
        self.playerFirst = playerFirst

    #not the one in tic.py this is just for evaluating branches
    def isAgentTurn(self,board):
        oCount = 0
        for i in range(3):
            for j in range(3):
                if board[i][j] == self.opNode:
                    oCount -= 1
                elif board[i][j] == self.playerNode:
                    oCount += 1

        if self.playerFirst:
            return oCount == 0 #since agent is the second player
        else:
            return oCount != 0 #since agent is the first Player

    def isFull(self,board):
        for row in board:
            if 0 in row:
                return False
        return True


    def getAvailableMoves(self,board):
        moves = []
        node = None
        if self.isAgentTurn(board):
            node = self.playerNode
        else:
            node = self.opNode 
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    deepBoardCopy = deepcopy(board)
                    deepBoardCopy[i][j] = node
                    moves.append(deepBoardCopy)
        return moves

    def isWin(self,playerNode,board):
        ccount = 0
        for i in range(3):
            if board[i][i] == playerNode:
                ccount+=1
                if ccount == 3:
                    return True     
        ccount = 0
        for i,j in zip(range(2,-1,-1),range(3)):
            if board[i][j] == playerNode:
                ccount+=1
                if ccount == 3:
                    return True

        for i in range(3):
            hcount = 0
            vcount = 0
            for j in range(3):
                if board[i][j] == playerNode:
                    hcount += 1
                    if hcount == 3:
                        return True
                if board[j][i] == playerNode:
                    vcount +=1
                    if vcount == 3:
                        return True
                
        return False

    def resetExpandedNodes(self):
        self.expandedNodes = -1 

    '''The min-max part'''    
    def chooseMove(self,board):
        branches=[]
        self.expandedNodes+=1
        if self.isWin(self.opNode,board):
            return (self.lose,board)
        elif self.isWin(self.playerNode,board):
            return (self.win,board)
        elif self.isFull(board):
            return (self.draw,board)
        else:
            branches = self.getAvailableMoves(board)
            evaluations = []
            for branch in branches:
                evaluations.append(self.chooseMove(branch)[0])
        
        agentTurn = self.isAgentTurn(board)
        if agentTurn:
            if self.win in evaluations:
                return (self.win,branches[evaluations.index(self.win)])
            elif self.draw in evaluations:
                return (self.draw,branches[evaluations.index(self.draw)])
            else:
                return (self.lose,branches[evaluations.index(self.lose)])
        else:
            if self.lose in evaluations:
                return (self.lose,branches[evaluations.index(self.lose)])
            elif self.draw in evaluations:
                return (self.draw,branches[evaluations.index(self.draw)])
            else:
                return (self.win,branches[evaluations.index(self.win)])
    '''Alpha beta puring'''
    def chooseWithAlphaBeta(self,board,alpha,beta):
        branches=[]
        self.expandedNodes+=1
        if self.isWin(self.opNode,board):
            return (self.lose,board)
        elif self.isWin(self.playerNode,board):
            return (self.win,board)
        elif self.isFull(board):
            return (self.draw,board)
        else:           
            agentTurn = self.isAgentTurn(board)
            if agentTurn:
                branches = self.getAvailableMoves(board)
                evaluations = []
                maximum = -2 #represents -infinity
                for branch in branches:
                    evaluation = self.chooseWithAlphaBeta(branch,alpha,beta)[0]
                    evaluations.append(evaluation)
                    maximum = max(maximum,evaluation)
                    alpha = max(alpha,maximum)
                    if beta <= maximum:
                        return(maximum,branch)
                return (maximum, branches[evaluations.index(maximum)])
                
            else:
                branches = self.getAvailableMoves(board)
                evaluations = []
                minimum = 2 #represents +infinity
                for branch in branches:
                    evaluation = self.chooseWithAlphaBeta(branch,alpha,beta)[0]
                    evaluations.append(evaluation)
                    minimum = min(minimum,evaluation)
                    beta = min(beta,minimum)
                    if minimum <= alpha:
                        return (minimum,branch)

                return (minimum,branches[evaluations.index(minimum)])
        

