from agent import TicTacAgent
from time import time
class TicTacToe:
    
    def __init__(self,playerNode,optNode,playerFirst):
        self.board = [[0,0,0],[0,0,0],[0,0,0]]
        self.winner = None
        self.optNode = optNode
        self.playerNode = playerNode
        self.playerFirst = playerFirst
        self.agent = TicTacAgent(playerNode,optNode,playerFirst)
    
    def initializeBoard(self):
        return [[0,0,0],[0,0,0],[0,0,0]]

    def drawBoard(self):
        for row in self.board:
            for element in row:
                print(element,end=' ')
            print("")

    def isFilled(self,xpos,ypos):
        if xpos>2 or ypos>2 or xpos<0 or ypos<0:
            return False
        return self.board[xpos][ypos] == 0

    def isWin(self,playerNode):
        ccount = 0
        for i in range(3):
            if self.board[i][i] == playerNode:
                ccount+=1
                if ccount == 3:
                    self.winner = playerNode
                    return True     
        ccount = 0
        for i,j in zip(range(2,-1,-1),range(3)):
            if self.board[i][j] == playerNode:
                ccount+=1
                if ccount == 3:
                    self.winner = playerNode
                    return True

        for i in range(3):
            hcount = 0
            vcount = 0
            for j in range(3):
                if self.board[i][j] == playerNode:
                    hcount += 1
                    if hcount == 3:
                        self.winner = playerNode
                        return True
                if self.board[j][i] == playerNode:
                    vcount +=1
                    if vcount == 3:
                        self.winner = playerNode
                        return True
                
        return False
    
    def isBoardFull(self):
        for row in self.board:
            if 0 in row:
                return False
        return True

    def startGame(self):
        gameFinished = False
        firstPlayerTurn = self.playerFirst
        while not gameFinished:
            self.drawBoard()
            
            if firstPlayerTurn:
                av = False #if player uses an invalid position
                print("Player:")
                while not av:    
                    xpos = int(input("posX:")) - 1
                    ypos = int(input("posY:")) - 1
                    av = self.isFilled(xpos,ypos)
                self.board[xpos][ypos] = self.playerNode
            else:
                print("Agent:")
                start = time()
                self.board = self.agent.chooseMove(self.board)[1]
                end = time()
                print("Nodes Expanded: {}".format(self.agent.expandedNodes))
                print("Time Elapsed: {}".format(end-start))
                self.agent.resetExpandedNodes()
            if self.isWin(self.playerNode) or self.isWin(self.optNode):
                self.drawBoard()
                print("Winner: " + self.winner)
                return

            if self.isBoardFull():
                self.drawBoard()
                print('DRAW')
                return

            firstPlayerTurn = not(firstPlayerTurn)

    def startOptimizedGame(self):
        gameFinished = False
        firstPlayerTurn = self.playerFirst
        while not gameFinished:
            self.drawBoard()
            
            if firstPlayerTurn:
                av = False #if player uses an invalid position
                print("Player:")
                while not av:    
                    xpos = int(input("posX:")) - 1
                    ypos = int(input("posY:")) - 1
                    av = self.isFilled(xpos,ypos)
                self.board[xpos][ypos] = self.playerNode
            else:
                print("Agent:")
                start = time()
                self.board = self.agent.chooseWithAlphaBeta(self.board,-2,2)[1]
                end = time()
                print("Nodes Expanded: {}".format(self.agent.expandedNodes))
                print("Time Elapsed: {}".format(end-start))
                self.agent.resetExpandedNodes()
            if self.isWin(self.playerNode) or self.isWin(self.optNode):
                self.drawBoard()
                print("Winner: " + self.winner)
                return

            if self.isBoardFull():
                self.drawBoard()
                print('DRAW')
                return

            firstPlayerTurn = not(firstPlayerTurn)

                
tictac = TicTacToe("H","A",True)
tictac.startGame()
#tictac.startOptimizedGame()