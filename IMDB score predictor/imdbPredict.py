import hypo
import pandas as p


data = p.read_csv('imdbdata2.csv')
#inital data
directorRate = list(data['directors_avg_imdb'])
directorRate= [i/10.0 for i in directorRate]
actorRate = list(data['actors_avg_imdb'])
actorRate = [i/10.0 for i in actorRate]
directorWin= list(data['director_wins'])
directorWin=[i/292.0 for i in directorWin]
directorNom= list(data['director_nominations'])
directorNom = [i/551.0 for i in directorNom]
actorWin= list(data['actors_wins'])
actorWin= [i/312.0 for i in actorWin]
actorNom= list(data['actors_nominations'])
actorNom= [i/384.0 for i in actorNom]
#maximums-----------------------------
maxRate = 10.0
maxDirectorWin = 292.0
maxDirectorNom = 551.0
maxActorWin = 312.0
maxActorNom = 384.0
#y---------------------------
dataResults = list(data['rating'])
#x---------------------------
number_of_records = len(dataResults)
X = [[1] for i in range(number_of_records)]
for i in range(number_of_records):
    X[i].extend([directorRate[i],directorNom[i],directorWin[i],actorRate[i],actorNom[i],actorWin[i]])


# X = [ones,directorRate,directorNom,directorWin,actorRate,actorNom,actorWin]
# hypo.cost_function(X[0],dataResults,number_of_records)
# print(hypo.theta)
#print(X[0])
#hypo.gradient_descent(DATA,result,len(result))
#hypo.theta=[3.7548032173178183, 5.6091920335895598, -1.4708286501532712, 1.418046369030016, 2.8221124863925602, 0.054831863165605864, 1.4912360968700671]
#print(hypo.theta)
# hypo.theta =[6.5312826504597625, 0.13721745461898341, -1.3076602372606507, 1.9758221657418475, 0.18152196460802555, -0.18745136684673477, 2.2480483582047652]
hypo.theta =[7.5316656328269449, 0.13564672455909632, -1.5072876919069587, 2.1269269018156551, 0.18008980764976984, -0.15807145879804022, 2.2146658807985933]

while True:
    print("Director INFO------------------")
    drate = float(input("Director IMDB rate:"))
    dwon = float(input("Number of Awards Won:"))
    dnom = float(input("Number of Awards Nomination:"))
    print("Actor INFO------------------")
    arate = float(input("Actor IMDB rate:"))
    awon = float(input("Number of Awards Won:"))
    anom = float(input("Number of Awards Nomination:"))
    print("Movie Rate--------------------")
    tempList = [1,drate/maxRate,arate/maxRate,dwon/maxDirectorWin,dnom/maxDirectorNom,awon/maxActorWin,anom/maxActorNom]
    print(hypo.hypothesis(tempList))
