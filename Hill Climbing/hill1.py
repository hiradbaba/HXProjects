from numpy import cos,sin,pi,log,exp
import hxplot
from time import time
fcounter = 0
start = time()
end = time()
values = []
points = []
def dx(n,a,xi):
    x = pi*2*xi
    dx = 2*xi + a*pi*2*sin(x)
    return dx

def f(n,a,dim):
    global fcounter
    global end
    global start,points,values
    fcounter += 1
    z = n*a
    for i in range(n):
        xi = dim[i]
        z += (xi*xi) - a*cos(2*pi*xi)
    
    if fcounter  > 1000:
        fcounter =0
        print(min(values))
        end =time()
        print(end-start)
        quit()
    return z

def f2(args,dim):
    n = args[0]
    a = args[1]
    z = n*a
    for i in range(n):
        xi = dim[i]
        z += (xi*xi) - a*cos(2*pi*xi)
    return z

def getNextPoint(n,a,point):
    nextPoint=[]
    alpha = 0.001
    for xi in point:
        newXi = xi - alpha*dx(n,a,xi)
        nextPoint.append(newXi)
        if abs(newXi) > 5.12:
            return False
    return nextPoint

    
def hillClimbing(currentPoint,n,a):
    initial = currentPoint
    prev_initial = [initial[0]+1,initial[0]+1]
    while initial != prev_initial:
        prev_initial = initial
        initial = getNextPoint(n,a,initial)
        
        if initial == False or f(n,a,initial) > f(n,a,prev_initial):
            return [prev_initial,f(n,a,prev_initial)]  
           
    return [initial,f(n,a,initial)]

'''
n = 2 , a =1 -> iteration: 50  min(10)
n = 2 , a =10 -> iteration:100 min(50)
n = 5 , a =10 -> iteration: _ min(200)
'''
def iterativeHillClimbing(n,a,iterCount):
    from random import uniform
    global points
    global values
    points = []
    values = []
    for i in range(iterCount):
        point = []
        for j in range(n):
            point.append(uniform(-5.12,5.12))
        res = hillClimbing(point,n,a)
        points.append(res[0])
        values.append(res[1])
    minVal = min(values)
    return [points[values.index(minVal)],minVal] 

def acceptance(delta,temp):
    return 1.0/(1+exp(-delta/temp))

def getProbability(delta,temperature,n):
    return (2*pi*temperature)**(-n/2)*exp(-(delta**2)/(temperature*2))

def getPoint(n):
    from random import uniform
    return [uniform(-5.12,5.12) for i in range(n)]

def simulatedAnnealing(n,a):
    # from random import random
    step=1
    current = getPoint(n)
    temperature = 30000.0
    alpha = 0.85
    p = []
    num=[]
    while temperature > 0.00001:
        #temperature -= temperature/log(step+1)
        temperature = alpha*temperature/log(step+1)
        
        nextPoint = getPoint(n)
        while nextPoint == current:
            nextPoint = getPoint(n)
        # current = hillClimbing(current,n,a)[0]
        # nextPoint = hillClimbing(current,n,a)[0]
        delta = f(n,a,current) - f(n,a,nextPoint)
        if delta>0:
            current = nextPoint     
        else:
            # probability = exp(-delta/temperature)
            probability = getProbability(delta,temperature,n)
            if probability > acceptance(delta,temperature):
                current = nextPoint
        step+=1
        num.append(f(n,a,current))
        p.append(current)
        
    # print(print(len(p)))
    # print(print(len(num)))
    minResult = min(num)    
    solution = p[num.index(minResult)]
    return [solution,minResult]

def main():
    print("\n--------OPTIONS-----------\n\n1- Simple Hill climbing \n2- Iterative Hill Climbing\n3- Simulated Annealing\n---any number to exit-----\n")
    while True:
        inp = int(input("Choose Algorithm: "))
        point = None
        xi = []
        if inp==1:
           
            n = int(input("N: "))
            a = float(input("A: "))
            for i in range(n):
                xi.append(float(input("x{} point:".format(i))))
            start=time()
            point = hillClimbing(xi,n,a)
            end =time()
            print(point)
            print("time: "+str(end-start))
            if n == 2:
                plotter = hxplot.HillPlotter((-5.12,5.12),f2,(n,a))
                plotter.plot(point[0])

        elif inp==2:
            
            n = int(input("N: "))
            a = float(input("A: "))
            ic = int(input("Number of iterations: "))
            start = time()
            point = iterativeHillClimbing(n,a,ic)
            end = time() 
            print(point)
            #print("time: "+str(end-start))
            if n == 2:
                plotter = hxplot.HillPlotter((-5.12,5.12),f2,(n,a))
                plotter.plot(point[0])
        elif inp==3:
            n = int(input("N: "))
            a = float(input("A: "))
            point = simulatedAnnealing(n,a)
            print (point)
            if n == 2:
                plotter = hxplot.HillPlotter((-5.12,5.12),f2,(n,a))
                plotter.plot(point[0])
        else:
            quit()

main()