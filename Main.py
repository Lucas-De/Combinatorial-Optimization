from InstanceVerolog2019 import passInstance
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random
import time
import argparse
import SolutionVerolog2019
import os
import subprocess


random.seed(2018)

File= "Instances/CO2018_1.txt"
Instance=passInstance(File,False)

Dataset = Instance.Dataset
Name = Instance.Name

Days =  Instance.Days
TruckCapacity = Instance.TruckCapacity
TruckMaxDistance = Instance.TruckMaxDistance
TruckDistanceCost =  Instance.TruckDistanceCost           
TruckDayCost = Instance.TruckDayCost
TruckCost = Instance.TruckCost 
TechnicianDistanceCost =   Instance.TechnicianDistanceCost           
TechnicianDayCost = Instance.TechnicianDayCost
TechnicianCost = Instance.TechnicianCost

Machines=Instance.Machines       #Machine objects have values: ID, size, idlePenalty
Requests=Instance.Requests       #Request objects have values: ID, customerLocID, fromDay, toDay, machineID, amount, totalSize
Locations=Instance.Locations     #Locations objects have values: ID, X, Y
Technicians=Instance.Technicians #Technicians objects have values: ID, locationID, maxDayDistance, maxNrInstallations, capabilities]

def printSolution():
    f=open("SOLUTION_"+str(File[-5:-4])+".txt", "w+")
    f.write("DATASET = CO2018 freestyle \n")
    f.write("NAME = Instance " + str(File[-5:-4]) + "\n")

    for i in range(1,Days+1):
        currList = []
        f.write("DAY = " + str(i) + "\n")
        f.write("NUMBER_OF_TRUCKS = " + str(len(mainList[i])) + "\n")
        for j in range(len(mainList[i])):
            if(MERGE_ROUTES):
                f.write(str(j+1) + " "+' '.join([str(k) for k in mainList[i][j].truePath]))
            else:
                f.write(str(j+1) + " "+' '.join([str(k.ID) for k in mainList[i][j].seq]))
            f.write("\n")
        f.write("NUMBER_OF_TECHNICIANS = " + str(len(techRoutes[i-1])) + "\n")
        for j in range(len(techRoutes[i-1])):
            f.write(str(techRoutes[i-1][j][0])+" "+' '.join([str(k.ID) for k in techRoutes[i-1][j][1].seq]))
            f.write("\n")

def showMap(RoutesList,Tech=False,ViewSize=False):
    color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    if(Tech):
        TechX= [Locations[Technicians[i].locationID-1].X for i in range(0,len(Technicians))]
        TechY= [Locations[Technicians[i].locationID-1].Y for i in range(0,len(Technicians))]
        plt.plot(TechX, TechY,'ro',marker='.',c='k')

    ReqX= [Locations[Requests[i].customerLocID-1].X for i in range(0,len(Requests))]
    ReqY= [Locations[Requests[i].customerLocID-1].Y for i in range(0,len(Requests))]
    ReqCol= [color[Requests[i].machineID] for i in range(0,len(Requests))]

    if(ViewSize):
        ReqSize= [2.5**Requests[i].amount for i in range(0,len(Requests))]
        plt.scatter(ReqX, ReqY, marker='s' ,c=ReqCol, s=ReqSize)
    else:
        plt.scatter(ReqX, ReqY, marker='s' ,c=ReqCol)

    plt.scatter(Locations[0].X, Locations[0].Y,c='k', marker='*')

    if(Route!=None):
        for routes in RoutesList:
            for route in routes:
                x=[Locations[0].X]+[Locations[i-1].X for i in [req.customerLocID for req in route.seq]]+[Locations[0].X]
                y=[Locations[0].Y]+[Locations[i-1].Y for i in [req.customerLocID for req in route.seq]]+[Locations[0].Y]
                plt.plot(x,y,c='k',linewidth=0.5)

    plt.show()

def updateRDist(fromHere):
    for i in range(0,len(Requests)):
        dist=Distances[fromHere][Requests[i].customerLocID-1]
        Requests[i].dist=dist

def get_size_per_request():
    for i in range(0,len(Requests)):
        totalSize= Requests[i].amount * Machines[Requests[i].machineID-1].size
        Requests[i].totalSize=totalSize
        delaypenal= Requests[i].amount * Machines[Requests[i].machineID-1].idlePenalty
        Requests[i].delayPenalty = delaypenal


def getDistMatrix():
    nrLoc=len(Locations)
    d= np.zeros((nrLoc, nrLoc))
    for i in range(nrLoc):
        for j in range(nrLoc):
            d[i][j]= math.ceil(math.sqrt((Locations[i].X-Locations[j].X)**2 + (Locations[i].Y-Locations[j].Y)**2 ))
    return d

class Route(object):
        Lock=False
        def __init__(self, routeType, technician=None):
            self.seq = []
            self.dist = 0

            if routeType == 'truck':
                self.load = 0
                self.homebase = 0
                self.truePath=[]
            elif routeType == 'technician':
                self.nrMachines = 0
                self.technician = technician
                self.homebase = technician.locationID - 1

        def removeAt(self, i, routeType):
            removed = None
            dist = None
            nrMachines = None
            seq = None
            if routeType == 'truck':
                removed = self.seq[i]
                cancelledLoad = removed.totalSize
                load = self.load - cancelledLoad
            elif routeType == 'technician':
                nrMachines = self.nrMachines - 1

            if (len(self.seq) == 1):
                if Route.Lock == True:
                    seq = []
                    if routeType == 'truck':
                        return (removed, 0, load)
                    elif routeType == 'technician':
                        return (removed, 0, nrMachines)
                else:
                    self.seq = []
                    self.dist = 0
                    if routeType == 'truck':
                        self.load = 0
                        return (removed, 0, self.load)
                    elif routeType == 'technician':
                        self.nrMachines = 0
                        return (removed, 0, self.nrMachines)

            if (i == len(self.seq) - 1):
                a = self.seq[i - 1].customerLocID
                b = self.seq[i].customerLocID
                dist = self.dist - Distances[a - 1][b - 1] - Distances[b - 1][self.homebase] + Distances[a - 1][
                    self.homebase]
            elif (i == 0):
                b = self.seq[i].customerLocID
                c = self.seq[i + 1].customerLocID
                dist = self.dist - Distances[self.homebase][b - 1] - Distances[b - 1][c - 1] + Distances[self.homebase][
                    c - 1]
            else:
                a = self.seq[i - 1].customerLocID
                b = self.seq[i].customerLocID
                c = self.seq[i + 1].customerLocID
                dist = self.dist - Distances[a - 1][b - 1] - Distances[b - 1][c - 1] + Distances[a - 1][b - 1]

            if (i == len(self.seq) - 1):
                seq = self.seq[:i]
            elif (i == 0):
                seq = self.seq[1:]
            else:
                seq = self.seq[:i] + self.seq[i + 1:]

            if (Route.Lock == True):
                if routeType == 'truck':
                    return (removed, dist, self.load)
                elif routeType == 'technician':
                    return (removed, dist, self.nrMachines)
            else:
                self.seq = seq
                self.dist = dist
                if routeType == 'truck':
                    self.load = load
                    return (removed, dist, self.load)
                elif routeType == 'technician':
                    self.nrMachines = nrMachines
                    return (removed, dist, self.nrMachines)

        def add(self,request,routeType,i=None):
            if(i==len(self.seq) or i is None):
                return self.addLast(request,routeType)
            elif(i==0):
                return self.addFirst(request,routeType)
            else:
                newLoc=request.customerLocID
                a= self.seq[i-1].customerLocID
                b= self.seq[i].customerLocID
                dist= self.dist - Distances[a-1][b-1] + Distances[newLoc-1][a-1] + Distances[newLoc-1][b-1]

                if routeType == 'truck':
                    load= self.load + request.totalSize
                    Valid=self.validTruckRoute(dist, load)
                    if(Route.Lock==True): return (Valid, dist, load)

                    if(Valid):
                        self.dist=dist
                        self.load=load
                        temp =self.seq[:i] + [request] + self.seq[i:]
                        self.seq=temp
                        return (Valid, dist, load)
                    else:
                        return (Valid, dist, load)
                elif routeType == 'technician':
                    nrMachines = self.nrMachines + 1
                    Valid = self.validTechRoute(dist, nrMachines)
                    if (Route.Lock == True): return (Valid, dist, nrMachines)

                    if (Valid):
                        self.dist = dist
                        self.nrMachines = nrMachines
                        temp = self.seq[:i] + [request] + self.seq[i:]
                        self.seq = temp
                        return (Valid, dist, nrMachines)
                    else:
                        return (Valid, dist, nrMachines)

        def validTruckRoute(self,dist,load):
            if( dist>TruckMaxDistance  or  load> TruckCapacity):
                return(False)
            else:
                return(True)

        def validTechRoute(self,dist,nrMachines):
            if( dist>self.technician.maxDayDistance  or  nrMachines> self.technician.maxNrInstallations):
                return(False)
            else:
                return(True)

        def addLast(self,request,routeType):
            newLast=request.customerLocID
            if len(self.seq)>0: 
                oldLast=self.seq[-1].customerLocID
                dist= self.dist + Distances[oldLast-1][newLast-1] + Distances[newLast-1][self.homebase] - Distances[self.homebase][oldLast-1]
            else: 
                dist= 2* Distances[newLast-1][self.homebase]

            if routeType == 'truck':
                load=self.load + request.totalSize
                Valid=self.validTruckRoute(dist, load)
                if(Route.Lock==True): return (Valid, dist, load)

                if(Valid):
                    self.dist=dist
                    self.load=load
                    self.seq.append(request)
                    return (Valid, dist, load)
                else:
                    return (Valid, dist, load)
            elif routeType == 'technician':
                nrMachines = self.nrMachines + 1
                Valid = self.validTechRoute(dist, nrMachines)
                if (Route.Lock == True): return (Valid, dist, nrMachines)

                if (Valid):
                    self.dist = dist
                    self.nrMachines = nrMachines
                    self.seq.append(request)
                    return (Valid, dist, nrMachines)
                else:
                    return (Valid, dist, nrMachines)

        def addFirst(self,request,routeType):
            newFirst=request.customerLocID
            oldFirst=self.seq[0].customerLocID
            dist= Distances[self.homebase][newFirst-1] + Distances[newFirst-1][oldFirst-1] - Distances[self.homebase][oldFirst-1]

            if routeType == 'truck':
                load=request.totalSize+self.load
                Valid=self.validTruckRoute(dist, load)
                if(Route.Lock==True): return (Valid, dist, load)

                if(Valid):
                    self.dist=dist
                    self.load=load
                    self.seq= [request]+self.seq
                    return (Valid, dist, load)
                else:
                    return (Valid, dist, load)
            elif routeType == 'technician':
                nrMachines = self.nrMachines + 1
                Valid = self.validTechRoute(dist, nrMachines)
                if (Route.Lock == True): return (Valid, dist, nrMachines)

                if (Valid):
                    self.dist = dist
                    self.nrMachines = nrMachines
                    self.seq = [request] + self.seq
                    return (Valid, dist, nrMachines)
                else:
                    return (Valid, dist, nrMachines)

        def sameTruck(self, mRoute):
            if(self.dist+mRoute.dist<=TruckMaxDistance):

                if len(self.truePath)==0:
                    self.truePath=[i.ID for i in self.seq]
                if len(mRoute.truePath)==0:
                    mRoute.truePath=[i.ID for i in mRoute.seq]

                self.truePath+=  [0] + mRoute.truePath

                self.seq= self.seq + mRoute.seq
                self.dist+=mRoute.dist
                self.load+=mRoute.load

                return True
            else: return False

        def strigify(self):
            tempSeq=[]
            for i in self.seq:
                tempSeq.append(i.ID)
            for i in self.returns:
                tempSeq.insert(i,0)
            self.seq=tempSeq

        def printSeq(self):
            print([i.ID for i in self.seq])

        def getReqIndex(self,requestID):
            index = None
            for i in range(len(self.seq)):
                if self.seq[i].ID == requestID:
                    index = i
            return (index)

        def mergeWith(self,routeType,mRoute,mergeType):
            newSeq=None
            if(mergeType==0):   
                newSeq=self.seq+mRoute.seq
                a=self.seq[-1].customerLocID
                b=mRoute.seq[0].customerLocID
            elif(mergeType==1): 
                newSeq=self.seq+list(reversed(mRoute.seq))
                a=self.seq[-1].customerLocID
                b=mRoute.seq[-1].customerLocID
            elif(mergeType==2): 
                newSeq=list(reversed(self.seq))+mRoute.seq
                a=self.seq[0].customerLocID
                b=mRoute.seq[0].customerLocID
            elif(mergeType==3): 
                newSeq=list(reversed(self.seq))+list(reversed(mRoute.seq))
                a=self.seq[0].customerLocID
                b=mRoute.seq[-1].customerLocID
            else: return False

            if routeType == 'truck':
                load=self.load+mRoute.load
                dist=self.dist+mRoute.dist
                dist-=Distances[a-1][self.homebase]
                dist-=Distances[b-1][self.homebase]
                dist+=Distances[b-1][a-1]

                Valid = self.validTruckRoute(dist,load)
                if(Route.Lock==True):
                    return (Valid, dist, load)
                elif(Valid):
                    self.dist=dist
                    self.load=load
                    self.seq=newSeq
                    return (Valid, dist, load)
                else:
                    return (Valid, dist, load)
            elif routeType == 'technician':
                nrMachines = self.nrMachines + mRoute.nrMachines
                dist=self.dist+mRoute.dist
                dist-=Distances[a-1][self.homebase]
                dist-=Distances[b-1][self.homebase]
                dist+=Distances[b-1][a-1]


                Valid = self.validTechRoute(dist,nrMachines)
                if (Route.Lock == True):
                    return (Valid, dist, nrMachines)
                elif (Valid):
                    self.dist = dist
                    self.nrMachines = nrMachines
                    self.seq = newSeq
                    return (Valid, dist, nrMachines)
                else:
                    return (Valid, dist, nrMachines)

def initRoutes(technician=None,closestReq=None,routeType='truck',avRequests=None):
    routes=[]
    if avRequests != None:
        requests = avRequests
    else:
        requests = Requests

    if routeType == 'truck':
        for i in range(0,len(requests)):
            r=Route(routeType)
            r.add(requests[i],routeType)
            routes.append(r)
    elif routeType == 'technician':
        for i in range(len(closestReq)):
            r = Route(routeType,technician)
            r.add(closestReq[i], routeType)
            routes.append(r)
    return (routes)

def getDistanceOfRoute(RouteList):
    cost=0
    for i in RouteList:
        cost+=i.dist
    return cost

def bestMergeType(r1,r2,routeType):
    Route.Lock=True
    bestType=None
    bestDistance=math.inf
    options=[]

    for i in range(4):
        o=r1.mergeWith(routeType,mRoute=r2,mergeType=i)
        options.append((i,o[0],o[1]))
    for opt in options:
        if(opt[1]==False):
            options.remove(opt)
        elif(opt[2]<bestDistance):
            bestDistance=opt[2]
            bestType=opt[0]
    Route.Lock=False
    return (bestType, bestDistance)

def mergeBestPair(routes,routeType):
    options=[]
    for i in range(len(routes)):
        for j in range(i+1,len(routes)):
            if routes[i].seq != [] and routes[j].seq != []:
                o=bestMergeType(routes[i],routes[j],routeType)
                saving=routes[i].dist+routes[j].dist-o[1]
                bestmergeType=o[0]
                if(o[0]!=None):
                    options.append([i,j,bestmergeType,saving])
    if(len(options)==0):
        return False
    bestPair=max(options,key=lambda x: x[3])
    routes[bestPair[0]].mergeWith(routeType,routes[bestPair[1]],bestPair[2])
    del routes[bestPair[1]]
    return True

#Savings Algorithm
def savingsAlgorithm(timeWindow=False,randomRequests=None,technician=None,closestReq=None,routeType='truck'):
    totalRoutes = []
    if (timeWindow):
        totRequests = copy.deepcopy(Requests)
        for i in range(Days):
            day = i + 1
            currAvRequests = []
            for request in totRequests:
                if request.fromDay <= day and day <= request.toDay:
                    currAvRequests.append(request)

            for request in currAvRequests:
                totRequests.remove(request)

            routes = initRoutes(technician, closestReq, routeType,currAvRequests)
            possible = True
            while (possible):
                possible = mergeBestPair(routes, routeType)

            for route in routes:
                route.day = day
                totalRoutes.append(route)
    elif (randomRequests != None):
        totalRoutes = initRoutes(routeType='truck',avRequests=randomRequests)
        possible = True
        while (possible):
            possible = mergeBestPair(totalRoutes, routeType)
    else:
        totalRoutes = initRoutes(technician,closestReq,routeType)
        possible=True
        while(possible):
            possible=mergeBestPair(totalRoutes,routeType)

    if len(totalRoutes) == 1 and totalRoutes[0].seq == []:
        totalRoutes = None

    return(totalRoutes)

def techniciansSchedule(requestDict):
    # Initial algorithm to create a schedule for the technicians, input is a list of available requests for each day
    availableTech = copy.deepcopy(Technicians)
    nonAvailableTech = []
    finalRouteList = []
    currentRequests = []
    Route.Lock = False

    for i in range(1,Days+1):
        if i in requestDict:
            dayRequests=requestDict[i]
            for request in dayRequests:
                currentRequests.append(request)

        currAvailableTech = [t for t in availableTech]
        dailyRouteList = []

        if currAvailableTech != None and currentRequests != None:
            while len(currAvailableTech) > 0 and len(currentRequests) > 0:
                techList = []
                for j in range(len(currAvailableTech)):
                    technician = currAvailableTech[j]
                    closestReq = computeClosestReq(technician,currentRequests)


                    if len(closestReq) > 0:
                        avgDistance = computeAVG(column(closestReq,1))
                        cost = (1 - technician.usedBefore) * TechnicianCost + (
                                    1 - technician.usedThisDay) * TechnicianDayCost +avgDistance*TechnicianDistanceCost
                        techList.append((technician, column(closestReq, 0), cost))

                if len(techList) > 0:
                    optimalTech = min(techList,key=lambda x:x[2])
                    routes = savingsAlgorithm(technician=optimalTech[0],closestReq=optimalTech[1],routeType='technician')

                    #for route in routes:
                    #    route.printSeq()

                    technician = optimalTech[0]
                    if routes != None and len(routes[0].seq) !=0:
                        finalRoute = getLargestRoute(routes)
                        finalRoute.day = i

                        for k in range(len(finalRoute.seq)):
                            currentRequests.remove(finalRoute.seq[k])

                        dailyRouteList.append((technician.ID,finalRoute))  # append tech ID and daily routes to list
                        technician.usedBefore=True

                    if technician.stillAvailable():
                        technician.prevWorkDays += 1
                    else:
                        technician.breakDaysLeft = 3
                        technician.prevWorkDays = 0
                        availableTech.remove(technician)
                        nonAvailableTech.append(technician)
                currAvailableTech.remove(technician)

        for t in nonAvailableTech:
            t.breakDaysLeft -= 1

            if t.availableAgain():
                availableTech.append(t)
                nonAvailableTech.remove(t)

        finalRouteList.append(dailyRouteList)

    return finalRouteList

def column(matrix,i):
    return [row[i] for row in matrix]

def getLargestRoute(routes):
    maxLength = 0
    finalRoute = None

    for i in range(len(routes)):
        if len(routes[i].seq) > maxLength:
            maxLength = len(routes[i].seq)
            finalRoute = routes[i]

    return finalRoute

def computeClosestReq(technician,requests):
    distList = []

    for i in range(len(requests)):
        if partOfSkillset(technician,requests[i]):
            dist = Distances[technician.locationID - 1][requests[i].customerLocID - 1]
            distList.append((requests[i],dist))
    sortedDist = sorted(distList,key=lambda x:x[1],reverse=True)

    nClosest = []
    n = 0
    while len(sortedDist) > 0:
        currentReq = sortedDist.pop()
        #n += currentReq[0].amount
        n += 1

        if n <= technician.maxNrInstallations:
            nClosest.append(currentReq)
        else:
            #n -= currentReq[0].amount
            break


    return nClosest

def partOfSkillset(technician,request):
    if technician.capabilities[request.machineID - 1] == 1:
        return True
    else:
        return False

def computeAVG(distList):
    sum = 0
    n = len(distList)

    for i in range(n):
        sum = sum + distList[i]

    return (sum/n)

def combine(routes):
    mainList = [[] for i in range(Days+1)]
    for r in routes:
        index=r.day
        mainList[index].append(r)
    for day in range(len(mainList)):
        mainList[day]=combineRoutes(mainList[day])
    combined= []
    for sublist in mainList:
        for item in sublist:
            if(len(item.truePath)==0): item.truePath= [k.ID for k in item.seq]
            combined.append(item)

    return combined

def combineRoutes(truckRoutes):
    sortedRoutes=sorted(truckRoutes,key=lambda x:x.dist)
    finishedSolution=[]

    while len(sortedRoutes)>1:
        current=sortedRoutes[0]
        i=1
        while((i<len(sortedRoutes)) and current.sameTruck(sortedRoutes[i])):
            del sortedRoutes[i]
            i+=1
        finishedSolution.append(current)
        del sortedRoutes[0]

    if len(sortedRoutes)>0:
        finishedSolution.append(sortedRoutes[0])
    return finishedSolution

def QuickRoute(method=1):
    # QuickRoute (I made this up): Prints routing solution which considers time windows
    # This is a stochastic algorithm and requires being run multiple times to get a good solution
    routes=[]
    AvailableRequests= [ r for r in Requests]
    OnDay= [[] for i in range(0,Days)]

    if(method==1): #Lower Variance, but slightly lower mean distance
        for i in range(Days):
            day=i+1
            for req in Requests:
                if(req.fromDay<=day and day<=req.toDay):
                    OnDay[i].append(req)
        dayOrder= list(range(Days))
        random.shuffle(dayOrder)
  
    elif(method==2): #Higher Variance, but slightly higher mean distance
        schedules=[(list(range(r.fromDay,r.toDay+1)),r) for r in Requests]
        for s in schedules:
            OnDay[random.choice(s[0])-1].append(s[1])
        dayOrder= list(range(Days))

    for i in dayOrder:
        while(len(OnDay[i])>0):
            possible=True
            r=Route('truck')
            updateRDist(0)
            while(possible and len(OnDay[i])>0):
                toAdd=min(OnDay[i],key=lambda x: x.dist)
                possible=r.add(toAdd,'truck')[0]
                if(possible):
                    updateRDist(toAdd.customerLocID-1)
                    AvailableRequests.remove(toAdd) 
                    for j in range(Days):
                        if toAdd in OnDay[j]:
                            OnDay[j].remove(toAdd) 
            r.day=i+1
            routes.append(r)
            if(MERGE_ROUTES):
                routes=combine(routes)
    return(routes)

def QuickRouteAlgorithm(iterations=1,method=2):
    optCost=math.inf
    optRoutes=[]

    for i in range(iterations):
        #print(i)
        routes=QuickRoute(method)
        cost=getDistanceOfRoute(routes)
        if(cost<optCost):
            optCost=cost
            optRoutes=routes
    return(optRoutes)

def combQuickSavings(iterations=1):
    optCost = math.inf
    optRoutes = []

    for i in range(iterations):
        OnDay = [[] for i in range(0, Days)]
        schedules=[(list(range(r.fromDay,r.toDay+1)),r) for r in Requests]
        for s in schedules:
            OnDay[random.choice(s[0])-1].append(s[1])

        totRoutes = []
        i = 1
        for requests in OnDay:
            routes = savingsAlgorithm(randomRequests=requests)
            for route in routes:
                route.day = i
                totRoutes.append(route)
            i += 1
        cost = getDistanceOfRoute(totRoutes)
        if (cost < optCost):
            optCost = cost
            optRoutes = totRoutes

    return (optRoutes)

def improveTruckSolution(truckRouteList,techRouteList,iterations):
    for iteration in range(iterations):
        requests = getReqRouteDict(truckRouteList,iteration)
        numOfTrucks = calcTrucksPerDay(truckRouteList)
        prevTechCosts = calcTechCost(techRouteList)

        print("iteration", iteration)

        COST_IMP = -1000000000
        finalRoute1=None
        finalRoute2=None
        finalRequest1=None
        finalRequest2=None
        finalTechRouteList = techRouteList
        VALID=None

        stepSize = int(len(Requests)/ iterations)
        a = iteration * stepSize + 1
        b = (iteration + 1) * stepSize + 1
        subset = list(range(a,b))
        requests = dict((i,requests[i]) for i in subset if i in requests)

        Route.Lock = True
        for i in range(a, b):
            for j in range(a, b):
                if i != j:
                    route1 = requests[i][1]
                    route2 = requests[j][1]

                    request1 = requests[i][0]
                    request2 = requests[j][0]

                    if request1.fromDay <= route2.day <= request1.toDay and route1 != route2:
                        Route.Lock = True
                        previousDist = getDistanceOfRoute([route1, route2])  # distance of old routes

                        req1Index = route1.getReqIndex(request1.ID)  # index of request1
                        req2Index = route2.getReqIndex(request2.ID)  # index of request2
                        (removed, dist1, load1) = route1.removeAt(routeType='truck',i=req1Index)  # remove request1 from route1
                        (valid, dist2, load2) = route2.add(request1, routeType='truck',i=req2Index)  # add request1 to route2

                        costImprovement = 0

                        if len(route1.seq) == 1:
                            costImprovement += TruckDayCost

                            if route1.day in getMaxIndices(numOfTrucks) and not multipleMax(numOfTrucks):
                                costImprovement += TruckCost

                        if valid:
                            currentDist = dist1 + dist2  # sum of two new routes
                        else:
                            costImprovement -= TruckDayCost  # otherwise a new route has to be created for request1
                            newRoute = Route('truck')
                            (nvalid, ndist, nload) = newRoute.add(request1, 'truck')
                            newRoute.day = route2.day
                            currentDist = dist1 + ndist + route2.dist

                            if route2.day in getMaxIndices(numOfTrucks):
                                costImprovement -= TruckCost

                        costImprovement += (previousDist - currentDist)

                        if route1.day != route2.day:
                            requestDict = getReqDict(truckRouteList)
                            r1List = requestDict[route1.day + 1]
                            r1List.remove(request1)
                            requestDict[route1.day + 1] = r1List

                            r2List = requestDict[route2.day + 1]
                            r2List.append(request1)
                            requestDict[route2.day + 1] = r2List

                            newTechRouteList = techniciansSchedule(requestDict)

                            currTechCosts = calcTechCost(newTechRouteList)
                            costImprovement += (prevTechCosts - currTechCosts)
                        else:
                            newTechRouteList = techRouteList

                        if (COST_IMP < costImprovement):
                            COST_IMP = costImprovement
                            finalRoute1 = route1
                            finalRoute2 = route2
                            finalRequest1 = request1
                            finalRequest2 = request2
                            finalTechRouteList = newTechRouteList
                            VALID = valid

        Route.Lock = False
        if COST_IMP > 0:                                                     #adapt routes when cost improvement is positive
            finalReq1Index = finalRoute1.getReqIndex(finalRequest1.ID)
            finalReq2Index = finalRoute2.getReqIndex(finalRequest2.ID)

            truckRouteList[finalRoute1.day].remove(finalRoute1)
            techRouteList = finalTechRouteList

            if VALID:
                truckRouteList[finalRoute2.day].remove(finalRoute2)
                finalRoute1.removeAt(routeType='truck', i=finalReq1Index)
                finalRoute2.add(finalRequest1, routeType='truck', i=finalReq2Index)

                if finalRoute1.seq != []:
                    truckRouteList[finalRoute1.day].append(finalRoute1)
                truckRouteList[finalRoute2.day].append(finalRoute2)
            else:
                finalRoute1.removeAt(routeType='truck', i=finalReq1Index)
                newRoute = Route('truck')
                newRoute.add(finalRequest1,'truck')
                newRoute.day = finalRoute2.day

                if finalRoute1.seq != []:
                    truckRouteList[finalRoute1.day].append(finalRoute1)
                truckRouteList[newRoute.day].append(newRoute)
        else:
            print("stopped because of no improvements")
            #return (truckRouteList,techRouteList)

        print(COST_IMP)
        print("\n")
    return (truckRouteList,techRouteList)

def multipleMax(numberList):
    maxNumber = max(numberList)
    count = 0
    for number in numberList:
        if number == maxNumber:
            count += 1

    if count > 1:
        return True
    else:
        return False

def getMaxIndices(numberList):
    maxNumber = max(numberList)
    maxList = []
    for i in range(len(numberList)):
        if numberList[i] == maxNumber:
            maxList.append(i)
    return maxList

def calcTechCost(techList):

    return calcTotTechDist(techList) * TechnicianDistanceCost + sum(calcTechsPerDay(techList)) * TechnicianDayCost + calcIndividualTechsUsed(techList) * TechnicianCost

def calcTruckCost(truckList):

    return calcTrucksPerDay(truckList) * TruckDistanceCost + sum(calcTrucksPerDay(truckList)) * TruckDayCost + max(calcTrucksPerDay(truckList)) * TruckCost

def calcDelayCost(truckList, techSchedule):
    delayCost = 0
    for day in truckList:
        for route in day:
            for request in route:
                installDay= getInstallationDay(request, techSchedule)
                delayCost+= (day-installDay-1)*request.delayPenalty

    return delayCost

def calcTotalCost(truckList, techSchedule):

    return calcTechCost(techSchedule) + calcTruckCost(truckList) + calcDelayCost(truckList, techSchedule)

def getInstallationDay(request, techSchedule):

    for day in techSchedule:
        for e in day:
            route=e[1].seq
            if request.ID in route.seq:
                return day
    print('ERROR in installation day')
    return 0

def calcTrucksPerDay(truckList):
    numOfTrucks = []
    for i in range(len(truckList)):
        numOfTrucks.append(len(truckList[i]))
    return (numOfTrucks)

def calcTechsPerDay(techList):
    numOfTechs = []
    for i in range(len(techList)):
        numOfTechs.append(len(techList[i]))
    return (numOfTechs)


def calcIndividualTechsUsed(techList):
    techSet = set()
    for day in techList:
        for e in day:
            techSet.add(e[0])
    return len(techSet)

def calcTotTruckDist(truckList):
    totalTruckDist = 0
    for routes in truckList:
        totalTruckDist += getDistanceOfRoute(routes)
    return (totalTruckDist)

def calcTotTechDist(techList):
    totalTechDist = 0
    techRoutes = []
    for routes in techList:
        for route in routes:
            techRoutes.append(route[1])
    totalTechDist += getDistanceOfRoute(techRoutes)
    return (totalTechDist)

def getMainList(routes):
    mainList = [[] for i in range(Days+1)]
    for r in routes:
        index=r.day
        mainList[index].append(r)
    return (mainList)

#creates a dictionary with key:requestID and value:[request,route], this is used as input for the improvements algorithm
def getReqRouteDict(mainList,iteration):
    requestDict={}
    for i in range(Days+1):
        for route in mainList[i]:
            for request in route.seq:
                requestDict[request.ID] = [request,route]
    return (requestDict)

#creates a dictionary where reqRouteDict is transformed so that it can be used for the technicians algorithm, key:day and value:[requests]
def transformReqToTime(reqRouteDict):
    requestDict = {}
    for i in range(1,Days+1):
        requests = []
        for j in range(1,len(reqRouteDict)+1):
            if reqRouteDict[j][1].day == i:
                requests.append(reqRouteDict[j][0])

        requestDict[i+1] = requests
    return (requestDict)

#creates a dictionary for the technicians algorithm, key:day and value:[requests], this method can be used when improvements algorithm is not used
def getReqDict(mainList):
    requestDict={}
    for i in range(2,Days+1):
        requests =[]
        for route in mainList[i-1]:
            for request in route.seq:
                requests.append(request)
        requestDict[i]=requests
    return (requestDict)

#transform the reqRouteDict back to the main list for output, a list of routes sorted by day
def backToMainList(reqDict):
    mainList = [[] for i in range(Days + 1)]
    for i in range(1,Days+1):
        routes = []
        for j in range(1,len(reqDict)+1):
            if reqDict[j][1].day == i:
                routes.append(reqDict[j][1])
        routes = list(set(routes))

        for route in routes:
            mainList[i].append(route)
    return mainList

############OPERATIONS FROM HERE############:


MERGE_ROUTES=False

get_size_per_request()     #Assigns to each request the total size of the request
Distances= getDistMatrix() #Builds distance matrix


#---------------TRUCKS--------------
# truckRoutes = combQuickSavings(iterations=100)
truckRoutes=QuickRouteAlgorithm(1000,2)
#truckRoutes=savingsAlgorithm(timeWindow=True)

#----------------ROUTE OPTIMIZER--------------
mainList = getMainList(truckRoutes)

'''
for day in truckRouteList:
    for route in day:
        print("day", route.day)
        route.printSeq()
'''

#reqRouteDict = getReqRouteDict(truckRouteList)

requestDict = getReqDict(mainList)
techRoutes = techniciansSchedule(requestDict)

(mainList,techRoutes) = improveTruckSolution(mainList,techRoutes,10)

'''
for day in techRoutes:
    for route in day:
        print(route[1].day)
        route[1].printSeq()
'''
#reqRouteDict=improveTruckSolution(reqRouteDict,200)

#mainList = backToMainList(reqRouteDict)


#requestDict = transformReqToTime(reqRouteDict)


#---------------TECHNICIANS-----------------

# t = time.time()


#print(techRoutes)
# elapsed = time.time() - t

printSolution()
# print("SECONDS:",elapsed, '\n')

# #run the solutionfile to get solutioncost:
var = os.system('python3 SolutionVerolog2019.py ' + '-s ' +"SOLUTION_"+str(File[-5:-4])+".txt " + '-i ' + File)










