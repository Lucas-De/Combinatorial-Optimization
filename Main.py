from InstanceVerolog2019 import passInstance
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random
import time

random.seed(2018)

File= "Instances/CO2018_1.txt"
Instance=passInstance(File,False)

Dataset = Instance.Dataset
Name = Instance.Name

Days =  Instance.Days
TruckCapacity = Instance.TruckCapacity
TruckMaxDistance = Instance.TruckMaxDistance
TruckDistanceCost =  Instance.TruckDistanceCost           
TruckDayCost = Instance.TruckDistanceCost
TruckCost = Instance.TruckCost 
TechnicianDistanceCost =   Instance.TechnicianDistanceCost           
TechnicianDayCost = Instance.TechnicianDayCost
TechnicianCost = Instance.TechnicianCost


 
Machines=Instance.Machines       #Machine objects have values: ID, size, idlePenalty
Requests=Instance.Requests       #Request objects have values: ID, customerLocID, fromDay, toDay, machineID, amount, totalSize
Locations=Instance.Locations     #Locations objects have values: ID, X, Y
Technicians=Instance.Technicians #Technicians objects have values: ID, locationID, maxDayDistance, maxNrInstallations, capabilities]


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
            elif routeType == 'technician':
                self.nrMachines = 0
                self.technician = technician
                self.homebase = technician.locationID - 1

        def removeAt(self,i,routeType):
            if routeType == 'truck':
                removed=self.seq[i]
                cancelledLoad=removed.totalSize
                self.load = self.load - cancelledLoad
            elif routeType == 'technician':
                self.nrMachines = self.nrMachines - 1

            if(len(self.seq)==1):
                self.seq=[]
                if routeType == 'truck':
                    return (removed, 0, self.load)
                elif routeType == 'technician':
                    return (removed, 0, self.nrMachines)

            if(i==len(self.seq)-1):
                a= self.seq[i-1].customerLocID
                b= self.seq[i].customerLocID
                dist = self.dist - Distances[a-1][b-1] - Distances[b-1][self.homebase] + Distances[a-1][self.homebase]
            elif(i==0):
                b= self.seq[i].customerLocID
                c= self.seq[i+1].customerLocID
                self.dist = self.dist - Distances[self.homebase][b-1] - Distances[b-1][c-1] + Distances[self.homebase][c-1]
            else:
                a= self.seq[i-1].customerLocID
                b= self.seq[i].customerLocID
                c= self.seq[i+1].customerLocID
                self.dist = self.dist - Distances[a-1][b-1] - Distances[b-1][c-1] + Distances[a-1][b-1]
         
            if(Route.Lock==True):
                if routeType == 'truck':
                    return (removed, dist, self.load)
                elif routeType == 'technician':
                    return (removed, dist, self.nrMachines)

            if(i==len(self.seq)-1): temp =self.seq[:i]
            elif(i==0): temp =self.seq[1:]
            else: temp= self.seq[:i]+ self.seq[i+1:]
            self.seq=temp

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

        def printSeq(self):
            print([i.ID for i in self.seq])

        def containsRequest(self,requestID):
            contains = False
            for i in range(len(self.seq)):
                if self.seq[i].ID == requestID:
                    contains = True
            return (contains)

        def isExtreme(self,requestID):
            if self.seq[0].ID == requestID:
                self.seq.reverse()
                return (True)
            elif self.seq[len(self.seq) - 1].ID == requestID:
                return (True)
            else:
                return (False)

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

def initRoutes(technician,closestReq,routeType,avRequests=None):
    routes=[]
    if avRequests != None:
        requests = avRequests
    else:
        requests = Requests

    if routeType == 'truck':
        for i in range(0,len(requests)):
            r=Route(routeType)
            r.add(Requests[i],routeType)
            routes.append(r)
    elif routeType == 'technician':
        for i in range(len(closestReq)):
            r = Route(routeType,technician)
            r.add(closestReq[i], routeType)
            routes.append(r)
    return (routes)

get_size_per_request()     #Assigns to each request the total size of the request
Distances= getDistMatrix() #Builds distance matrix

def getCosts(RouteList):
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

#Savings Algorithm (Doesn't consider time windows): prints routing solution map
def savingsAlgorithm(timeWindow=False,technician=None,closestReq=None,routeType='truck'):
    if (timeWindow):
        totRequests = copy.deepcopy(Requests)
        totalRoutes = []
        for i in range(Days):
            day = i + 1
            currAvRequests = []
            for request in totRequests:
                if request.fromDay <= day:
                    currAvRequests.append(request)
            routes = initRoutes(technician, closestReq, routeType,currAvRequests)
            possible = True
            while (possible):
                possible = mergeBestPair(routes, routeType)
            totalRoutes.append(routes)
    else:
        totalRoutes = initRoutes(technician,closestReq,routeType)
        possible=True
        while(possible):
            possible=mergeBestPair(totalRoutes,routeType)

    if len(totalRoutes) == 1 and totalRoutes[0].seq == []:
        totalRoutes = None

    return(totalRoutes)

#Initial algorithm to create a schedule for the technicians, input is a list of available requests for each day
def techniciansSchedule(requestDict):
    availableTech = Technicians
    nonAvailableTech = None
    finalRouteList = []
    currentRequests = []

    for i in range(1,Days+1):
        if i in requestDict:
            dayRequests=requestDict[i]
            for request in dayRequests:
                currentRequests.append(request)

        currAvailableTech = copy.deepcopy(availableTech)
        dailyRouteList = []

        if currAvailableTech != None and currentRequests != None:
            while len(currAvailableTech) > 0 and len(currentRequests) > 0:
                techList = []
                for j in range(len(currAvailableTech)):
                    technician = currAvailableTech[j]
                    closestReq = computeClosestReq(technician,currentRequests)

                    if len(closestReq) > 0:
                        avgDistance = computeAVG(column(closestReq,1))
                        techList.append((technician,column(closestReq,0),avgDistance))

                if len(techList) > 0:
                    optimalTech = min(techList,key=lambda x:x[2])
                    routes = savingsAlgorithm(technician=optimalTech[0],closestReq=optimalTech[1],routeType='technician')

                    if routes != None:
                        finalRoute = getLargestRoute(routes)
                        finalRoute.day = i

                        for k in range(len(finalRoute.seq)):
                            currentRequests.remove(finalRoute.seq[k])

                    technician = optimalTech[0]
                    dailyRouteList.append((technician.ID,finalRoute))  # append tech ID and daily routes to list

                    if technician.stillAvailable():
                        technician.prevWorkDays += 1
                    else:
                        technician.breakDaysLeft = 3
                        technician.prevWorkDays = 0
                        availableTech.remove(technician)
                        nonAvailableTech.append(technician)

                currAvailableTech.remove(technician)

        if nonAvailableTech != None:
            for l in range(len(nonAvailableTech)):
                technician = nonAvailableTech[i]
                technician.breakDaysLeft -= 1

                if technician.availableAgain():
                    availableTech.append(technician)
                    nonAvailableTech.remove(technician)

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
        n += currentReq[0].amount

        if n <= technician.maxNrInstallations:
            nClosest.append(currentReq)
        else:
            n -= currentReq[0].amount

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

#QuickRoute (I made this up): Prints routing solution which considers time windows
# This is a stochastic algorithm and requires being run multiple times to get a good solution
def QuickRoute(method=1):
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
    return(routes)

def QuickRouteAlgorithm(iterations=1,method=2):
    optCost=math.inf
    optRoutes=[]

    for i in range(iterations):
        routes=QuickRoute(method)
        cost=getCosts(routes)
        if(cost<optCost):
            optCost=cost
            optRoutes=routes
    return(optRoutes)



r=Route('truck')
r.add(Requests[1],'truck')
r.add(Requests[2],'truck')


r2=Route('truck')
r2.add(Requests[0],'truck')
r2.add(Requests[18],'truck')

print("r.dist=",r.dist)

r.mergeWith('truck',r2,0)

print(getCosts([r]))
# showMap([[r]])
print([k.ID for k in r.seq])


#Runs Savings
#t = time.time()
#routes=savingsAlgorithm(timeWindow=True)
#elapsed = time.time() - t
#print(elapsed)
#print("Savings Alg Total Dist:",getCosts(routes))

#for route in routes:
#    for i in route:
#        print(i.seq)
    #showMap(route)

# Run QuickRoute
t = time.time()
truckRoutes=QuickRouteAlgorithm(100,2)
elapsed = time.time() - t
print(elapsed)
print(getCosts(truckRoutes))
# for b in r:
#   print(b.day,[c.ID for c in b.seq])
# showMap(r)



mainList = [[] for i in range(Days+1)]
for r in truckRoutes:
    index=r.day
    mainList[index].append(r)



'''
requestList = []
for i in range(1,Days+1):
    currDayList = []
    for j in range(len(truckRoutes)):
        currRoute = truckRoutes[j]
        if currRoute.day == i:
            for l in range(len(currRoute.seq)):
                currDayList.append(currRoute.seq[l])
    requestList.append(currDayList)
'''

requestDict={}
for i in range(2,Days+1):
    abc=[]
    for route in mainList[i-1]:
        for request in route.seq:
            abc.append(request)
    requestDict[i]=abc

techRoutes = techniciansSchedule(requestDict)

'''
for routes in techRoutes:
    for route in routes:
        print("day", route[1].day)

        print(route[0], end="")
        route[1].printSeq()
'''

#showMap(finalRoutes, Tech=True)


def printSolution():
    print("DATASET = CO2018 freestyle")
    print("NAME = Instance", File[-5:-4])

    for i in range(1,Days+1):
        currList = []
        print ("DAY = ", i)
        print ("NUMBER_OF_TRUCKS = ", len(mainList[i]))
        for j in range(len(mainList[i])):
            print ( j+1,' '.join([str(k.ID)  for k in mainList[i][j].seq]) )
        print("NUMBER_OF_TECHNICIANS = ", len(techRoutes[i-1]))
        for j in range(len(techRoutes[i-1])):
            print ( techRoutes[i-1][j][0],' '.join([str(k.ID) for k in techRoutes[i-1][j][1].seq]) )


printSolution()

#print(len(techRoutes))


