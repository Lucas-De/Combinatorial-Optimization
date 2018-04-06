from InstanceVerolog2019 import passInstance
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

File= "Instances/STUDENT005.txt"
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
        for r in RoutesList:
            x=[Locations[0].X]+[Locations[i-1].X for i in [req.customerLocID for req in r.seq]]+[Locations[0].X]
            y=[Locations[0].Y]+[Locations[i-1].Y for i in [req.customerLocID for req in r.seq]]+[Locations[0].Y]
            plt.plot(x,y,c='k',linewidth=0.5)

    plt.show()


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
        def __init__(self):
            self.seq = []
            self.dist = 0
            self.load = 0

        def removeAt(self,i):

            removed=self.seq[i]
            cancelledLoad=removed.totalSize
            self.load = self.load - cancelledLoad
            if(len(self.seq)==1):
                self.seq=[]
                return (removed, 0, self.load)
            if(i==len(self.seq)-1):
                a= self.seq[i-1].customerLocID
                b= self.seq[i].customerLocID
                dist = self.dist - Distances[a-1][b-1] - Distances[b-1][0] + Distances[a-1][0]
            elif(i==0):
                b= self.seq[i].customerLocID
                c= self.seq[i+1].customerLocID
                self.dist = self.dist - Distances[0][b-1] - Distances[b-1][c-1] + Distances[0][c-1]
            else:
                a= self.seq[i-1].customerLocID
                b= self.seq[i].customerLocID
                c= self.seq[i+1].customerLocID
                self.dist = self.dist - Distances[a-1][b-1] - Distances[b-1][c-1] + Distances[a-1][b-1]
         
            if(Route.Lock==True): return (removed, dist, load)

            if(i==len(self.seq)-1): temp =self.seq[:i]
            elif(i==0): temp =self.seq[1:]
            else: temp= self.seq[:i]+ self.seq[i+1:]
            self.seq=temp


        def add(self,request,i=None):
            if(i==len(self.seq) or i is None):
                return self.addLast(request)
            elif(i==0):
                return self.addFirst(request)
            else:
                newLoc=request.customerLocID
                a= self.seq[i-1].customerLocID
                b= self.seq[i].customerLocID
                dist= self.dist - Distances[a-1][b-1] + Distances[newLoc-1][a-1] + Distances[newLoc-1][b-1]
                load= self.load + request.totalSize
                Valid=self.Valid(dist, load)
                if(Route.Lock==True): return (Valid, dist, load)

                if(Valid): 
                    self.dist=dist
                    self.load=load
                    temp =self.seq[:i] + [request] + self.seq[i:]
                    self.seq=temp
                    return (Valid, dist, load)
                else:
                    return (Valid, dist, load)

        def Valid(self,dist,load):
            if( dist>TruckMaxDistance  or  load> TruckCapacity):
                return(False)
            else:
                return(True)

        def addLast(self,request): 
            newLast=request.customerLocID
            if len(self.seq)>0: 
                oldLast=self.seq[-1].customerLocID
                dist= self.dist + Distances[oldLast-1][newLast-1] + Distances[newLast-1][0] - Distances[0][oldLast-1]
            else: 
                dist= 2* Distances[newLast-1][0]
            load=self.load + request.totalSize
            Valid=self.Valid(dist, load)
            if(Route.Lock==True): return (Valid, dist, load)

            if(Valid): 
                self.dist=dist
                self.load=load
                self.seq.append(request)
                return(True)
            else:
                return(False)

        def addFirst(self,request):
            newFirst=request.customerLocID
            oldFirst=self.seq[0].customerLocID
            dist= Distances[0][newFirst-1] + Distances[newFirst-1][oldFirst-1] - Distances[0][oldFirst-1]
            load=request.totalSize+self.load
            Valid=self.Valid(dist, load)
            if(Route.Lock==True): return (Valid, dist, load)

            if(Valid): 
                self.dist=dist
                self.load=load
                self.seq= [request]+self.seq
                return(True)
            else:
                return(False)

        def printSeq(self):
            print([i.ID for i in self.seq])

        def containsRequest(self,requestID):
            if requestID in self.seq:
                return (True)
            else:
                return (False)

        def isExtreme(self,requestID):
            if self.seq[0] == requestID or self.seq[len(self.seq) - 1] == requestID:
                return (True)
            else:
                return (False)

def initRoutes():
    routes=[]
    for i in range(0,len(Requests)):
        r=Route()
        r.add(Requests[i])
        r.lock = True
        routes.append(r)
    return (routes)


get_size_per_request()     #Assigns to each request the total size of the request
Distances= getDistMatrix() #Builds distance matrix
# Routes= initRoutes()     #Creates a list of routes containing one Route for each request


# r=Route()             #Creates a Route object with following attributes:
                        #Route Distance: r.dist
                        #Route Load (size of goods transported on the route) : r.load
                        #Sequence of requests visited: r.seq

# r.add(Requests[0])    #Adds Request[0] to the end of Route r if and only it meets the constraints.
                        #Returns: (True if constraints are met, r.dist, r.load)

# r.add(Requests[1],1)  #Adds Request[1] to position 1 in Route r if and only it meets the constraints.
                        #Returns: (True if constraints are met, r.dist, r.load)

# r.removeAt(0)         #Removes Request at position 3 in Route r
                        #Returns: (removed Request, r.dist, r.load)



# Route.Lock=True       #Locks all routes to their current states:
                        #
                        #add() returns the values of the updated route (True if constraints are met, r.dist, r.load) but does not update the route
                        #
                        #remove() returns the values of the updated route (removed Request, r.dist, r.load) but does not update the route
                        #
                        #Setting Route.Lock=True can be used to test different route updates without changing the cureent solution

#r.printSeq()            #Prints route sequance

#showMap(r)              #Shows a map of the Route

def getSavingsList(day):
    availableRequests = []
    nrReq = len(Requests)

    for i in range(nrReq):
        if Requests[i].fromDay <= day <= Requests[i].toDay:
            availableRequests.append(Requests[i])

    s = []
    nrAvReq = len(availableRequests)
    for i in range(nrAvReq):
        for j in range(i,nrAvReq):
            if i != j:
                s.append([availableRequests[i].ID,availableRequests[j].ID,Distances[0][availableRequests[i].customerLocID - 1] + Distances[availableRequests[j].customerLocID - 1][0] - Distances[availableRequests[i].customerLocID - 1][availableRequests[j].customerLocID - 1]])
    return (s)

def savingsAlgorithm():
    routes = initRoutes()
    routes[1].add(Requests[4])
    routes[1].printSeq()
    showMap(routes)


    for i in range(Days):
        currentDay = i + 1
        savings = getSavingsList(currentDay)
        sortedSavings = sorted(savings, key=lambda x:x[2], reverse=True)
        #print(sortedSavings)

        while len(sortedSavings) > 0:
            newLeg = sortedSavings.pop()
            req1ID = newLeg[0]
            req2ID = newLeg[1]

            r1Index = r2Index = None
            req1IsExtreme = req2IsExtreme = False

            for i in range(len(routes)):
                if routes[i].containsRequest(req1ID):
                    r1Index = i
                    if routes[i].isExtreme(req1ID):       #isExtreme() has to be made
                        req1IsExtreme = True
                        #print(req1IsExtreme)
                if routes[i].containsRequest(req2ID):
                    r2Index = i
                    if routes[i].isExtreme(req2ID):
                        req2IsExtreme = True

            if (r1Index != r2Index) and req1IsExtreme and req2IsExtreme:
                newRoute = merge(routes[r1Index],routes[r2Index])               #merge has to be made
                routes.pop(r1Index)
                routes.pop(r2Index)
                routes.append(newRoute)

savingsAlgorithm()


