# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

# import the necessary packages
from typing import Callable, Generic, List, Literal, TypeVar, Union, Dict
from scipy import optimize
from scipy.spatial import distance as dist
import numpy as np

T = TypeVar('T');
class CentroidTracker(Generic[T]):
    class Dum: ##I hate this so much
        def __init__(self,val):
            self.val = val;
            
    def __init__(self, 
            distance_metric:Callable[[T,T],float],
            frame_filter:Union[None,Callable[[T,T,int,float],bool]]=None,
            maxDisappearedFrames:int=50,
            minPersistenceFrames:int=20,
            optimization_type:Literal["nearest_neighbors","linear_sum_assignment"]="linear_sum_assignment"):
        # initialize the next unique object ID along with three ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid, number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.distance_metric = distance_metric;
        self.frame_filter = frame_filter;
        self.nextObjectID = 0
        self.objects:Dict[int,T] = dict()
        self.disappeared:Dict[int,int] = dict()
        self.active_time:Dict[int,int] = dict()
        self.optim_type = optimization_type

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappearedFrames

        # store how long an object needs to be actively tracked
        # to keep its reference active when it "disappears"
        self.minPersistence = minPersistenceFrames

    def register(self, centroid:T):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.active_time[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids:List[T], allow_new:bool = True):

        # check to see if the list of input object centroids is empty
        if len(inputCentroids) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()).copy():
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared or self.active_time[objectID] < self.minPersistence:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects


        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])


        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            if self.distance_metric is None:
                D = dist.cdist(np.array(objectCentroids), np.array(inputCentroids));
            else:
                ##the object being used must NOT be arraylike or iterable or numpy will expand it. 
                ##Since we may be using dataframes or arrays, wrap in a noniterable class to fix this
                ##Once again: hate
                D = dist.cdist(np.array([self.Dum(o) for o in objectCentroids]).reshape(-1,1), np.array([self.Dum(i) for i in inputCentroids]).reshape(-1,1),metric=lambda x,y: self.distance_metric(x[0].val,y[0].val));
                ##TODO: This step can maybe be sped up if distance_metric is faster because this is n^2? test to see if significantly impacts performance


            ## apply frame filter: to any pairing we deem "unsatisfactory", make it the biggest value so it's never prioritized
            ## also save said value so that when pulled as a last resort, the match can be discarded
            biggest = np.max(D) + 2;
            if self.frame_filter:
                filtered_out = [i for i in np.ndindex(*D.shape) if not(self.frame_filter(
                    objectCentroids[i[0]],
                    inputCentroids[i[1]],
                    self.disappeared[objectIDs[i[0]]],
                    D[i]
                    ))];
                try:
                    for i in filtered_out:
                        D[i] = biggest;
                except IndexError as e:
                    print(filtered_out);
            
            ##NOTE: from profiling - the filtering and distance-calculation steps have pretty similar performance costs; 
            # distance is about 45% of execution time, and filtering is about 42%. The actual solving of the distance matrix is essentially negligible.
            

            ## now we have our cost matrix D from which we can do assignment. The simplest possible way, as implemented originally, is a nearest neighbors:
            if self.optim_type == "nearest_neighbors":
                links, newIdx, lostIdx = self.link_nearest_neighbors(D,biggest)
            elif self.optim_type == "linear_sum_assignment":
                links, newIdx, lostIdx = self.link_linear_sum_assignment(D,biggest)
            else:
                raise ValueError(f"Optimization type {self.optim_type} is invalid")

            for row,col in links:
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.active_time[objectID] += 1;
                self.disappeared[objectID] = 0

            # loop over the lost indexes
            for row in lostIdx:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                # check to see if the number of consecutive
                # frames for which the object has been marked "disappeared"
                # warrants deregistering the object
                # also deregister if the object has not been actively tracked
                # for enough time to warrant keeping it
                if self.disappeared[objectID] > self.maxDisappeared or self.active_time[objectID] < self.minPersistence:
                    self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            if allow_new:
                for col in newIdx:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects

    def link_linear_sum_assignment(self,D:np.ndarray,nullvalue):
        rowidx,colidx = optimize.linear_sum_assignment(D)

        usedrows = set()
        usedcols = set()
        links = []
        for row,col in zip(rowidx,colidx):
            if D[row,col] != nullvalue:
                links.append((row,col))
                usedrows.add(row)
                usedcols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedrows)
        unusedCols = set(range(0, D.shape[1])).difference(usedcols)
        
        
        return links, unusedCols, unusedRows
    
        ###TODO: TEST TO ENSURE CORRECTNESS
        # the issue here is that for an nxm cost matrix, linear_sum_assignment will always try to link every 
        # element of the shorter dimension to one of the longer dimension. The filtering system needs to be able to account for fails,
        # and to prioritize the minimum cost. I **think** this will work as-is, but I'd really like a proof


    def link_nearest_neighbors(self,D:np.ndarray,nullvalue:float):
        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value as at the *front* of the index list
        rows = np.min(D,axis=1).argsort()


        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = np.argmin(D,axis=1,)[rows]

        # the resulting list of pairs (from zip(rows,cols)) gives us 
        # the coordinates of the smallest value in the table,
        # the coordinates of the next smallest value including the 


        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        usedRows = set()
        usedCols = set()

        links = []

        # loop over the combination of the (row, column) index
        # tuples
        for (row, col) in zip(rows, cols):
            # print(D[row,col],row,col);
            if (D[row,col].item()) == nullvalue: #cell match is not allowed
                # print("bigelow");
                continue;
            # if we have already examined either the row or
            # column value before, ignore it - that object has already been matched
            if row in usedRows or col in usedCols:
                continue

            # otherwise, grab the object ID for the current row,
            # set its new centroid, reset the disappeared counter,
            # and increment the active time for that object
            links.append((row,col))


            # indicate that we have examined each of the row and
            # column indexes, aka the existing and new objects respectively
            usedRows.add(row)
            usedCols.add(col)

        # compute both the row and column index we have NOT yet
        # examined
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)


        return links,unusedCols,unusedRows
