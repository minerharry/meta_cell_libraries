##FMI: two axis (x and y) - total displacement in the axis across a track divided by the total distance traveled by the cell
from collections import UserDict
import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union
from fastprogress import master_bar,progress_bar
import numpy as np
from pandas import DataFrame
import pandas as pd


def scale_tracks(in_tracks:Dict[int,Dict[int,pd.DataFrame]],centertype:str,distance_per_pixel:float,time_per_frame:float):
    centerx = centertype + 'x';
    centery = centertype + 'y';
    scaled_tracks:Dict[int,Dict[int,pd.DataFrame]] = {}
    for movie,tracks in in_tracks.items():
        scaled_tracks[movie] = stracks = {}
        for trackid,track in tracks.items():
            stracks[trackid] = strack = track.copy()
            strack[centerx] *= distance_per_pixel
            strack[centery] *= distance_per_pixel
            strack['area'] *= distance_per_pixel**2
            strack['time'] = (strack['frame'] - 1)*time_per_frame
    return scaled_tracks

def dist(x1,x2,y1,y2):
  return math.sqrt((x1-x2)**2+(y1-y2)**2)


class ExperimentAnalysis(Mapping[str,Any]):
    scaled_tracks:Dict[int,Dict[int,DataFrame]] = {};

    FMI: Dict[int,Dict[int,Tuple[float,float]]] = {}; #{movie, {trackid:(FMI.x,FMI.y)}}
    Persistence: Dict[int,Dict[int,float]] = {}; #{movie, {trackid:Persistence}}
    trackVelocity: Dict[int,Dict[int,Tuple[float,float,float]]] = {}; #{movie, {trackid:(velocityX,velocityY,velocityMag)}}
    trackDisplacement: Dict[int,Dict[int,Tuple[float,float,float]]] = {}; #{movie, {trackid:(displacementX,displacementY,displacementDist)}}
    trackLength: Dict[int,Dict[int,float]] = {}; #{movie, {trackid:tracklength}}
    trackTime: Dict[int,Dict[int,int]] = {}; #{movie, {trackid:trackTime}}

    avgFMI: Dict[int,Tuple[float,float]] = {}; #{movie,(avgX,avgY)};
    avgPersistence: Dict[int,float] = {}; #{movie,average};
    avgVelocity: Dict[int,Tuple[float,float,float]] = {}; #{movie,(averageX,averagyY,averageMag)};
    avgDisplacement: Dict[int,Tuple[float,float,float]] = {}; #{movie,(avgX,avgY,avgDist)}
    avgTracklength: Dict[int,float] = {}; #{movie,average}
    avgTracktime: Dict[int,float] = {};

    _keys = ([k for k,v in locals().items() if not k.startswith("_")]) #all typed variables

    def __init__(self,analysis:Dict[str,Any]) -> None:
        for k in analysis:
            if k in self._keys:
                setattr(self,k,analysis[k])
            else:
                raise KeyError(k)
        
        for k in self._keys:
            if k not in analysis:
                raise ValueError(f"Analysis missing key {k}")

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)
    
    def keys(self):
        return self._keys
    
    def __contains__(self, __o: object) -> bool:
        return __o in self._keys;

    def __getitem__(self,key:str):
        if key in self._keys:
            return getattr(self,key)
        else:
            raise KeyError(key)

def analyze_experiment_tracks(scaled_tracks:Dict[int,Dict[int,DataFrame]],centertype:str,do_progressbar:bool=True)->ExperimentAnalysis:
    FMI: Dict[int,Dict[int,Tuple[float,float]]] = {}; #{movie, {trackid:(FMI.x,FMI.y)}}
    Persistence: Dict[int,Dict[int,float]] = {}; #{movie, {trackid:Persistence}}
    trackVelocity: Dict[int,Dict[int,Tuple[float,float,float]]] = {}; #{movie, {trackid:(velocityX,velocityY,velocityMag)}}
    trackDisplacement: Dict[int,Dict[int,Tuple[float,float,float]]] = {}; #{movie, {trackid:(displacementX,displacementY,displacementDist)}}
    trackLength: Dict[int,Dict[int,float]] = {}; #{movie, {trackid:tracklength}}
    trackTime: Dict[int,Dict[int,int]] = {}; #{movie, {trackid:trackTime}}

    avgFMI: Dict[int,Tuple[float,float]] = {}; #{movie,(avgX,avgY)};
    avgPersistence: Dict[int,float] = {}; #{movie,average};
    avgVelocity: Dict[int,Tuple[float,float,float]] = {}; #{movie,(averageX,averagyY,averageMag)};
    avgDisplacement: Dict[int,Tuple[float,float,float]] = {}; #{movie,(avgX,avgY,avgDist)}
    avgTracklength: Dict[int,float] = {}; #{movie,average}
    avgTracktime: Dict[int,float] = {}; #{movie,average}

    result = {}
    result.update(locals())
    del result['centertype']
    del result['result']
    del result['do_progressbar']



    centerx = centertype + 'x';
    centery = centertype + 'y';

    bar = master_bar(scaled_tracks.items()) if do_progressbar else scaled_tracks.items()
    for movie,tracks in bar:
        FMI[movie] = {};
        Persistence[movie] = {};
        trackVelocity[movie] = {};
        trackDisplacement[movie] = {}
        trackLength[movie] = {}
        trackTime[movie] = {}

        numPoints = 0;
        FMI_accX = 0;
        FMI_accY = 0;
        Persistence_acc = 0;
        Velocity_acc = np.array([0,0,0]);
        Displacement_acc = np.array([0,0,0]);
        Tracklength_acc = 0;
        Tracktime_acc = 0

        for id,data in (progress_bar(tracks.items(),parent=bar) if do_progressbar else tracks.items()):
            numPoints += 1;

            start = data.iloc[0];
            end = data.iloc[-1];
        
            ##Get accumulated distance (total movement within track)
            accDist = 0;
            accVel = np.array([0,0,0]);
            prevpos = (start[centerx],start[centery]);
            prevtime = start['time']
            for x,y,t in zip(data.iloc[1:][centerx],data.iloc[1:][centery],data.iloc[1:]['time']):
                x1,y1 = prevpos;
                dt = (t-prevtime)
                accVel = np.sum([accVel,[(x-x1)/dt,(y-y1)/dt,dist(x,x1,y,y1)/dt]],axis=0);
                accDist += math.sqrt((y-y1)**2+(x-x1)**2);
                prevpos = (x,y);
                prevtime = t
        
            ##Get vertical, horizontal displacement
            xDisp = (end[centerx] - start[centerx]);
            yDisp = (end[centery] - start[centery]);

            ##Get individual cell FMI
            xMI = xDisp/accDist if accDist != 0 else xDisp*0;
            yMI = yDisp/accDist if accDist != 0 else yDisp*0;

            FMI[movie][id] = (xMI,yMI);

            ## Get net cell distance
            netDist = math.sqrt(xDisp**2 + yDisp**2);

            ## Get Persistence
            direct = netDist/accDist if accDist != 0 else 0;
            Persistence[movie][id] = direct;

            ##Get Average Velocity
            avgTrackVel = accVel/len(data);
            trackVelocity[movie][id] = tuple(avgTrackVel);
            
            #Get Displacements
            trackDisplacement[movie][id] = (xDisp,yDisp,netDist)
            
            #Get Tracklength
            trackLength[movie][id] = accDist
            
            #Get Tracktime
            trackTime[movie][id] = end['time']-start['time']

            ##Accumulate FMI, Persistence, and Velocity
            FMI_accX += xMI;
            FMI_accY += yMI;
            Persistence_acc += direct;
            Velocity_acc = np.sum([Velocity_acc,avgTrackVel],axis=0)
            Displacement_acc = np.sum([Displacement_acc,(xDisp,yDisp,netDist)],axis=0)
            Tracklength_acc += accDist
            Tracktime_acc += trackTime[movie][id]

        if (numPoints > 0):
            avgFMI[movie] = (FMI_accX/numPoints,FMI_accY/numPoints);
            avgPersistence[movie] = Persistence_acc/numPoints;
            avgVelocity[movie] = tuple(Velocity_acc/numPoints);
            avgDisplacement[movie] = tuple(Displacement_acc/numPoints);
            avgTracklength[movie] = Tracklength_acc/numPoints
            avgTracktime[movie] = Tracktime_acc/numPoints
        else:
            avgFMI[movie] = (0,0);
            avgPersistence[movie] = 0;
            avgVelocity[movie] = (0,0,0);
            avgDisplacement[movie] = (0,0,0);
            avgTracklength[movie] = 0
            avgTracktime[movie] = 0

    return ExperimentAnalysis(result)

def save_tracks_analysis_csv(local_path:Union[str,Path,os.PathLike],tracks_analysis:Union[ExperimentAnalysis,Dict[str,Any]],distance_unit:str,time_unit:str):
    with open(local_path,"w", newline='') as file:
        fieldnames = ['movie', 'trackid','FMI.x','FMI.y',f'Velocity.x ({distance_unit}/{time_unit})',f'Velocity.y ({distance_unit}/{time_unit})',f'Speed ({distance_unit}/{time_unit})','Persistence',f'Displacement.x ({distance_unit})',f'Displacement.y ({distance_unit})',f'Displacement Distance ({distance_unit})',f'Tracklength ({distance_unit})',f'Track time ({time_unit})'];
        writer = csv.DictWriter(file, fieldnames=fieldnames);
        writer.writeheader()
        movies = tracks_analysis["scaled_tracks"].keys()
        for movie in movies:
            writer.writerow(dict(zip(fieldnames,
                                    [movie,"average",
                                    tracks_analysis["avgFMI"][movie][0],
                                    tracks_analysis["avgFMI"][movie][1],
                                    tracks_analysis["avgVelocity"][movie][0],
                                    tracks_analysis["avgVelocity"][movie][1],
                                    tracks_analysis["avgVelocity"][movie][2],
                                    tracks_analysis["avgPersistence"][movie],
                                    tracks_analysis["avgDisplacement"][movie][0],
                                    tracks_analysis["avgDisplacement"][movie][1],
                                    tracks_analysis["avgDisplacement"][movie][2],
                                    tracks_analysis["avgTracklength"][movie],
                                    tracks_analysis["avgTracktime"][movie]
                                    ])));
        for movie in progress_bar(movies):
            for id in tracks_analysis["scaled_tracks"][movie].keys():
                writer.writerow(dict(zip(fieldnames,
                                    [movie,id,
                                    tracks_analysis["FMI"][movie][id][0],
                                    tracks_analysis["FMI"][movie][id][1],
                                    tracks_analysis["trackVelocity"][movie][id][0],
                                    tracks_analysis["trackVelocity"][movie][id][1],
                                    tracks_analysis["trackVelocity"][movie][id][2],
                                    tracks_analysis["Persistence"][movie][id],
                                    tracks_analysis["trackDisplacement"][movie][id][0],
                                    tracks_analysis["trackDisplacement"][movie][id][1],
                                    tracks_analysis["trackDisplacement"][movie][id][2],
                                    tracks_analysis["trackLength"][movie][id],
                                    tracks_analysis["trackTime"][movie][id],
                                    ])));