import builtins
import math
from pathlib import Path
import datetime
from typing import DefaultDict, Dict, List, Tuple, Union
from pandas import DataFrame
import copy

wow = 7


def distance(p1:Tuple[int,int],p2:Tuple[int,int]):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2);

def apply_qc(in_tracks:Dict[int,Dict[int,DataFrame]],
        qc_logfile:Union[Path,str],
        centertype:str,
        keep:Dict[int,List[int]],
        trim:Dict[Tuple[int,int],Tuple[int,int]],
        removemov:List[int],
        exclude:List[Tuple[int,int]],
        minTrackLength:Union[None,int]=None,
        minTrackDisplacement:Union[None,float]=None,
        initialTrackDelay:Union[int,None]=None,
        )->Tuple[Dict[int,Dict[int,int]],Dict[int,Dict[int,DataFrame]]]:

    with open(qc_logfile,'w') as f:

        def printlog(*args,**kwargs):
            print(*args,**kwargs)
            log(*args,**kwargs)

        def log(*args,**kwargs):
            print(*args,file=f,**kwargs)

        log("QC started at",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        centerx,centery = centertype + 'x', centertype + 'y'

        initialTrackDelay = initialTrackDelay or None
        if initialTrackDelay is not None and initialTrackDelay < 0:
            raise ValueError("argument initialTrackDelay must be nonnegative")

        if initialTrackDelay is not None:
            if minTrackLength is not None:
                if minTrackLength < initialTrackDelay:
                    printlog("Warning: initial track delay larger than minimum track length; short tracks would get swallowed by the delay. Setting mintracklength to the delay");
                    minTrackLength = initialTrackDelay
            else:
                minTrackLength = initialTrackDelay;

        
        out_tracks=copy.deepcopy(in_tracks)
        #a sample contains movies, each movie contains tracks
        
        #initialize the status of all tracks in the sample as 1 (active)
        sampTrStatus:Dict[int,Dict[int,int]]={};
        for movie in in_tracks:
            sampTrStatus[movie] = {};
            for id in in_tracks[movie]:
                sampTrStatus[movie][id] = 1

        ###CONDITIONS ON TRACK LENGTHS
        #trim tracks
        #input: dict (trim) of elements with the form {(movie, track):(begginning frame, end frame)]
        #trim={(7,1):(1,53)}
        trims:Dict[int,Dict[int,Tuple[Union[int,None],Union[int,None]]]] = {m:{t:(None,None) for t in in_tracks[m]} for m in in_tracks};
        for (mov,track),(start,end) in trim.items():
            if track == -1:
                trims[mov] = DefaultDict(lambda: (start,end));
                log(f"applying manual trim: all tracks in movie {mov} limited to frames {start}-{end}")
            else:
                trims[mov][track] = (start,end);
                log(f"applying manual trim: track {track} in movie {mov} limited to frames {start}-{end}")
            

        for mov,t in trims.items():
            for (track,(start,end)) in t.items():

                if start is None and end is None: continue
                
                #get 'frame' column
                framec=out_tracks[mov][track]['frame']
                
                ##check bounds of input frames:

                #get first and last frames of track
                firstframe = framec.iloc[0]
                lastframe = framec.iloc[-1]

                if initialTrackDelay is not None:
                    firstframe = firstframe+initialTrackDelay;
                    

                #check that input frames are in bounds
                if start is None:
                    start = firstframe
                    #only apply initial track delay if no manual start specified
                    if initialTrackDelay is not None:
                        #only apply initial delay if not the start of the movie
                        if start != 1:
                            start += initialTrackDelay
                            log(f"Applying automatic initial track delay: first frame of track {track} in movie {mov} trimmed from {start-initialTrackDelay} to {start}")
                        else:
                            log(f"track {track} in movie {mov} starts at the beginning of the movie, no track delay applied")
                elif start < firstframe or start > lastframe:
                    raise Exception(f'in movie {mov} track {track} beggining of trimming {start} is out of range {(firstframe,lastframe)}');
                
                if end is None:
                    end = lastframe
                elif end < firstframe or end > lastframe:
                    raise Exception(f'in movie {mov} track {end} end of trimming {end} is out of range {(firstframe,lastframe)}');

                if start >= end :
                    raise Exception(f'in movie {mov} track {track} end of trimming {end} is smaller or equal than beggining of trimming {start}')
                    
                #get indices of desired first and last frames
                ifirstframe = framec[framec==start].index[0]
                ilastframe = framec[framec==end].index[0]

                #trim track
                out_tracks[mov][track]=out_tracks[mov][track].loc[ifirstframe:ilastframe+1]

        
        ###CONDITIONS ON TRACKS
        if minTrackLength is not None and minTrackLength > 0:
            for imov in out_tracks:
                for itr in out_tracks[imov]:        
                    #if track length less than min length deactivate track
                    if (l := len(out_tracks[imov][itr])) < minTrackLength:
                        sampTrStatus[imov][itr]=0
                        log(f"removing track {itr} from movie {imov}: track length {l} too short")

        if minTrackDisplacement is not None:
            for imov in out_tracks:
                for itr in out_tracks[imov]:
                    #if track length less than min length deactivate track
                    startframe = out_tracks[imov][itr].head(1)
                    endframe = out_tracks[imov][itr].tail(1)
                    d = distance((startframe[centerx].iloc[0],startframe[centery].iloc[0]),(endframe[centerx].iloc[0],startframe[centery].iloc[0]))
                    if d < minTrackDisplacement:
                        sampTrStatus[imov][itr]=0
                        log(f"removing track {itr} from movie {imov}: track displacement {d} too short")
        

        #exclude tracks after visual inspection
        #exclude=[[2,3],[2,4]]
        #exclude=[[1,7]]                
        for mov,track in exclude:
            sampTrStatus[mov][track]=0
            log(f"Manually exclduing track {track} from movie {mov}")
        

        #only keep certain tracks
        #input: dict of elements of the form {movie:[track1,track2,...]}
        for mov,tracks in keep.items():
            #turn on the desired tracks
            for itracks in tracks:
                sampTrStatus[mov][itracks]=1
                log(f"Manually including track {itracks} from movie {mov}")

        #remove movies
        for mov in removemov:
            for itr in sampTrStatus[mov]: 
                sampTrStatus[mov][itr]=0;
            log(f"Manually removing movie {mov} from experiment")
                
        return sampTrStatus, out_tracks
