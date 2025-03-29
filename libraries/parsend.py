import itertools
import re
from typing import Dict, List, Tuple
from libraries.parse_moviefolder import filename_regex
series_regex = "s([0-9]+)"
time_regex = "t([0-9]+)"

##Parses .nd files from metamorph on the optotaxis microscope

def parseND(filePath)->Dict[str,str]:
    with open(filePath,'r') as f:
        lines = f.readlines();
    args = {};
    for line in lines:
        largs = line.rstrip("\n").split(", "); #line args lol
        if largs[0] == '':
          continue;
        if len(largs) == 1 or largs[1] == '':
            if largs[0].startswith("\"EndFile\""):
              break;
            continue;
        args[largs[0].replace("\"","")] = largs[1].replace("\"","");
    return args;

def StageDict(filePath):
    result:Dict[int,str] = {}
    data = parseND(filePath);
    for i in itertools.count(1):
        try:
            result[i] = (data[f"Stage{i}"])
        except KeyError:
            # print(f"Stage{i}")
            break;
    return result;
    
    

def sorted_dir(paths:List[str]):
    def get_key(s:str):
        out = [];
        series = re.findall(series_regex,s);
        if series: 
            out.append(int(series[0]));
        else:
            print(s);
        time = re.findall(time_regex,s);
        if time:
            out.append(int(time[0]));
        else:
            print(s);
        return out;
    try:
        paths = filter(lambda s: s.endswith(".TIF"),paths);
        paths = sorted(paths,key=get_key);
    except Exception as e:
        print(e);
        print("hello my darling")
    return paths;

def stage_from_name(name:str):
    m = re.match(filename_regex,name);
    return m.group(2) if m else "-1";

def grouped_dir(paths:List[str]):
    out = [];
    for k,g in itertools.groupby(paths,stage_from_name):
        g = list(g)
        # print(g)
        if k == "-1": continue;
        out.append(sorted_dir(g));
    return out;


##takes a stage dict and groups stages with the same nonnumeric prefix together (good for the metamorph stage position naming scheme)
def group_stage_basenames(stage_dict:Dict[int,str]):
    invmap = {v:k for k,v in stage_dict.items()};
    order = sorted(invmap.keys());
    print(order)
    grouped = itertools.groupby([(k,invmap[k]) for k in order],key=lambda t: re.split("\\d",t[0])[0])
    groups:Dict[str,List[Tuple[str,int]]] = {}
    for k1,k2 in grouped:
        res = groups[k1] = []
        for a1,a2 in k2:
            res.append((a1,a2))
    return groups


#groupby: itertools function that splits a list into sublists based on the value of a key function

