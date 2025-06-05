from collections import UserDict
import itertools
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple, TypeVar, Union
from libraries.parse_moviefolder import filename_regex
series_regex = "s([0-9]+)"
time_regex = "t([0-9]+)"

##Parses .nd files from metamorph on the optotaxis microscope
T = TypeVar("T")
null = object()
class NDData(dict[str,str|list[str]]):
    def get(self,val:str,default:T|None=None):
        if val in self:
            if isinstance(self[val],str):
                return self[val]
            else:
                raise ValueError(f"Duplicate entry detected for key {val}, use __getitem__ or getEntry instead")
        else:
            if default:
                return default
            else:
                raise KeyError(val)
            
    def getEntry(self,val:str,default:T=null):
        if val in self:
            return self[val]
        else:
            if default is not null:
                return default
            else:
                raise KeyError(val)

def parseND(filePath)->NDData:
    with open(filePath,'r') as f:
        lines = f.readlines();
    args = NDData();
    for line in lines:
        largs = line.rstrip("\n").split(", "); #line args lol
        if largs[0] == '':
          continue;
        if len(largs) == 1 or largs[1] == '':
            if largs[0].startswith("\"EndFile\""):
              break;
            continue;
        key = largs[0].replace("\"","")
        val = ", ".join(larg.replace("\"","") for larg in largs[1:]);
        if key in args:
            ##DUPLICATE ROW! This happens sometimes. result is a list of str for each instance
            if isinstance(args[key],list):
                args[key].append(val)
            else:
                args[key] = [args[key],val]
        else:
            args[key] = val
    return args;

def StageDict(filePath):
    result:Dict[int,str] = {}
    data = parseND(filePath) if not isinstance(filePath,NDData) else filePath
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
    grouped = itertools.groupby([(k,invmap[k]) for k in order],key=lambda t: re.split("\\d+",t[0])[0])
    groups:Dict[str,List[Tuple[str,int]]] = {}
    for k1,k2 in grouped:
        res = groups[k1] = []
        for a1,a2 in k2:
            res.append((a1,a2))
    return groups

suffix_regex = ' \\${0}'
def strsuffix(exp:str,suffixes=[]):
    if isinstance(suffixes,str):
        suffixes = [suffixes]
    elif not isinstance(suffixes,Iterable):
        suffixes = ["\\S*"]
    for suffix in suffixes:
        r = suffix_regex.format(suffix)
        exp = re.sub(r,'',exp)
    return exp

def is_gcp_path(path:Union[str,Path]):
  if not isinstance(path,Path):
    path = Path(path);
  return path.parts[0].lower() == "gs:";

def gs_str(p:Union[str,Path]):
    p = Path(p);
    out = ""
    if is_gcp_path(p):
        p = Path(*p.parts[1:])
        out = "gs://"
    out += p.as_posix();
    return out

def try_fetch_nd(exp:str,as_file=False):
    print(exp)
    exp_nosuffix = strsuffix(exp)
    
    #uses gsutil to try to fetch the nd file from the cloud and return its data
    #@markdown Movie numbers are associated with gradient images by their stage position name; stage positions can be extracted from a p.nd file from metamorph. Supported formatting: {experiment}
    nd_location = (f"gs://optotaxisbucket/movies/{exp_nosuffix}/{exp_nosuffix}/p.nd",f"gs://optotaxisbucket/movies/{exp_nosuffix}/Phase/p.nd")

    nd_local = Path("gcp_transfer_folder")/"nd"/exp/("p.nd");

    if not nd_local.exists():
        from gsutilwrap import copy,stat
        print(nd_location)
        for nd_loc in nd_location:
            try:
                s = stat(nd_loc);
            except Exception as e:
                import traceback as tb
                print("\n".join(tb.format_exception(e)))
                raise e
            # print(s)
            if stat is not None:
                copy(nd_loc,nd_local)
                break;
        else:
            raise FileNotFoundError(f"Cannot find .nd file for experiment {exp} [stripped: {exp_nosuffix}]")
    
    if not as_file:
        return parseND(nd_local)
    else:
        return nd_local
