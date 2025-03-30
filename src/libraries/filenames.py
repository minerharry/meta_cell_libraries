##default filename formats and regexes. metamorph style naming scheme.
##these regexes are constructed such that filename_format.format(*re.match(filename_regex,filename).groups()) always returns filename (if filename is valid)

#basic filename format. designed to be used with .format(basename,movie,frame,ext) where ext includes the .
import re
from typing import Iterable, NamedTuple, Union


filename_format = r"(?P<basename>{})_s(?P<stage>{})_t(?P<time>{})(?P<ext>{})$" #dollar sign signifies end of string, ensures that the full ext is captured or discarded

#filename regex pieces. mixing and matching these into the regex gives lots of options for file matching specificity
numeric_basename = r'p\d*'
numeric_match = r'\d+'
alphanumeric_match = r"[^_]+"
tiff_ext = r'\.(?:TIFF|tiff|TIF|tif)' 
png_ext = r'\.(?:png|PNG)'
tiff_png_ext = r"\.(?:TIFF|tiff|TIF|tif|PNG|png)"
any_ext = r'.+'

#each filename regex piece can be slotted into the filename format directly, as shown here

#default filename regex
filename_regex = filename_format.format(numeric_basename,numeric_match,numeric_match,tiff_ext);

#mixin filename regexes
filename_regex_png = filename_format.format(numeric_basename,alphanumeric_match,numeric_match,png_ext);
filename_regex_tiff_png = filename_format.format(numeric_basename,numeric_match,numeric_match,tiff_png_ext);
filename_regex_alphanumeric = filename_format.format(numeric_basename,alphanumeric_match,numeric_match,tiff_ext);
filename_regex_alphanumeric_png = filename_format.format(numeric_basename,alphanumeric_match,numeric_match,png_ext);
filename_regex_alphanumeric_tiff_png = filename_format.format(numeric_basename,alphanumeric_match,numeric_match,tiff_png_ext);

#primarily for training, where alternate prefixes are used
filename_regex_anybasename = filename_format.format(alphanumeric_match,alphanumeric_match,numeric_match,tiff_ext)

class FilenameMatch(NamedTuple):
    basename:str
    stage:str
    time:str
    ext:str

    @classmethod
    def fromMatch(cls:type["FilenameMatch"],m:re.Match|None):
        if m is None:
            return None
        if any([b not in m.groupdict() for b in cls._fields]):
            return cls(*m.groups())
        return cls(m["basename"],m["stage"],m["time"],m["ext"])
    



def parse_filename(names:Union[str,Iterable[str]],regex:Union[str,re.Pattern]=filename_regex):
    if isinstance(names,str):
        names = [names]
    return [FilenameMatch.fromMatch(re.match(regex,name)) for name in names]
    
    
