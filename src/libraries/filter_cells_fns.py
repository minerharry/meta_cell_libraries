from enum import Enum, EnumMeta, IntEnum
import functools
from typing import Collection, Iterable, Literal, Sized, Tuple, Union
from typing import overload
import numpy as np
from skimage.io import imread
import os
from scipy import ndimage
from skimage import measure
import pandas as pd
from libraries.centers import get_centers
from tqdm import tqdm
from typing import Any


class CellFiltersMeta(EnumMeta):
    def __call__(cls,value,*args,**kwargs):
        f = lambda c: super(CellFiltersMeta,cls).__call__(c,*args,**kwargs)
        if not isinstance(value,Iterable):
            return f(value)
        return np.vectorize(f,otypes=(object,))(value)

DEFAULT_BITS = 16
invalid = 0b1000 #dum

class CellFilters(IntEnum,metaclass=CellFiltersMeta): ##4-bit code: first bit is validity (0 means no error in filtering) and other three bits are error code
    touching_edge = invalid | 0b000
    too_small = invalid | 0b001
    too_large = invalid | 0b010
    two_nuclei = invalid | 0b1011

    valid = 0b0000
    invalid = invalid

    ##edge case: a pixel of zero and a pixel with a valid label theoretically have the same code. to fix this, we define the code of empty pixels to be -1
    ##this specifically and only affects get_code, and the bit representation remains unaffected. empty is never used in encoding, only decoding.
    ##this has a few ramifications: primarily, get_code inputs unsigned values and returns signed ones, which affects the dtype of ndarrays
    ##this introduces some overhead, so get_code also has a kwarg to disable it and return empty pixels as valid
    empty = -1 
    
    #shifts the 4 bit code to the beginning of a number with a given number of bits. 
    def in_place(self,bits:int=DEFAULT_BITS):
        return self.value << bits-4
    
    ##return the max valid label for a given number of bits, due to the 4 leading code bits
    @classmethod
    def max_label(cls,bits:int=DEFAULT_BITS):
        return (1 << (bits-4)) - 1

    ##return whether the given number represents a valid (first bit zero) or invalid (first bit 1) 
    @classmethod
    def is_valid(cls,num:Union[int,np.ndarray],bits:int=DEFAULT_BITS):
        return ~cls.is_invalid(num,bits=bits)
    @classmethod
    def is_invalid(cls,num:Union[int,np.ndarray],bits:int=DEFAULT_BITS):
        return (np.array(num) & cls.invalid.in_place(bits)).astype(bool)
    
    @classmethod
    def valid_only(cls,num:Union[int,np.ndarray],bits:int=DEFAULT_BITS,mark_removed=0):
        return np.where(cls.is_valid(num),num,mark_removed)
    
    @classmethod
    def invalid_only(cls,num:Union[int,np.ndarray],bits:int=DEFAULT_BITS,mark_removed=0):
        return np.where(cls.is_invalid(num),num,mark_removed)
    
    @classmethod
    def get_code(cls,num:Union[int,np.ndarray],bits:int=DEFAULT_BITS,valid_empty=False):
        assert np.all(num >= 0)
        n = (num >> bits-4)
        if valid_empty:
            return n
        if isinstance(n,np.ndarray):
            n = n.astype(np.int8);
            n[num==0] = -1
        elif num == 0:
            n = -1
        return n

    @classmethod
    def get_label(cls,num:Union[int,np.ndarray],bits:int=DEFAULT_BITS):
        return num & ((1 << (bits-4)) - 1) #get lower n-4 bits
    


@overload
def getcells(filecell:Union[Union[str, bytes, os.PathLike],np.ndarray],filenuc:Union[Union[str, bytes, os.PathLike],np.ndarray],parameters:dict[str,Any],return_metrics:Literal[False]=...,centertype="approximate-medoid",all_centers=False,
    )->tuple[np.ndarray,np.ndarray]: ...

@overload
def getcells(filecell:Union[Union[str, bytes, os.PathLike],np.ndarray],filenuc:Union[Union[str, bytes, os.PathLike],np.ndarray],parameters:dict[str,Any],return_metrics:Literal[True]=...,centertype="approximate-medoid",all_centers=False,
    )->tuple[pd.DataFrame,np.ndarray,np.ndarray]: ...

def getcells(filecell:Union[Union[str, bytes, os.PathLike],np.ndarray],filenuc:Union[Union[str, bytes, os.PathLike],np.ndarray],parameters:dict[str,Any],return_metrics=False,centertype="approximate-medoid",all_centers=False):

    #membrane
    maskMem:np.ndarray=imread(filecell) if not isinstance(filecell,np.ndarray) else filecell;
    maskMem[maskMem>0]=1
    maskMem = np.array(ndimage.binary_fill_holes(maskMem));

    #nuclei
    maskNuc:np.ndarray = imread(filenuc) if not isinstance(filenuc,np.ndarray) else filenuc;
    maskNuc[maskNuc>0]=1
    maskNuc = np.array(ndimage.binary_fill_holes(maskNuc));

    #label different objectes in masks
    maskMem,numMem = measure.label(maskMem,return_num=True)
    maskNuc,numNuc = measure.label(maskNuc,return_num=True);


    if numMem <= CellFilters.max_label(16) and numNuc < CellFilters.max_label(16):
        nbits = 16
        maskMem = maskMem.astype('uint16');
        maskNuc = maskNuc.astype('uint16');
    else:
        raise NotImplementedError("Too many objects in frame, not storable in a 16 bit integer with 4bit filter code")

  
    #FILTERS
    ## since the functions replace the original label with the code, or the original and the result to add codes to existing labels.
    ## Once invalid, the label will never be passed to future filters, so codes will never get or'd together. 
    ## THE ABOVE RESTRICTION IS VERY IMPORTANT, if a filter ever needs to use invalid labels as well as valid ones, care should be taken
    ## that codes either never get or'd together or that they are independent and produce the correct code when or'd
    if parameters['remove_cells_touching_edge'] == True:
        maskMem |= remove_touching_edge(CellFilters.valid_only(maskMem,nbits),mark_removed=CellFilters.touching_edge.in_place(nbits))
  
    if parameters['filter_cell_size'] == True:
        maskMem |= remove_extreme_objects(CellFilters.valid_only(maskMem,nbits), parameters['minareacell'], parameters['maxareacell'], mark_removed = (CellFilters.too_small.in_place(nbits),CellFilters.too_large.in_place(nbits)))
    
    if parameters['filter_nuc_size'] == True:
        maskNuc |= remove_small_objects(CellFilters.valid_only(maskNuc,nbits), parameters['minareanuc'], mark_removed=CellFilters.too_large.in_place(nbits))
  
    if parameters['remove_multi_nuclei_cells'] == True:
        maskMem |= remove_multiple_nuclei_cells(CellFilters.valid_only(maskMem,nbits),CellFilters.valid_only(maskNuc,nbits),mark_removed=CellFilters.two_nuclei.in_place(nbits))

    if (return_metrics):
        #if there are cells get metrics
        #(make sure not to count zero)
        ids=np.unique(maskMem[maskMem.nonzero()])
        # print(ids)
        if len(ids) > 0:

            #NOTE: This does all of the analysis and measurements for **all labels**, not just the valid ones. this is so that invalid label data shows up for later stages of analysis
            cellsmetrics = measure.regionprops_table(maskMem, properties=('label','area'))
            cellsmetrics=pd.DataFrame(cellsmetrics)
            if (len(cellsmetrics['label']) > 0 and len(cellsmetrics['area']) > 0):
                #GET CENTERS
                #get labels
                labels=cellsmetrics['label']    
                #Because 'label' was copied from the table, after computing the centers 
                #and concatenating them to the table they should be in the right order
                metrics = [cellsmetrics]
                for ctype in (get_centers.valid_centers if all_centers else [centertype]): 
                    ## to save on import complexity, I (H) just set get_centers.valid_centers to the list of valid center inputs. 
                    ## get_centers is still a function, it just has the attribute now. source is libraries.centers. deal with it.
                    centers = get_centers(maskMem,ctype,labels,False)
                    #add centers to cell properties
                    metrics.append(pd.DataFrame(data=np.asarray(centers),columns=[ctype+'x',ctype+'y']))

                #get filter codes and names
                codes = CellFilters.get_code(np.array(labels))
                metrics.append(pd.DataFrame(data=[(c,CellFilters(c).name) for c in codes],columns=["filter code","filter code name"]))

                #get cellsmetrics
                cellsmetrics = pd.concat(metrics,axis=1)
        else:
            cellsmetrics=pd.DataFrame();
        return cellsmetrics, maskMem, maskNuc 
    else:
        return maskMem,maskNuc
 

def remove_multiple_nuclei_cells(labeledcellsmask:np.ndarray,labelednucsmask:np.ndarray,mark_removed:int=0)->np.ndarray:
    out = np.copy(labeledcellsmask)
    labels=set(out.ravel())
    for i in labels:
        if i > 0: #ignore background           
            #get obect i coordinates 
            icoord=np.where(out==i)
            #get values of nuclear mask for those coordinates
            nucvalues=labelednucsmask[icoord]
            #get labels list (get individual elements)
            nuclabels=np.unique(nucvalues)
            #if there is more than one nuclei (2 objects counting backround)
            if len(nuclabels) > 2: 
                #remove cell object:
                out[out==i]=mark_removed
    return out


#essentially identical procedure to morphology's version, but with less input validation and custom mark_removed value
def remove_large_objects(labeledmask:np.ndarray,maxarea:float,mark_removed:int=0)->np.ndarray:
    out = np.copy(labeledmask)
    pos = np.copy(out)
    pos[pos<0] = 0
    component_sizes = np.bincount(pos.ravel()) #label-indexed array of sizes (starting from zero)
    too_large = component_sizes > maxarea #label-indexed bool array
    too_large[0] = False #don't affect the zero object
    too_large_mask = too_large[pos] #use labels as indexes into the array
    out[too_large_mask] = mark_removed #mask: array with each label replaced by its corresponding element in too_large
    return out

def remove_small_objects(labeledmask:np.ndarray,minarea:float,mark_removed:int=0)->np.ndarray:
    out = np.copy(labeledmask)
    pos = np.copy(out)
    pos[pos<0] = 0
    component_sizes = np.bincount(pos.ravel()) #label-indexed array of sizes (starting from zero)
    too_small = component_sizes < minarea #label-indexed bool array
    too_small[0] = False #don't affect the zero object
    too_small_mask = too_small[pos] #mask: array with each label replaced by its corresponding element in too_small
    out[too_small_mask] = mark_removed
    return out

def remove_extreme_objects(labeledmask:np.ndarray,minarea:float,maxarea:float,mark_removed:int|tuple[int,int]=0):
    if isinstance(mark_removed,int):
        mark_removed = (mark_removed,mark_removed)
    out = np.copy(labeledmask)
    pos = np.copy(out)
    pos[pos<0] = 0
    component_sizes = np.bincount(pos.ravel())

    too_large = component_sizes > maxarea
    too_large[0] = False
    too_large_mask = too_large[pos]

    too_small = component_sizes < minarea
    too_small[0] = False
    too_small_mask = too_small[pos]

    out[too_small_mask] = mark_removed[0]
    out[too_large_mask] = mark_removed[1]
    return out


def remove_touching_edge(labeledmask:np.ndarray,margins:Union[int,Tuple[int,int],Tuple[int,int,int,int]]=1,mark_removed:int=0)->np.ndarray:
    '''remove regions that touch the edge of the image.

    margins param can be:
    - int - will remove regions that have pixels within distance margins from each edge; will not remove if margins=0
    - tuple[int,int] - filter regions within margins[0] of the first axis, within margins[1] of the second axis
    - tuple[int,int,int,int] - filter regions by: [-first axis, +first axis, -second axis, +second axis]
    '''
    if isinstance(margins,Collection) and len(margins) == 2:
        margins = (margins[0],margins[0],margins[1],margins[1]);
    elif isinstance(margins,int):
        margins = (margins,margins,margins,margins);
    

    out = np.copy(labeledmask)

    #scan each edge of the image
    first=out.shape[0]
    second=out.shape[1]

    def rm_idx(a,b):
        if out[a,b] > 0:
            out[out==out[a,b]] = mark_removed;
    
    #i = first axis, j = second axis

    #negative first axis
    for i in range(margins[0]): 
        for j in range(second):
            rm_idx(i,j);            
    
    #positive first axis
    for i in range(first-margins[1],first):
        for j in range(second):
            rm_idx(i,j);

        
    #negative second axis
    for i in range(first):
        for j in range(margins[2]):
            rm_idx(i,j);
    
    
    #positive second axis
    for i in range(first):
        for j in range(second-margins[3],second):
            rm_idx(i,j);
    
    return out


if __name__ == "__main__":
    c = np.array([CellFilters.valid.value,CellFilters.too_large.value])
    print(CellFilters(c))