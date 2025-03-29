#modified from
#https://github.com/juglab/EmbedSeg/blob/main/EmbedSeg/utils/generate_crops.py
  
from typing import Iterable, List, Tuple, Union
from warnings import warn
import cv2
import numpy as np
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial import distance_matrix
from skimage.segmentation import find_boundaries
#import hdmedians as hd
#from numba import jit

'''
@jit(nopython=True)
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float32) 
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D
'''

valid_centers = ["centroid","approximate-medoid","medoid","largest-circle","ellipse"]
def get_centers(instance:np.ndarray, center:str, ids:Iterable[int], one_hot:bool=False)->List[Tuple[float,float]]:
    """
        Get the centers of all the labeled objects in a mask
        ----------
        instance: numpy array
            `instance` image containing unique `ids` for each object (YX)
             or present in a one-hot encoded style where each object is one in it own slice and zero elsewhere.
        center: string
            One of 'centroid', 'approximate-medoid', 'medoid' or 'largest-circle'
        ids: list
            Unique ids corresponding to the objects present in the instance image.
        one_hot: boolean [Default: False]
            True (in this case, `instance` has shape DYX) or False (in this case, `instance` has shape YX).
    """
    centers:List[Tuple[float,float]] = []
    for _, id in enumerate(ids):
        if (not one_hot):
            mask = (instance == id)
        else:
            mask = (instance[id] == 1)
        y,x = np.where(mask)
        if len(y) != 0 and len(x) != 0:
            if (center == 'centroid'):
                ym, xm = np.mean(y), np.mean(x)
            elif (center == 'approximate-medoid'):
                ym,xm = get_appmedoid(x=x,y=y)
            elif (center == 'medoid'):
                ### option - 1 (scipy `distance_matrix`) (slow-ish)
                dist_matrix = distance_matrix(np.vstack((x, y)).transpose(), np.vstack((x, y)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                ym, xm = y[imin], x[imin]
                
                ### option - 2 (`hdmedoid`) (slightly faster than scipy `distance_matrix`)
                #ym, xm = hd.medoid(np.vstack((y,x))) 
                
                ### option - 3 (`numba`) 
                #dist_matrix = pairwise_python(np.vstack((x, y)).transpose())
                #imin = np.argmin(np.sum(dist_matrix, axis=0))
                #ym, xm = y[imin], x[imin]		
            elif (center=='largest-circle'):
                (ym,xm),r = getcircle(mask,x,y);
            elif (center == "ellipse"): #xm,ym backwards here is correct
                (xm,ym),size,rotation = getellipse(mask)
            else:
                raise Exception("Unrecognized center type:",center)

                
            centers.append((xm,ym))
    return centers
get_centers.valid_centers = valid_centers ##you can just do this apparently


def getellipse(mask:np.ndarray):
    contours,_ = cv2.findContours(mask.astype("uint8"),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
    if len(contours) > 1:
        warn("Multiple contours detected for id " + str(id) + ",using only the first one...")
    try:
        ellipse = cv2.fitEllipseDirect(contours[0])
    except cv2.error as e:
        if "Incorrect size of input array" in e.msg:
            ##this triggers when opencv says the contour array is too small, so it can't fit an ellipse
            #TODO: if this happens, maybe just use centroid? 
            # on the other hand, objects this small really shouldn't be going through, so it may be better to error
            raise Exception(f"object with id {id} too small to get contours for ellipse fitting")
        else:
            raise e
    return ellipse

def get_appmedoid(mask:Union[np.ndarray,None]=None,x:Union[np.ndarray,None]=None,y:Union[np.ndarray,None]=None):
    if x is None or y is None:
        assert mask is not None
        y,x = np.where(mask)
        # ,f"x: {x is None}, y: {y is None}, mask: {mask}"
    ym_temp, xm_temp = np.median(y), np.median(x)
    imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
    ym, xm = y[imin], x[imin]
    return (ym,xm)
    
def getcircle(mask:np.ndarray,)->Tuple[Tuple[float,float],float]: #center,radius
    if x is None or y is None:
        y,x = np.where(mask)
    boundary = find_boundaries(mask, mode='inner').astype(np.uint8)
    yb,xb = np.where(boundary==1)
    dist_matrix = distance_matrix(np.vstack((x, y)).transpose(), np.vstack((xb, yb)).transpose())
    mindist = np.min(dist_matrix, 1)                
    imax = np.argmax(mindist)
    ym, xm = y[imax], x[imax]
    return (ym,xm),np.max(mindist)


def generate_center_image(instance, center, ids, one_hot):
    """
        Generates a `center_image` which is one (True) for all center locations and zero (False) otherwise.
        Parameters
        ----------
        instance: numpy array
            `instance` image containing unique `ids` for each object (YX)
             or present in a one-hot encoded style where each object is one in it own slice and zero elsewhere.
        center: string
            One of 'centroid', 'approximate-medoid' or 'medoid'.
        ids: list
            Unique ids corresponding to the objects present in the instance image.
        one_hot: boolean
            True (in this case, `instance` has shape DYX) or False (in this case, `instance` has shape YX).
    """

    if (not one_hot):
        center_image = np.zeros(instance.shape, dtype=bool)
    else:
        center_image = np.zeros((instance.shape[-2], instance.shape[-1]), dtype=bool)
    centers = get_centers(instance,center,ids,one_hot=one_hot)
    for x,y in centers:
        center_image[int(y),int(x)] = True
    return center_image


def generate_annotated_image(instance:np.ndarray, center:str, ids:Iterable[int], annotation_value=100,one_hot:bool=False):
    """
        Generate a copy of the input with annotations added for centers and fit parameters
        ----------
        instance: numpy array
            `instance` image containing unique `ids` for each object (YX)
             or present in a one-hot encoded style where each object is one in it own slice and zero elsewhere.
        center: string
            One of 'centroid', 'approximate-medoid', 'medoid' or 'largest-circle'
        ids: list
            Unique ids corresponding to the objects present in the instance image.
        one_hot: boolean [Default: False]
            True (in this case, `instance` has shape DYX) or False (in this case, `instance` has shape YX).
    """
    image = instance.copy()
    for _, id in enumerate(ids):
        if (not one_hot):
            mask = (instance == id)
        else:
            mask = (instance[id] == 1)
        y,x = np.where(mask)
        if len(y) != 0 and len(x) != 0:
            if (center == 'centroid'):
                ym, xm = np.mean(y), np.mean(x)
            elif (center == 'approximate-medoid'):
                ym_temp, xm_temp = np.median(y), np.median(x)
                imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
                ym, xm = y[imin], x[imin]
            elif (center == 'medoid'):
                ### option - 1 (scipy `distance_matrix`) (slow-ish)
                dist_matrix = distance_matrix(np.vstack((x, y)).transpose(), np.vstack((x, y)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                ym, xm = y[imin], x[imin]
                
                ### option - 2 (`hdmedoid`) (slightly faster than scipy `distance_matrix`)
                #ym, xm = hd.medoid(np.vstack((y,x))) 
                
                ### option - 3 (`numba`) 
                #dist_matrix = pairwise_python(np.vstack((x, y)).transpose())
                #imin = np.argmin(np.sum(dist_matrix, axis=0))
                #ym, xm = y[imin], x[imin]		
            elif (center=='largest-circle'):
                image_only_id=np.zeros(instance.shape, dtype=bool)
                image_only_id[y,x] = True
                boundary = find_boundaries(image_only_id, mode='inner').astype(np.uint8)
                yb,xb = np.where(boundary==1)
                dist_matrix = distance_matrix(np.vstack((x, y)).transpose(), np.vstack((xb, yb)).transpose())
                mindist = np.min(dist_matrix, 1)                
                imax = np.argmax(mindist)
                ym, xm = y[imax], x[imax]

            elif (center == "ellipse"):
                contours,_ = cv2.findContours(mask.astype("uint8"),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
                if len(contours) > 1:
                    warn("Multiple contours detected for id " + str(id) + ",using only the first one...")
                try:
                    cpos,size,rotation = cv2.fitEllipseDirect(contours[0])
                except cv2.error as e:
                    if "Incorrect size of input array" in e.msg:
                        raise Exception(f"object with id {id} too small to get contours for ellipse fitting")
                    else:
                        raise e

                image = cv2.ellipse(image.astype("uint8"),(cpos,size,rotation),annotation_value,3);
                xm,ym = cpos

            else:
                raise Exception("Unrecognized center type:",center)
            image = cv2.circle(image.astype("uint8"),(int(xm),int(ym)),3,annotation_value,-10)
                
            
    return image


def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img == l
        mask_filled = binary_fill_holes(mask, **kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """
        Fill small holes in label image.
    """

    def grow(sl, interior):
        return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior))

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None: continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """
        Percentile-based image normalization.
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x
