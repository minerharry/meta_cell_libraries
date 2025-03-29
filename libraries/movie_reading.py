from __future__ import annotations
from collections.abc import Sequence
from abc import abstractmethod
import os
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Generic, Iterable, Iterator, Mapping, Protocol, Tuple, Type, Union, TypeVar, final, overload
from contextlib import AbstractContextManager, contextmanager
from matplotlib.backend_bases import key_press_handler
from tifffile import TiffFile,TiffPageSeries
from numpy import isin, ndarray
from skimage.io import imread, imsave
from tifffile.tifffile import TiffWriter

# raise NotImplementedError("not finished lmao");

MovieKey = TypeVar("MovieKey")

class Movie(Mapping[MovieKey,Sequence[ndarray]],AbstractContextManager,Generic[MovieKey]):
    @abstractmethod
    def __init__(self,movies:Sequence[MovieKey],frames:Dict[MovieKey,Sequence[int]],location:Union[Path,str],**kwargs) -> None: ...

    #methods needed to be overriden: __len__(), __getsequence__(), __iter__() [__getitem__() for slices], __enter__() and __exit__() for opening file handles and such, and other sequence methods if necessary (__iter__(),__reversed__(),index(),count(),contains())
    def __getitem__(self, __key: Union[MovieKey,slice]) -> MovieSequence[MovieKey]:
        if isinstance(__key,slice):
            raise TypeError("Slicing of movies not supported");
        return self.__getsequence__(__key);
    
    @abstractmethod
    def __getsequence__(self,key:MovieKey) -> MovieSequence[MovieKey]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[MovieKey]: ...

    def __enter__(self) -> Movie[MovieKey]:
        return self;

    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        return super().__exit__(__exc_type, __exc_value, __traceback);

    @classmethod
    @abstractmethod
    def write(cls,movies:Sequence[MovieKey],frames:Dict[MovieKey,Sequence[int]],location:Union[str,Path],**kwargs) -> AbstractContextManager[MovieWriter[MovieKey]]: ...
    
MKey = TypeVar("MKey",contravariant=True) #touch dum
class MovieWriter(Protocol[MKey]):
    def sequence(self,key:MKey)->AbstractContextManager[SequenceWriter]: ...

class SequenceWriter(Protocol):
    def write(self,image:ndarray,**kwargs)->None: ...

class MovieSequence(Sequence[ndarray],Generic[MovieKey]):
    @overload
    def __getitem__(self,frame:int) -> ndarray: ...

    @overload
    def __getitem__(self,frame:slice) -> MovieSequence[MovieKey]: ...
    
    @final
    def __getitem__(self,frame:Union[int,slice]) -> Union[ndarray,MovieSequence[MovieKey]]:
        return self.__getframe__(frame);

    @abstractmethod
    def __getframe__(self,frame:Union[int,slice]) -> Union[ndarray,MovieSequence[MovieKey]]: pass;

    @abstractmethod
    def __len__(self) -> int: pass;

    @abstractmethod
    def __iter__(self) -> Iterator[ndarray]: pass


class KeyedMovie(Movie[MovieKey]):
    def __init__(self,movies:Sequence[MovieKey]) -> None:
        self.movies = movies;

    def __len__(self) -> int:
        return len(self.movies);

    def __iter__(self) -> Iterator[MovieKey]:
        return iter(self.movies);

class FramedMovieSequence(MovieSequence[MovieKey]):
    def __init__(self,frames:Sequence[int]):
        self.frames = frames;

    def __getframe__(self, frame: Union[int, slice]) -> Union[ndarray, MovieSequence[MovieKey]]:
        if isinstance(frame,slice):
            frameStart = self.frames.index(frame.start) if frame.start is not None else None;
            frameStop = self.frames.index(frame.stop) if frame.stop is not None else None;
            frameSlice = slice(frameStart,frameStop,frame.step);
            return self.__slice__(frameSlice);
        else:
            return self.__readframe__(frame);
            
    @abstractmethod
    def __readframe__(self,frame:int) -> ndarray: ...

    @abstractmethod
    def __slice__(self,frSlice:slice) -> MovieSequence[MovieKey]: ...

    def __len__(self) -> int:
        return len(self.frames);

    def __iter__(self) -> Iterator[ndarray]:
        return (self[f] for f in self.frames);


class ImageMapMovie(KeyedMovie[MovieKey]):
    def __init__(self,movies:Sequence[MovieKey],frames:Union[Dict[MovieKey,Sequence[int]],Sequence[int]],folder:Union[str,Path],framePaths:Union[None,Dict[MovieKey,Dict[int,Union[str,Path]]]]=None) -> None:
        if framePaths is None:
            raise TypeError(f"Image Movie constructor missing required keyword argument framePaths (Dict[MovieKey,Dict[int,Union[str,Path]]])");
        super().__init__(movies);
        #Frames Paths should be relative to the folder
        self.folder = Path(folder);
        if not isinstance(frames,Dict):
            frames = {m:frames for m in movies}
        self.frames = frames;
        self.framePaths = framePaths

    def __getsequence__(self, key: MovieKey) -> ImageMapSequence[MovieKey]:
        if key not in self.movies:
            raise IndexError(f"Movie {key} not in specified movie range: {self.movies}");
        return ImageMapSequence(self.frames[key],self.framePaths[key],self.folder,key);

    @classmethod
    @contextmanager
    def write(cls,movies:Sequence[MovieKey],frames:Dict[MovieKey,Sequence[int]],location:Union[str,Path],framePaths:Dict[MovieKey,Dict[int,Union[str,Path]]]={}):
        yield ImageMapWriter[MovieKey](movies,frames,framePaths,location);
        

class ImageMapWriter(MovieWriter[MovieKey],SequenceWriter):
    def __init__(self,movies:Sequence[MovieKey],frames:Dict[MovieKey,Sequence[int]],framePaths:Dict[MovieKey,Dict[int,Union[str,Path]]],folder:Union[str,Path]):
        self.movies = movies;
        self.frames = frames;
        self.framePaths = framePaths;
        self.folder = Path(folder);
        self.currentMovie:Union[None,MovieKey] = None;
        self.frameIter = None

    @contextmanager
    def sequence(self, key: MovieKey):
        if key not in self.movies:
            raise IndexError(f"Movie {key} not in specified movie range: {self.movies}")
        self.currentMovie = key;
        self.frameIter = iter(self.frames[self.currentMovie]);
        yield self
        self.frameIter = None;
        self.currentMovie = None;

    def write(self, image: ndarray,**kwargs) -> None:
        if self.currentMovie is None or self.frameIter is None:
            raise Exception("Cannot call write without calling .sequence(movie) as a context manager")
        f = next(self.frameIter)
        path = self.folder/self.framePaths[self.currentMovie][f];
        imsave(path,image,**kwargs);
    


class ImageMapSequence(FramedMovieSequence[MovieKey]):
    def __init__(self,frames:Sequence[int],framePaths:Dict[int,Union[str,Path]],folder:Union[str,Path],movie:MovieKey) -> None:
        super().__init__(frames);
        self.frameDict = framePaths;
        self.movie = movie;
        self.folder = Path(folder);
    
    def __slice__(self, frSlice: slice) -> ImageMapSequence[MovieKey]:
        return ImageMapSequence(self.frames[frSlice],self.frameDict,self.folder,self.movie);
    
    def __readframe__(self, frame: int) -> ndarray:
        if frame not in self.frames:
            raise IndexError(f"Frame number {frame} out of specified frame range for movie {self.movie}: {self.frames}");
        
        return imread(self.folder/self.frameDict[frame])



class TiffPageSequence(FramedMovieSequence[MovieKey]):
    #wrapper of TiffPageSeries that just returns ndarrays
    def __init__(self,series:TiffPageSeries,frames:Sequence[int],movieKey:MovieKey) -> None:
        super().__init__(frames);
        self.series = series;
        self.movie = movieKey;

    def __slice__(self, frSlice: slice) -> MovieSequence[MovieKey]:
        return TiffPageSequence(self.series,self.frames[frSlice],self.movie);

    def __readframe__(self, frame: int) -> ndarray:
        if frame not in self.frames:
            raise IndexError(f"Frame number {frame} out of specified frame range for movie {self.movie}: {self.frames}");
        f = self.series[frame];
        assert f is not None, f"Error getting frame {frame} from the tif series for movie {self.movie} - series[frame] returned None"
        return f.asarray(); 

class MultiTiffMovie(KeyedMovie[MovieKey]):
    def __init__(self,movies:Sequence[MovieKey],frames:Dict[MovieKey,Sequence[int]],location:Union[str,Path],moviePaths:Union[Dict[MovieKey,Union[str,Path]],None]=None) -> None:
        if moviePaths is None:
            raise TypeError("Multi Tiff Movie constructor missing required keyword argument moviePaths");
        super().__init__(movies);
        self.location = Path(location);
        self.paths = moviePaths;
        self.frames = frames;
        self.files:Dict[MovieKey,TiffFile] = {};
    
    def __getsequence__(self, key: MovieKey) -> MovieSequence[MovieKey]:
        if key not in self.movies:
            raise IndexError(f"Movie {key} not in specified movie range: {self.movies}");
        if key not in self.files:
            self.files[key] = TiffFile(self.location/self.paths[key]);
        return TiffPageSequence(self.files[key].series[0],self.frames[key],key);
        
    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        for file in self.files.values():
            file.close();
        self.files = {};
        return super().__exit__(__exc_type, __exc_value, __traceback)

    @classmethod
    @contextmanager
    def write(cls, movies: Sequence[MovieKey], frames: Dict[MovieKey, Sequence[int]], location: Union[str, Path], moviePaths:Union[Dict[MovieKey,Union[str,Path]],None]=None):
        assert moviePaths is not None
        yield MultiTiffWriter(movies,frames,location,moviePaths);

class MultiTiffWriter(MovieWriter[MovieKey],SequenceWriter):
    def __init__(self,movies:Sequence[MovieKey],frames:Dict[MovieKey,Sequence[int]],location:Union[str,Path],moviePaths:Dict[MovieKey,Union[str,Path]]):
        self.movies = movies;
        self.frames = frames;
        self.location = Path(location);
        self.moviePaths = moviePaths
        self.writer = None;
        self.frameIter = None;

    @contextmanager
    def sequence(self, key: MovieKey):
        if key not in self.movies:
            raise IndexError(f"Movie {key} not in specified movie range: {self.movies}")
        with TiffWriter(self.location/self.moviePaths[key]) as writer:
            self.writer = writer;
            self.frameIter = iter(self.frames[key]);
            self.lastFrame = 0;
            yield self
        self.writer = None;

    def write(self, image: ndarray,**kwargs) -> None:
        if self.writer is None or self.frameIter is None:
            raise Exception("Cannot call write without calling .sequence(movie) as a context manager");
        f = next(self.frameIter)
        for _ in range(self.lastFrame,f):
            self.writer.write(None,**kwargs); #write blank frames for excluded frames
        self.writer.write(image,**kwargs);
        self.lastFrame = f;
        

class SingleTiffMovie(KeyedMovie[int]):
    """Container for reading a tiff movie"""
    def __init__(self,movies:Sequence[int],frames:Dict[int,Sequence[int]],tiffPath:Union[str,Path]):
        super().__init__(movies);
        self.frames:Dict[int,Sequence[int]] = frames
        self.path = Path(tiffPath);
        self.file = None;

    def __enter__(self) -> SingleTiffMovie:
        self.file = TiffFile(self.path);
        return self

    def __getsequence__(self, __key: int) -> MovieSequence[int]:
        if self.file is None:
            raise Exception("Single Tiff Movie must be used as a context manager, e.g. with movie: or with load_movie(args):");
        if not isinstance(__key,int):
            raise TypeError(f"Single Tiff movie takes series numbers (integers) as input, not {type(__key)}")
        if __key not in self.movies:
            raise IndexError(f"Movie {__key} not in specified movie range: {self.movies}")
        try:
            s = self.file.series[__key];
        except IndexError:
            raise IndexError(f"Movie {__key} not in tiff file series range: {range(len(self.file.series))}");
        return TiffPageSequence(s,self.frames[__key],__key);

    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        assert self.file is not None
        self.file.close();
        self.file = None;
        return super().__exit__(__exc_type, __exc_value, __traceback)

    @classmethod
    @contextmanager
    def write(cls, movies: Sequence[int], frames: Dict[int, Sequence[int]], location: Union[str, Path]):
        with SingleTiffWriter(movies, frames, location) as w:
            yield w


class SingleTiffWriter(MovieWriter[int],SequenceWriter,AbstractContextManager):
    def __init__(self, movies: Sequence[int], frames: Dict[int, Sequence[int]], location: Union[str, Path]):
        self.movies = movies;
        self.frames = frames;
        self.tiffPath = Path(location);
        self.movie = None
        self.lastMovie = 0
        self.frameIter = None

    def __enter__(self):
        self.writer = TiffWriter(self.tiffPath);
        return self

    def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
        self.writer.close();
        return super().__exit__(__exc_type, __exc_value, __traceback)
        
    @contextmanager
    def sequence(self, key: int):
        if key not in self.movies:
            raise IndexError(f"Movie {key} not in specified movies {self.movies}")
        if key <= self.lastMovie:
            raise IndexError(f"Single tiff movie must be written in ascending order; given movie {key} not larger than previously written movie {self.lastMovie}");
        self.movie = key
        for _ in range(self.lastMovie,self.movie):
            self.writer.write(None,contiguous=False); #write blank series for skipped movies
        self.frameIter = iter(self.frames[self.movie]);
        self.lastFrame = 0
        yield self
        self.lastMovie = self.movie;
        self.movie = None

    def write(self, image: ndarray, **kwargs) -> None:
        if self.writer is None or self.frameIter is None:
            raise Exception("Cannot call write without calling .sequence(movie) as a context manager");
        f = next(self.frameIter)
        for fn in range(self.lastFrame,f):
            self.writer.write(None,contiguous=(fn!=0),**kwargs); #write blank frames for excluded frames
        self.writer.write(image,contiguous=(f!=0),**kwargs);
        self.lastFrame = f;
        
        

IMAGE_MOVIE = "images"
MULTI_TIFF_MOVIE = "multitiff"
SINGLE_TIFF_MOVIE = "tiff"

movie_map:Dict[str,type[Movie]] = {IMAGE_MOVIE: ImageMapMovie, MULTI_TIFF_MOVIE: MultiTiffMovie, SINGLE_TIFF_MOVIE: SingleTiffMovie};


@contextmanager
def load_movie(reading_parameters:Dict[str,Any],location:Union[Path,str],keyword:Union[str,None]=None):
    if "movie_types" in reading_parameters and keyword is not None:
        type_info:Dict[str,Any] = reading_parameters["movie_types"][keyword];
        movieClass:Type[Movie] = movie_map[type_info.get("movie_type",IMAGE_MOVIE)];
        movieArgs:Dict[str,Any] = type_info["movie_params"];
    else:
        movieClass = movie_map[reading_parameters.get("movie_type",IMAGE_MOVIE)];
        movieArgs = reading_parameters["movie_params"];

    with movieClass(reading_parameters["movies"],reading_parameters["frames"],**movieArgs) as m:
        yield m

# @contextmanager
# def load_movies(reading_parameters:Dict[str,Any],*movies:Tuple[Union[Path,str],Union[str,None]]):
#     for m in movies:



if __name__ == "__main__":
    movies = [0]
    frames = {0:range(0,100)};
    from filegetter import askopenfilename
    from skimage.io import imshow,show
    from skimage.exposure import rescale_intensity
    with load_movie({"movies":movies,"frames":frames,"movie_type":SINGLE_TIFF_MOVIE,"movie_params":{"tiffPath":askopenfilename()}},"images") as m:
        for frame in m[0]:
            imshow(rescale_intensity(frame));
            show()
            # input()





