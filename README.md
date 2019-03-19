# Fasτ: Accelerated Data Driven Microstructural Design of Porous Electrodes for Battery Applications

<!--
<p align="middle">
<img src="https://github.com/krsvojte/fast/blob/master/images/solver.gif" alt="Solver Tortuosity"/>
</p>
-->

<p align="left">
<img src="https://github.com/krsvojte/fast/blob/master/images/packed_ellipsoids_01.gif" alt="Ellipsoids r=1/1" width="295"/>
<img src="https://github.com/krsvojte/fast/blob/master/images/packed_ellipsoids_035.gif" alt="Ellipsoids r=0.35" width="295"/>
<img src="https://github.com/krsvojte/fast/blob/master/images/packed_ellipsoids_10.gif" alt="Ellipsoids r=1/10" width="295"/>

</p>



## Libraries required:
- [GLEW](http://glew.sourceforge.net/) *>= 2.0.0*
- [EIGEN](http://eigen.tuxfamily.org/index.php?title=Main_Page) *>= 3.3.4*
- [CUDA](https://developer.nvidia.com/cuda-downloads) (tested on 8.0+, 9.2 or newer recommended)
- Contains submodules for further libraries. Run ```git submodule init``` and ```git submodule update```

### Compiler requirements
- C++14 compatible 
- Tested on MSVC 2017, Clang 6+, GCC 5.4.0+
- NVCC (CUDA Toolkit 8.0+)

## Build:
- CMake 3.2+ required

```
git clone https://github.com/krsvojte/fast.git
cd fast
git submodule init
git submodule update
mkdir build
cd build
export CUDA_PATH=(PATH TO CUDA)
cmake .. 
make
make install
cd ..
```

- ```make install``` installs to default CMake specified directory (```/usr/local``` or ```C:/Program Files/```), to change that use ```make install DESTDIR=/your/path```

## Usage:
```
./fast [action] [parameters]
```

### Tortuosity calculation (tau)
```
./fast tau [input file or folder] -o output.csv -dx
```
Calculates torutosity in x-direction and outputs to a comma separated file.
#### Flags
- ```-d[DIR]``` sets direction of tortuosity calculation 
  - DIR can be ```x,y,z,all,pos,neg``` for x, y, z, all directions, positive and negative directions
- ```--solver [SOLVER]``` chooses the solver ```CG``` or ```BICGSTAB```
- ```-v``` verbose output
- ```--tol [x]``` sets tolerance to 1e-x (default 6) 
- ```--iter [x]``` sets maximum iterations (default 10000)
- ```--cpu``` and ```--cputhreads [x]``` uses CPU version of the solver with x threads
- ```--over``` allows GPU memory oversubscription (only works under unix systems), enable whenever the solver will not fit into GPU memory

### Reactive area density calculation (alpha)
```
./fast alpha [input file or folder] -o output.csv -dx 
```
Calculates reactive area density in x-direction and outputs to a comma separated file.
#### Flags
- ```-d[DIR]``` sets direction 
  - DIR can be ```x,y,z,all,pos,neg``` for x, y, z, all directions, positive and negative directions
- ```--basic``` includes unreachable areas

### Volume manipulation
Can be used wtih both ```tau``` and ```alpha```

- ```--sub [X]``` ```--origin [x0]``` crops subvolume of size ```X^3``` at position ```(x0,x0,x0)```
- ```--sub [X,Y,Z]``` ```--origin [x0,y0,z0]``` crops subvolume of size ```(X,Y,Z)``` at position ```(x0,y0,z0)```

### Acknowledgements
Vojtech Krs and Bedrich Benes would  like  to  thank  NVidia  for  the graphics hardware provided.  This research was funded in part by National Science Foundation grant #1608762, *Inverse Procedural Material Modeling for Battery Design*. Abhas Deva and R. Edwin García thank the D3Batt Center at the Toyota Research Institute for the financial support.





