# Heterogeneous Programming - Deliverable 1 - Matrix Scaling on GPU  
This code compared the performance of matrix scaling on CPU and GPU. It is intended to run on Finis Terrae 3 supercomputer using 1 A100 Nvidia GPU.  
  
For this deliverable, matrix scaling is performed iteratively multiplying (scaling) a matrix's elements with a factor between 0.0 and 1.0 .  

This is the first graded deliverable for the Heterogeneous Programming subject of the HPC Master at UDC.  
  
## Compiling  
For compilation, it is enough to execute the command:  
`nvcc -o mat MatScaling.cu -lm -Xcompiler -Wall -Xcompiler -Wextra`   

## Executing  
For executing, it can be done directly on an interactive GPU node (for testing) or using the slurm bash submission script included in the repository (pending):    
`sbatch submit_script.sh`  

The program takes up to 4 arguments, which are in order `m n r threads_blk`:  
- m represents the number of rows of the matrix  
- n represents the number of columns of the matrix  
- r represents the number of scaling factors used  
- threads_blk represents the number of threads per block when calling the CUDA kernell. Only accepted values are {32,64,128}  

By default, `m=4000`, `n=4000`, `r=5` and `threads_blk=32`.  
  
## Examples
When running the program with default settings, the following output is shown (speed-up pending):  

```
 ---Program start---

 Configuration chosen --> m: 4000, n: 4000, r: 5, threads_blk: 32
Starting sequential matrix scaling ... 
Starting parallel matrix scaling ... 
Results are OK :)
Time for sequential:                       243.376 ms
Time for parallel (computation only):      1.093 ms
Time for parallel (memory management too): 8.264 ms
```