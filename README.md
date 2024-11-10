# Heterogeneous Programming - Deliverable 1 - Matrix Scaling on GPU  
This code compared the performance of matrix scaling on CPU and GPU. It is intended to run on Finis Terrae 3 supercomputer using 1 A100 Nvidia GPU.  
  
For this deliverable, matrix scaling is performed iteratively multiplying (scaling) a matrix's elements with a factor between 0.0 and 1.0 .  

This is the first graded deliverable for the Heterogeneous Programming subject of the HPC Master at UDC.  
  
## Compiling  
For compilation, it is enough to execute the command:  
`nvcc -o mat MatScaling.cu -lm -Xcompiler -Wall -Xcompiler -Wextra`   

## Executing  
For executing, it can be done directly on an interactive GPU node (for testing) or using the slurm bash submission script included in the repository:      
`sbatch submit_script.sh`  

The program takes up to 4 arguments, which are in order `m n r threads_blk`:  
- m represents the number of rows of the matrix  
- n represents the number of columns of the matrix  
- r represents the number of scaling factors used  
- threads_blk represents the number of threads per block when calling the CUDA kernell. Only accepted values are {32,64,128}  

By default, `m=4000`, `n=4000`, `r=5` and `threads_blk=32`.  
m=4k, blk=32 | m=10k, blk=32 | m=20k, blk=32  

## Execution results  
These execution results only account for computation time only, without memory management time:  

| r = 5, t(ms) | m=4k, n=4k | m=10k, n=10k | m=20k, n=20k |
| ------------ | ---------- | ------------ | ------------ |
| sequential   | 242.869    | 1515.802     | 6062.012     |
| blk = 32     | 0.584      | 4.362        | 17.375       |
| blk = 64     | 0.246      | 2.190        | 8.694        |
| blk = 128    | 0.143      | 1.401        | 5.530        |

| r = 10, t(ms) | m=4k, n=4k | m=10k, n=10k | m=20k, n=20k |
| ------------- | ---------- | ------------ | ------------ |
| sequential    | 474.167    | 2964.916     | 11840.800    |
| blk = 32      | 0.389      | 4.365        | 17.380       |
| blk = 64      | 0.276      | 2.413        | 9.600        |
| blk = 128     | 0.236      | 2.320        | 9.215        | 

These execution results account for computation time AND memory management time:  

| r = 5, t(ms) | m=4k, n=4k | m=10k, n=10k | m=20k, n=20k |
| ------------ | ---------- | ------------ | ------------ |
| sequential   | 242.869    | 1515.802     | 6062.012     |
| blk = 32     | 7.354      | 43.495       | 171.911      |
| blk = 64     | 7.033      | 41.766       | 164.641      |
| blk = 128    | 6.876      | 40.552       | 160.230      | 

| r = 10, t(ms) | m=4k, n=4k | m=10k, n=10k | m=20k, n=20k |
| ------------- | ---------- | ------------ | ------------ |
| sequential    | 474.167    | 2964.916     | 11840.800    |
| blk = 32      | 7.183      | 43.882       | 173.476      |
| blk = 64      | 7.033      | 41.587       | 164.273      |
| blk = 128     | 7.020      | 41.851       | 164.865      | 


## Examples
When running the program with default settings, the following output is shown:  

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