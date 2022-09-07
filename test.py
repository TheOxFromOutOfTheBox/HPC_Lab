import os,sys

filename="matmulparallel.c"
count_threads =[1,2,4,6,8,10,12,14,16,18,20,24,32,40,48,64,80,96,128]

# sys.stdout=open("results.txt","a")
for each in count_threads:
    os.system(f"gcc -fopenmp {filename} && export OMP_NUM_THREADS={each} && ./a.out >> results.txt")
# sys.stdout.close()
# os.system("echo Hello World")