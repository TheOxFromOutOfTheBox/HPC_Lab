import os,sys

filename="sudokupll.cpp"
# count_threads =[1,2,4,6,8,10,12,14,16,18,20,24,32,40,48,64,80,96,128]
count_threads=[4]

# sys.stdout=open("results.txt","a")
for each in count_threads:
    os.system(f"g++ -fopenmp {filename} && export OMP_NUM_THREADS={each} && ./a.out >> results.txt")
# sys.stdout.close()
# os.system("echo Hello World")