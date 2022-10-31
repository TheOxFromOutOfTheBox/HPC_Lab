import pandas as pd

df=pd.read_csv("sudoku.csv") #read csv

#print(df['puzzle'][0])

with open("nsodoku.txt",encoding="utf-8",mode="w+") as f:
	for i in range(100):
		f.write(f"{df['puzzle'][i]}\n")
