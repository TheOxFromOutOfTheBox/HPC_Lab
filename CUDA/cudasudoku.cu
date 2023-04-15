%%cu

#include "stdlib.h"
#include "iostream"
#include "bits/stdc++.h"

using namespace std;
#define WIDTH_9X9 9
#define SUB_WIDTH 3
//((int)sqrt(BOARD_WIDTH));
#define START_CHAR '1'

int BOARD_WIDTH = WIDTH_9X9;

__device__ bool solveBoard(char **board, int rStart, int cStart, int n);
__device__ char *getHorizontalSubArray(char **board, int ix, int n)
{
	char *subarray = new char[n];

	for (int i = 0; i < n; i++)
	{
		subarray[i] = board[ix][i];
	}

	return subarray;
}

__device__ char *getVerticalSubArray(char **board, int ix, int n)
{
	char *subarray = new char[n];
	for (int i = 0; i < n; i++)
	{
		subarray[i] = board[i][ix];
		// if (blockIdx.x == 0)
		// {
		// 	printf("%c", subarray[i]);
		// }
	}
	return subarray;
}

__device__ char *getMxMSubArray(char **board, int ix, int n)
{
	char *subarray = new char[n];

	int cOffset = SUB_WIDTH * (ix % SUB_WIDTH);
	int rOffset = SUB_WIDTH * (ix / SUB_WIDTH);

	for (int i = 0; i < n; i++)
	{
		subarray[i] = board[rOffset + (i / 3)][cOffset + (i % 3)];
	}

	return subarray;
}

__device__ bool checkSudokuSubarray(char *array, int n)
{
	int nBOARD_WIDTH = n;
	bool *temp = new bool[nBOARD_WIDTH];

	for (int i = 0; i < nBOARD_WIDTH; i++)
	{
		temp[i] = false;
	}
	bool res = true;
	
	for (int i = 0; i < nBOARD_WIDTH; i++)
	{
		if ((array[i] >= START_CHAR) && (array[i] <= (START_CHAR + nBOARD_WIDTH)))
		{
			int iPos = (array[i] - START_CHAR);
			if (false == temp[iPos])
			{
				temp[iPos] = true;
			}
			else
			{
				res = false;
			}
		}
		else if (array[i] == '.')
		{
			continue;
		}
		else
		{
			res = false;
		}
	}
	return res;
}

__device__ bool isValidSudoku(char **&board, int n)
{
	if (nullptr == board)
	{
		printf("board is null.\n");
		return false;
	}
	if (n <= 0)
	{
		printf("board.length is <= 0.\n");
		return false;
	}

	// check rows
	for (int i = 0; i < n; i++)
	{

		if (false == checkSudokuSubarray(getHorizontalSubArray(board, i, n), n))
		{
			printf("Block number %d", blockIdx.x);
			printf("Invalid Horizontal Subarray %d\n", i);
			return false;
		}
	}
	// check columns
	for (int i = 0; i < n; i++)
	{
		if (false == checkSudokuSubarray(getVerticalSubArray(board, i, n), n))
		{
			printf("Invalid vertical subarray %d\n", i);
			return false;
		}
	}
	// check 3x3
	for (int i = 0; i < n; i++)
	{
		if (false == checkSudokuSubarray(getMxMSubArray(board, i, n), n))
		{
			printf("Invalid subarray\n");
			return false;
		}
	}

	return true;
}

__device__ void printHorizontalBorder(char **board, int n)
{
	for (int c = 0; c < n; c++)
	{
		if (0 == (c % SUB_WIDTH))
		{
			printf("-");
		}
		printf("--");
	}
	printf("-");
}

__device__ void printBoard(char **board, int n)
{
	if (nullptr == board)
		return;
	printf("\n");
	for (int r = 0; r < n; r++)
	{
		if (0 == (r % SUB_WIDTH))
			printHorizontalBorder(board, n);
		for (int c = 0; c < n; c++)
		{
			if (0 == (c % SUB_WIDTH))
			{
				printf("|");
			}
			printf("%c ", board[r][c]);
		}
		printf("|");
	}
	printHorizontalBorder(board, n);
}

__device__ bool canPutChar(char **board, int r, int c, char digit, int n)
{
	if ((r >= 0) && (r < n))
	{
		if ((c >= 0) && (c < n))
		{
			if ('.' == board[r][c])
			{
				printf("BEFORE BOOLS : %c ",digit);
				board[r][c] = digit;
				bool a = checkSudokuSubarray(getHorizontalSubArray(board, r, n), n);
				bool b = checkSudokuSubarray(getVerticalSubArray(board, c, n), n);
				bool c =  checkSudokuSubarray(getMxMSubArray(board, SUB_WIDTH * (r / SUB_WIDTH) + c / SUB_WIDTH, n), n);
				bool d =  solveBoard(board, r, c + 1, n);
				printf("d working\n");

				///

				printf("a:%d | b:%d | c:%d | d:%d",a,b,c,d);

				if (a &&b &&c &&d)
				{
				printf("IN CANPUTCHARRR!");
					return true;
				}
				else
				{
					board[r][c] = '.';
					return false;
				}
			}
			else
			{
				// already contains a potentially valid digit
				return true;
			}
		}
	}
	return true;
}

__device__ bool isBoardSolved(char **board, int n)
{
	bool isSolved = true;
	if(blockIdx.x==0){
		printf("STARTT\n");
		for(int i=0;i<9;i++){
			for(int j=0;j<9;j++){
				printf("%c ",board[i][j]);
			}
			printf("\n");
		}
		printf("\nENDDD");
	}

	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < n; c++)
		{
			if ('.' == board[r][c])
			{
				isSolved = false;
				// printf("NOT COMPLETELY SOLVED!!");
			}
			// return false;
		}
	}
	return isSolved && isValidSudoku(board, n);
}

__device__ bool solveBoard(char **board, int rStart, int cStart, int n)
{	
	// if(blockIdx.x==0){

	// 	printf("rstart:%d | cstart:%d\n",rStart,cStart);
	// }
	if (cStart >= n)
	{
		printf("INSIDE CSTART>=N\n");
		// roll over to the next row
		cStart = 0;
		rStart++;
	}

	// printf ("\nSolved :%d", ((rStart * BOARD_WIDTH + cStart) * 100) / (BOARD_WIDTH * BOARD_WIDTH));
	// printBoard(board, n);
	
	bool bPutChar = false;
	int ctr=0;
	for (int r = rStart; r < n; r++,cStart = 0)
	{
		for (int c = cStart; c < n; c++)
		{
			for (int i = 0; i < n; i++)
			{
				// bPutChar = i%2;
				if(blockIdx.x==0){

					printf("rstart:%d | cstart:%d\n",rStart,cStart);
				}
				bPutChar = canPutChar(board, r, c, (char)(START_CHAR + i), n);
				ctr++;
				// printf("\n\n\nhhh11 %d hhhh11\n\n\n",ctr);

				// printf("%s\n", bPutChar ? "true" : "false");
				if (bPutChar)
				{	
					// printf("TRUEEEE!!!!!\n");
					// return isBoardSolved(board, n);
					break; // potentially solved !
				}

			}
			if (!bPutChar)
				return false; // exhausted all possibilities
		}
		 // for next cycle cStart starts from zero.
		//  printf("\n\n\nhhh %d hhhh\n\n\n",ctr);
	}
	printf("GOING IN!\n");
	return isBoardSolved(board, n);
}

__global__ void bigloop(char *da)
{
	int sudoku_num = threadIdx.x + (blockIdx.x * blockDim.x);
	// printf("%d ",sudoku_num);
	// printf("%c",da[blockIdx.x*82]);
	char **board = new char *[WIDTH_9X9];
	for (int i = 0; i < WIDTH_9X9; i++)
	{
		// Declare a memory block of size n
		board[i] = new char[WIDTH_9X9];
		for (int j = 0; j < WIDTH_9X9; j++)
		{
			board[i][j] = da[sudoku_num * 82 + i * WIDTH_9X9 + j];
			if(board[i][j]=='0'){
				board[i][j]='.';
			}
		}
	}
	int n = 9;

	if (isValidSudoku(board, n))
	{
		// printf("isValidSudoku() before solving returned true.\n");
		solveBoard(board, 0, 0, n);
		// printf("\nSolved board:");
		// printBoard(board, n);
		if (isValidSudoku(board, n))
		{
			// printf("isValidSudoku() after solving returned true.\n");
			if (isBoardSolved(board, n))
			{
				printf("isBoardSolved() after solving returned true.\n");
			}
			else
			{
				// printf("isBoardSolved() after solving returned false.\n");
			}
		}
		else
		{
			printf("isValidSudoku() after solving returned false.\n");
		}
	}
	else
	{
		// printf("isValidSudoku() before solving returned false.\n");
	}
	// }
}
int main()
{
	char a[10][82] = {"301086504046521070500000001400800002080347900009050038004090200008734090007208103", "048301560360008090910670003020000935509010200670020010004002107090100008150834029", "070000043040009610800634900094052000358460020000800530080070091902100005007040802", "008317000004205109000040070327160904901450000045700800030001060872604000416070080", "040890630000136820800740519000467052450020700267010000520003400010280970004050063", "561092730020780090900005046600000427010070003073000819035900670700103080000000050", "310450900072986143906010508639178020150090806004003700005731009701829350000645010", "800134902041096080005070010008605000406310009023040860500709000010080040000401006", "165293004000001632023060090009175000500900018002030049098000006000000950000429381", "000003610000015007000008090086000700030800100500120309005060904060900530403701008"};

	char b[82 * 10];
	int k = 0;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 82; j++, k++)
		{
			b[k] = a[i][j];
		}
	}

	char **da;
	char *db;
	// same as size of sudoku array
	int size = 82 * 10 * sizeof(char);

	// trying to convert to 1D array
	cudaMalloc((void ***)&da, size);
	cudaMalloc((void **)&db, size);

	// copy 1D array to device mem
	// cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
	// printf SUB_WIDTH ;

	bigloop<<<10, 1>>>(db);

	cudaFree(db);

	return 0;
}