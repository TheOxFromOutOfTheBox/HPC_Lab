%%cu

#include "stdlib.h"
#include "iostream"
#include "bits/stdc++.h"

using namespace std;
#define WIDTH_9X9 9

__device__ bool isValid(char **board, int row, int col, char c)
{
    for (int i = 0; i < 9; i++)
    {
        if (board[row][i] == c) // checking row if the value already there
            return false;
        if (board[i][col] == c) // checking column if value already there
            return false;
        // checking the 3X3 sub box where the blank value is present to check if element is there
        if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c)
            return false;
    }
    return true;
}

__device__ bool solve(char **board)
{
    stack<char**> stk;
    stk.push(board);
    while(!stk.empty()){
        char** newboard=stk.top();
        stk.pop();
        // Tranversing the given sudoku
        for (int i = 0; i < WIDTH_9X9; i++)
        {
            for (int j = 0; j < WIDTH_9X9; j++)
            {
                if (board[i][j] == '.')
                { // blank found

                    for (char c = '1'; c <= '9'; c++)
                    { // trying all the possible numbers from 1-9
                        if (isValid(board, i, j, c))
                        {
                            board[i][j] = c;

                            if (solve(board) == true)
                            {
                            // checking for the next blank
                                printf("HERE\n");
                                return true;
                            } 
                            else
                            {
                                board[i][j] = '.'; // if blank cannot be filled after checking then revert the changes
                            }
                        }
                    }
                    return false;
                }
            }
        }
    }
    return true; // if everything is filled then return true
}


__global__ void solveSudoku(char * db)
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
            board[i][j] = db[sudoku_num * 82 + i * WIDTH_9X9 + j];
            if (board[i][j] == '0')
            {
                board[i][j] = '.';
            }
        }
    }
    // stack<char**> stk;
    int n = 9;
    if (blockIdx.x == 0)
    {
        // stk.push(board);
        solve(stk);
        for (int i = 0; i < WIDTH_9X9; i++)
        {
            for (int j = 0; j < WIDTH_9X9; j++)
            {
                printf("%c ",board[i][j]);
            }
            printf("\n");
        }
    }
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

    char *db;
    // same as size of sudoku array
    int size = 82 * 10 * sizeof(char);

    // trying to convert to 1D array
    cudaMalloc((void **)&db, size);

    // copy 1D array to device mem
    // cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
    // printf SUB_WIDTH ;

    solveSudoku<<<10, 1>>>(db);

    cudaFree(db);

    return 0;
}