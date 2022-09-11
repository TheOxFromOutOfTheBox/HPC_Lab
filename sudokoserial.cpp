#include "stdlib.h"
#include "iostream"
#include "math.h"
#include "omp.h"
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
using namespace std::chrono;

using namespace std;

int WIDTH_9X9 = 9;
int BOARD_WIDTH = WIDTH_9X9;
int SUB_WIDTH = ((int)sqrt(BOARD_WIDTH));
char START_CHAR = '1';

bool solveBoard(char **board, int rStart, int cStart, int n);

char *getHorizontalSubArray(char **board, int ix, int n)
{
    char *subarray = new char[n];
    for (int i = 0; i < n; i++)
    {
        subarray[i] = board[ix][i];
    }

    return subarray;
}

char *getVerticalSubArray(char **board, int ix, int n)
{
    char *subarray = new char[n];

    for (int i = 0; i < n; i++)
    {
        subarray[i] = board[i][ix];
    }

    return subarray;
}

char *getMxMSubArray(char **board, int ix, int n)
{
    char *subarray = new char[n];

    int cOffset = SUB_WIDTH * (ix % SUB_WIDTH);
    int rOffset = SUB_WIDTH * (ix / SUB_WIDTH);

    int i = 0;
    for (int r = 0; r < n / SUB_WIDTH; r++)
    {
        for (int c = 0; c < n / SUB_WIDTH; c++)
        {
            subarray[i] = board[rOffset + r][cOffset + c];
            i++;
        }
    }
    return subarray;
}

bool checkSudokuSubarray(char *array, int n)
{
    int nBOARD_WIDTH = n;
    bool *temp = new bool[nBOARD_WIDTH];

    for (int i = 0; i < nBOARD_WIDTH; i++)
    {
        // cout<<array[i]<<" ";
        temp[i] = false;
    }

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
                // cout<<"Why u do this? "<<array[i]<<endl;
                return false;
            }
        }
        else if (array[i] == '.')
        {
            continue;
        }
        else
        {
            // cout<<"Wrong at "<<array[i]<<endl;
            return false;
        }
    }
    // cout<<endl;
    return true;
}

bool isValidSudoku(char **&board, int n)
{
    if (nullptr == board)
    {
        cout << "board is null." << endl;
        return false;
    }
    if (n <= 0)
    {
        cout << "board.length is <= 0." << endl;
        return false;
    }

    // check rows
    for (int i = 0; i < n; i++)
    {
        if (false == checkSudokuSubarray(getHorizontalSubArray(board, i, n), n))
        {
            cout << "Invalid Horizontal Subarray" << i << endl;
            return false;
        }
    }
    // check columns
    for (int i = 0; i < n; i++)
    {
        if (false == checkSudokuSubarray(getVerticalSubArray(board, i, n), n))
        {
            cout << "Invalid vertical subarray " << i << endl;
            return false;
        }
    }
    // check 3x3
    for (int i = 0; i < n; i++)
    {
        if (false == checkSudokuSubarray(getMxMSubArray(board, i, n), n))
        {
            cout << "Invalid subarray" << endl;
            return false;
        }
    }

    return true;
}

void printHorizontalBorder(char **board, int n)
{
    for (int c = 0; c < n; c++)
    {
        if (0 == (c % SUB_WIDTH))
        {
            cout << "-";
        }
        cout << "--";
    }
    cout << "-" << endl;
}

void printBoard(char **board, int n)
{
    if (nullptr == board)
        return;
    cout << "\n";
    for (int r = 0; r < n; r++)
    {
        if (0 == (r % SUB_WIDTH))
            printHorizontalBorder(board, n);
        for (int c = 0; c < n; c++)
        {
            if (0 == (c % SUB_WIDTH))
            {
                cout << ("|");
            }
            cout << " " << board[r][c];
        }
        cout << "|" << endl;
    }
    printHorizontalBorder(board, n);
}

bool canPutChar(char **board, int r, int c, char digit, int n)
{
    if ((r >= 0) && (r < n))
    {
        if ((c >= 0) && (c < n))
        {
            if ('.' == board[r][c])
            {
                board[r][c] = digit;
                if (checkSudokuSubarray(getHorizontalSubArray(board, r, n), n) &&
                    checkSudokuSubarray(getVerticalSubArray(board, c, n), n) &&
                    checkSudokuSubarray(getMxMSubArray(board, SUB_WIDTH * (r / SUB_WIDTH) + c / SUB_WIDTH, n), n) &&
                    solveBoard(board, r, c + 1, n))
                {
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

bool isBoardSolved(char **board, int n)
{

    for (int r = 0; r < n; r++)
    {
        for (int c = 0; c < n; c++)
        {
            if ('.' == board[r][c])
                return false;
        }
    }
    return isValidSudoku(board, n);
}

bool solveBoard(char **board, int rStart, int cStart, int n)
{
    if (cStart >= n)
    {
        // roll over to the next row
        cStart = 0;
        rStart++;
    }

    cout << "\nSolved :" << ((rStart * BOARD_WIDTH + cStart) * 100) / (BOARD_WIDTH * BOARD_WIDTH);
    printBoard(board, n);

    bool bPutChar = false;
    for (int r = rStart; r < n; r++)
    {
        for (int c = cStart; c < n; c++)
        {
            for (char i = 0; i < n; i++)
            {
                bPutChar = canPutChar(board, r, c, (char)(START_CHAR + i), n);
                if (bPutChar)
                    break; // potentially solved !
            }
            if (false == bPutChar)
                return false; // exhausted all possibilities
        }
        cStart = 0; // for next cycle cStart starts from zero.
    }
    return isBoardSolved(board, n);
}

int main()
{
    auto start = high_resolution_clock::now();
    fstream newfile;
    vector<string> sudokulist;
    newfile.open("nsudoku.txt", ios::in); // open a file to perform read operation using file object
    cout << "Hi" << endl;
    if (newfile.is_open())
    { // checking whether the file is open
        string tp;
        while (getline(newfile, tp))
        { // read data from file object and put it into string.
            sudokulist.push_back(tp);
        }
        newfile.close(); // close the file object.
    }
    for (int i = 0; i < sudokulist.size(); i++)
    {
        char **board = new char *[WIDTH_9X9];
        for (int i = 0; i < WIDTH_9X9; i++)
        {
            // Declare a memory block of size n
            board[i] = new char[WIDTH_9X9];
        }
        string testcase = sudokulist[i];

        for (int i = 0; i < WIDTH_9X9; i++)
        {
            for (int j = 0; j < WIDTH_9X9; j++)
            {
                if (testcase[(i * 9) + j] == '0')
                {
                    board[i][j] = '.';
                }
                else
                {
                    board[i][j] = testcase[(i * 9) + j];
                }
            }
        }

        cout << ("\nProblem board:");
        printBoard(board, WIDTH_9X9);
        if (isValidSudoku(board, WIDTH_9X9))
        {
            cout << ("isValidSudoku() before solving returned true.") << endl;
            solveBoard(board, 0, 0, WIDTH_9X9);
            cout << ("\nSolved board:");
            printBoard(board, WIDTH_9X9);
            if (isValidSudoku(board, WIDTH_9X9))
            {
                cout << ("isValidSudoku() after solving returned true.") << endl;
                if (isBoardSolved(board, WIDTH_9X9))
                {
                    cout << ("isBoardSolved() after solving returned true.") << endl;
                }
                else
                {
                    cout << ("isBoardSolved() after solving returned false.") << endl;
                }
            }
            else
            {
                cout << ("isValidSudoku() after solving returned false.") << endl;
            }
        }
        else
        {
            cout << ("isValidSudoku() before solving returned false.") << endl;
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // To get the value of duration use the count()
    // member function on the duration object
    cout << duration.count() << endl;
    return 0;
}

// void init_board_properties(int n)
// {
//     // cout << "init_board_properties" << n << endl;
//     BOARD_WIDTH = n;
//     SUB_WIDTH = (int)sqrt(BOARD_WIDTH);

//     if (WIDTH_9X9 == BOARD_WIDTH)
//     {
//         START_CHAR = '1';
//     }
//     else if (WIDTH_16X16 == BOARD_WIDTH)
//     {
//         START_CHAR = 'A';
//     }
//     else
//     {
//         // use defaults
//     }
// }

// char **getRandomBoard(int N)
// {
//     srand(time(0));
//     char **board = new char *[N];
//     for (int i = 0; i < N; i++)
//     {
//         // Declare a memory block of size n
//         board[i] = new char[N];
//     }
//     char *aNums = new char[N];

//     init_board_properties(N);

//     int iAttempt = 0;
//     while (!isValidSudoku(board, N))
//     {
//         iAttempt++;
//         for (int ix = 0; ix < N; ix++)
//         {
//             for (char i = 0; i < N; i++)
//             {
//                 aNums[i] = (char)(START_CHAR + i);
//             }

//             int cOffset = SUB_WIDTH * (ix % SUB_WIDTH);
//             int rOffset = SUB_WIDTH * (ix / SUB_WIDTH);

//             int i = 0;
//             for (int r = 0; r < N / SUB_WIDTH; r++)
//             {
//                 for (int c = 0; c < N / SUB_WIDTH; c++)
//                 {
//                     int iRandom = (int)(rand() % (N - i) + i);             // randomize
//                     if ((rand() % BOARD_WIDTH) >= ((BOARD_WIDTH * 7 / 9))) // sparseness
//                     {
//                         board[rOffset + r][cOffset + c] = aNums[iRandom];
//                     }
//                     else
//                     {
//                         board[rOffset + r][cOffset + c] = '.';
//                     }
//                     aNums[iRandom] = aNums[i];
//                     i++;
//                 }
//             }
//         }
//     }
//     // long stoptime = System.currentTimeMillis();
//     // System.out.format("Board generation took %d ms after %d attempts.\n", (stoptime-starttime), iAttempt);
//     return board;
// }
