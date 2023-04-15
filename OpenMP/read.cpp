#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
int main()
{
    fstream newfile;
    vector<string> sudokulist;
    newfile.open("nsudoku.txt", ios::in); // open a file to perform read operation using file object
    // cout << "Hi" << endl;
    if (newfile.is_open())
    { // checking whether the file is open
        string tp;
        while (getline(newfile, tp))
        {                       // read data from file object and put it into string.
            sudokulist.push_back(tp);
        }
        newfile.close(); // close the file object.
    }
    for(auto x:sudokulist){
        cout << x<<endl;
    }
    char **board;
    for (int i = 0; i < 100; i++)
    {
        
    }
    
}