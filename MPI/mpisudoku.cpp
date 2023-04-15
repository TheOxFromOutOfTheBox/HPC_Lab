#include "stdlib.h"
#include "iostream"
#include "math.h"
#include "mpi.h"
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#define FROM_MASTER 1
#define FROM_WORKER 2

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
    // #pragma omp parallel for

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

    for (int i = 0; i < n; i++)
    {
        subarray[i] = board[rOffset + (i / 3)][cOffset + (i % 3)];
    }

    return subarray;
}

bool checkSudokuSubarray(char *array, int n)
{
    int nBOARD_WIDTH = n;
    bool *temp = new bool[nBOARD_WIDTH];

    for (int i = 0; i < nBOARD_WIDTH; i++)
    {
        // // cout<<array[i]<<" ";
        temp[i] = false;
    }
    bool res = true;
    // #pragma omp parallel for shared(res,temp)
    // {
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
                // // cout<<"Why u do this? "<<array[i]<<endl;
                // return false;
                res = false;
            }
        }
        else if (array[i] == '.')
        {
            continue;
        }
        else
        {
            // // cout<<"Wrong at "<<array[i]<<endl;
            // return false;
            res = false;
        }
    }
    // }
    // // cout<<endl;
    return res;
}

bool isValidSudoku(char **&board, int n)
{
    if (nullptr == board)
    {
        // cout << "board is null." << endl;
        return false;
    }
    if (n <= 0)
    {
        // cout << "board.length is <= 0." << endl;
        return false;
    }

    // check rows
    for (int i = 0; i < n; i++)
    {
        if (false == checkSudokuSubarray(getHorizontalSubArray(board, i, n), n))
        {
            // cout << "Invalid Horizontal Subarray" << i << endl;
            return false;
        }
    }
    // check columns
    for (int i = 0; i < n; i++)
    {
        if (false == checkSudokuSubarray(getVerticalSubArray(board, i, n), n))
        {
            // cout << "Invalid vertical subarray " << i << endl;
            return false;
        }
    }
    // check 3x3
    for (int i = 0; i < n; i++)
    {
        if (false == checkSudokuSubarray(getMxMSubArray(board, i, n), n))
        {
            // cout << "Invalid subarray" << endl;
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
            // cout << "-";
        }
        // cout << "--";
    }
    // cout << "-" << endl;
}

void printBoard(char **board, int n)
{
    if (nullptr == board)
        return;
    // cout << "\n";
    for (int r = 0; r < n; r++)
    {
        if (0 == (r % SUB_WIDTH))
            printHorizontalBorder(board, n);
        for (int c = 0; c < n; c++)
        {
            if (0 == (c % SUB_WIDTH))
            {
                // cout << ("|");
            }
            // cout << " " << board[r][c];
        }
        // cout << "|" << endl;
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
    bool isSolved = true;
    // #pragma omp parallel for collapse(2)
    for (int r = 0; r < n; r++)
    {
        for (int c = 0; c < n; c++)
        {
            if ('.' == board[r][c])
                // #pragma omp cancel parallel
                isSolved = false;
            // return false;
        }
    }
    return isSolved && isValidSudoku(board, n);
}

bool solveBoard(char **board, int rStart, int cStart, int n)
{
    if (cStart >= n)
    {
        // roll over to the next row
        cStart = 0;
        rStart++;
    }

    // cout << "\nSolved :" << ((rStart * BOARD_WIDTH + cStart) * 100) / (BOARD_WIDTH * BOARD_WIDTH);
    // printBoard(board, n);

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

int main(int argc, char *argv[])
{
    int total_workers, src_device, dest_device, mtype;
    MPI_Status status;
    // Initialize the MPI environment
    // MPI_Init(NULL, NULL);
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    total_workers = world_size - 1;

    // MASTER
    if (world_rank == 0)
    {
        // vector<char [81]>
        double start = MPI_Wtime();
        mtype = FROM_MASTER; // send some number of sudoku from master to each of the workers.

        int numSudoku = 100;
        // char res_sudoku[numSudoku][82];
        vector<string> res_sudoku;
        // char all_sudoku[numSudoku][82] = {"301086504046521070500000001400800002080347900009050038004090200008734090007208103", "048301560360008090910670003020000935509010200670020010004002107090100008150834029", "070000043040009610800634900094052000358460020000800530080070091902100005007040802", "008317000004205109000040070327160904901450000045700800030001060872604000416070080", "040890630000136820800740519000467052450020700267010000520003400010280970004050063", "561092730020780090900005046600000427010070003073000819035900670700103080000000050", "310450900072986143906010508639178020150090806004003700005731009701829350000645010", "800134902041096080005070010008605000406310009023040860500709000010080040000401006", "165293004000001632023060090009175000500900018002030049098000006000000950000429381", "000003610000015007000008090086000700030800100500120309005060904060900530403701008"};
        char all_sudoku[numSudoku][82]={"070000043040009610800634900094052000358460020000800530080070091902100005007040802",
"301086504046521070500000001400800002080347900009050038004090200008734090007208103",
"048301560360008090910670003020000935509010200670020010004002107090100008150834029",
"008317000004205109000040070327160904901450000045700800030001060872604000416070080",
"040890630000136820800740519000467052450020700267010000520003400010280970004050063",
"561092730020780090900005046600000427010070003073000819035900670700103080000000050",
"310450900072986143906010508639178020150090806004003700005731009701829350000645010",
"800134902041096080005070010008605000406310009023040860500709000010080040000401006",
"165293004000001632023060090009175000500900018002030049098000006000000950000429381",
"000003610000015007000008090086000700030800100500120309005060904060900530403701008",
"405001068073628500009003070240790030006102005950000021507064213080217050612300007",
"960405100020060504001703006100004000490130050002007601209006038070218905600079000",
"904520070001890240002643000070960380000108700600000010090080000000750030000312569",
"001408006093520741000010520602080300007060000005039060064052109020000654500607083",
"007300054245080900003040070070960000000020760000801002008294016609108020000007003",
"005346170000000050000800009502930741070000003000700020090050632207600400600420007",
"320090400705021800001060372218037009500480700000005000670000280000873900804000107",
"000030007480960501063570820009610203350097006000005094000000005804706910001040070",
"087002010204017003006800705508001000640008100002050670439180007020900030700023091",
"040000008760020349000470500900000030000036702308947000000004010200700603690001000",
"007009050040000930059740080000016790083000002710000000830060020000395018605020070",
"620740100070100052508000370067300900090000060800970031002000006000800000450002003",
"627140503345206971089503602000700364793054018460008059056031097971005836834067105",
"720890500390460100000217890809002000204008000105049287610000028080020915950701040",
"803700000026000004097100203705000908901070040038401567170950800680210435352846000",
"206007905345092018000850060000509000708000450004083126420060580571200094860000200",
"700000000400708061100296847000001400801000030090600075080010006007052394935467010",
"900000002010060390083900100804095007130670049060041000302010050000500000541080030",
"005000060000006302040081597012038754000200810087014000120007680000092030954860200",
"709000100421000050300700008100000302908320000002070809070530400090000675000600093",
"001300002079000000020670903000967300750001049080503100040702530205806700107405060",
"600017400401003008059800721120000050000040800008020100004530007700090086263170000",
"800005047040008500000000630000000490590040002072006305980000273067804051030070000",
"400000070060850240000301065049078500007032008280009430120703004700010000006200009",
"500000260024086000807152300000600703003400006700098120030800600072000481000070000",
"604001035003450001521900000069807104250014007410090006000060010000039070070140503",
"420796050300280497879004612690005201538400009010369004983647025006150000100020346",
"080000090090502000003001408007090630000000001650020000900300080310040970002879013",
"005020040007090318106840070510000693300000700074230001050764189040001002081902030",
"000000065004056701070813940006005490500690000009042000062504179000000030000000628",
"790048060125976300040305720980657413007100856001000007006002135009701000010500000",
"290800300000000046786500200020000100100009482647000903875200004310645700009008000",
"180023000942500008060010092209840000608395040300067850806000027407002900001700004",
"908260351500094872002010409003000084154083007020000905760100040009006000001005090",
"950064037046081059001539080034106070865000020090028000500612000613470090000890005",
"070620509029400000680570000300000000806750093000086005000000170064030000005004032",
"790400801100780090000910402975821046000000785006504000207090034300200908009100627",
"600502000450093700030684090203800560001000002007025010005036400320750601976418053",
"870500060010000207640180509000001000120006075008072006000605008000004000904008621",
"032104070500300002000625000080061020007402608201780054640000030098003700723056001",
"480060001103008526065700080058200010004080300021679005006023907510946008009800000",
"030074096600398700008061340053007009400059070096002084000023600310040000862000437",
"000001000003006097061037500000709800830040001000020006740008610300060052000900003",
"001002003004000900296015048560008197009000054700950300400187600010500402900020005",
"003060005792051400006200009268719054030026978000000601050103090029004000800002003",
"049003825500709106361025007600590084800004309190370562008000691415986273970130450",
"026853700000000000053040096240076810319005460600134020064397180708502604002060079",
"468050702273140509009002463001860000090030001030000600980501200327600104016324908",
"000009038000005004350700091000400000407500020060020805500900006840001070036040009",
"604300090210079064905014270450003900000001000008000000032700080890206010506000000",
"591300200004062031000917508063005080100480003057000000000090610048270395000050870",
"309000050007003060081407200108900020700340910932068000004080090506704000003020607",
"406150083703200000508003200630700500157009024982500307049680702000900030071300009",
"100000809950018000300000076580906000003500000204700560000000081600004003830169045",
"000000000084500970291000008809100500753942060010000000002714000370080400100035700",
"243600950005201000706953802657009080804507029009030060501402700008016005092785004",
"001090000009080054080576900076925301130000007495700080608250070000000006943607020",
"030810070000067105710003084002085790807436000600792040160329000920571860573048910",
"690538072080000600030907085040000900006029051000751800904000010008610090003000740",
"100500007090830004800200100400010072000048351001302006204900715315007690000000023",
"072834069045020008000075020014080000867040003200300800523408000000003080486007310",
"907201000000000006010070250825019064749032800003508090302000501680300907500027638",
"008052070091040006705800400570083204600000017810764035156400320024030700387095641",
"037604250200300100009250008000700090902005031740130006090070604001060000526800903",
"006007208310082076700061430009018500105470060800203004000030680678095300000820750",
"000000840090207100017568903700000600040602000003000002030850271900036500001700006",
"000000048300100720840000003003002006017060200620580304000013402100600890004857001",
"760009001024810076001706902600905018005030607800060295200600700400357009530090800",
"210089357090002800004700912341506200509004001600800430070325060100007589065910020",
"580062041060901000009000600006107839720008010001000006000705000075406003648039150",
"140000270070054319500207048000068024080400107024730000007000802802003400491806000",
"010000608900000100706300000500743000000005400340091075003200091104000300069000050",
"380500091697010005005080700460200003073098140800743000140000907730904000950070030",
"090143802600590007480672593547261009368459721019837654906384215154706930030915006",
"000050780000600010090020603100400000054260090007030001582910370000002500003000129",
"040301000000427060270560048500094600904006852706005090001950780090013000325700906",
"750314600830006001010000407507208000400000059000950084300000070081000003002001800",
"000003648605700923003680015109408207000009050500070000807905000306804070250300189",
"007310902380050006025000031073009104140830007650701328034080600500207010716493285",
"108600003000080602060054819900045231050960000481700090000097065019000380605410900",
"000070060200486010080032000400800030000047002600213740831054000527000000960020570",
"370006905040300000906102004164059200800030047735248600520067010010003809083900752",
"000534007000297500020016004000008000058409070000000600500000010900641850183700946",
"709000040032640059080279106000320000290001308013000020076100200000804003500002091",
"000963000301520490060001325810290574096000130072004080930048060108659740647130058",
"050600041003045600960800300600000100007008204402000003029500036006024510000000802",
"763541289048009050009860070476003095530096002002150730005000400027005060080627510",
"600053904000600082098712305780304520109200706520176839000431258840000600000068007",
"407851030000470805001032640018723400670089020930604708349208000000300269720090384",
"000090015000100840050380207000530000000602431012009500028003004037000002400050083"};
        // int worker_load = numSudoku / world_size;
        // int extra_load = numSudoku % world_size;
        // int start_sudoku_index = (world_rank * worker_load) + extra_load; // giving extra load to master only

        // int counts[numSudoku];
        // int displs[numSudoku];

        // counts[0]=worker_load+extra_load;
        // displs[0]=0;
        // for (int i = 1; i < numSudoku; i++)
        // {
        //     counts[i]=worker_load;
        //     displs[i]=displs[i-1]+counts[i-1];
        // }

        // MPI_Scatterv();

        vector<int> sudokuMasterIdx;

        // send total numsudoku to workers
        MPI_Bcast(&numSudoku, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // send worker_load number of sudoku to each worker
        for (int i = 0; i < numSudoku; i++)
        {
            int rcvr = i % world_size;
            if (rcvr != 0)
            {
                MPI_Send(all_sudoku[i], 81, MPI_CHAR, rcvr, FROM_MASTER, MPI_COMM_WORLD);
                // cout<<"IN MASTER SENT "<<i<<endl;
            }
            else
            {
                sudokuMasterIdx.push_back(i);
            }
        }

        // cout << "SENT " << endl;
        int ctr=0;
        for (auto &x : sudokuMasterIdx)
        {
            char **board = new char *[WIDTH_9X9];
            // cout << "IN HERE" << endl;
            for (int i = 0; i < WIDTH_9X9; i++)
            {
                board[i] = new char[WIDTH_9X9];
                for (int j = 0; j < WIDTH_9X9; j++)
                {
                    // cout << all_sudoku[x][(i * 9) + j] << " " << endl;
                    if (all_sudoku[x][(i * 9) + j] == '0')
                    {
                        board[i][j] = '.';
                    }
                    else
                    {
                        board[i][j] = all_sudoku[x][(i * 9) + j];
                    }
                }
            }
            if (isValidSudoku(board, WIDTH_9X9))
            {
                // cout << ("isValidSudoku() before solving returned true.") << endl;
                solveBoard(board, 0, 0, WIDTH_9X9);
                // // cout << ("\nSolved board:");
                // printBoard(board, WIDTH_9X9);
                if (isValidSudoku(board, WIDTH_9X9))
                {
                    // cout << ("isValidSudoku() after solving returned true.") << endl;
                    if (isBoardSolved(board, WIDTH_9X9))
                    {
                        // cout << ("isBoardSolved() after solving returned true.") << endl;
                    }
                    else
                    {
                        // cout << ("isBoardSolved() after solving returned false.") << endl;
                    }
                }
                else
                {
                    // cout << ("isValidSudoku() after solving returned false.") << endl;
                }
            }
            else
            {
                // cout << ("isValidSudoku() before solving returned false.") << endl;
            }
            // for (int i = 0; i < WIDTH_9X9; i++)
            // {
            //     for (int j = 0; j < WIDTH_9X9; j++)
            //     {
            //         // cout << board[i][j] << " ";
            //     }
            //     // cout << endl;
            // }
            // cout << "END ONE TIME FROM MASTER" << endl;
            cout<<++ctr<<" done in master"<<endl;
        }
        char rcvd_sudoku[81];
        mtype=FROM_WORKER;
        for(int i=0;i<numSudoku;i++){
            int sender=i%world_size;
            if(sender!=0){
                MPI_Recv(&rcvd_sudoku,81,MPI_CHAR,sender,FROM_WORKER,MPI_COMM_WORLD,&status);
                cout<<"RCVD "<<i<<" FROM WORKER "<<sender<<endl;
                string res="";
                for(int i=0;i<81;i++){
                    res+=rcvd_sudoku[i];
                }
                res_sudoku.push_back(res);
                cout<<res<<endl;
            }
        }
        // for(auto x:res_sudoku){
        //     cout<<x<<endl;
        // }
        //MPI_Barrier(MPI_COMM_WORLD);
        // get finish time
        double finish = MPI_Wtime();
        printf("Time= %f\n", finish - start);
    }

    // WORKER
    else
    {
        mtype = FROM_MASTER;
        int numSudoku;
        MPI_Bcast(&numSudoku, 1, MPI_INT, mtype, MPI_COMM_WORLD);
        int rank_with_more_sudoku = numSudoku % world_size;
        int num_to_recv;
        if (world_rank < rank_with_more_sudoku)
        {
            num_to_recv = (numSudoku / world_size) + 1;
        }
        else
        {
            num_to_recv = (numSudoku / world_size);
        }
        // cout << "INSIDE WORKER NUMBER " << world_rank << " NUM TO RECV " << num_to_recv << endl;
        // if(world_rank==1)
        int ctr=0;
        for (int w = 0; w < num_to_recv; w++)
        {
            char a[81];
            MPI_Recv(&a, 81, MPI_CHAR, 0, mtype, MPI_COMM_WORLD, &status);
            cout<<"IN WORKER "<<world_rank<<" RCVD "<<w<<endl;
            char **board = new char *[WIDTH_9X9];
            // cout << "IN HERE" << endl;
            for (int i = 0; i < WIDTH_9X9; i++)
            {
                board[i] = new char[WIDTH_9X9];
                for (int j = 0; j < WIDTH_9X9; j++)
                {
                    // cout << a[(i * 9) + j] << " " << endl;
                    if (a[(i * 9) + j] == '0')
                    {
                        board[i][j] = '.';
                    }
                    else
                    {
                        board[i][j] = a[(i * 9) + j];
                    }
                }
            }
            cout<<"IN WORKER "<<world_rank<<" GOT PAST INIT FOR "<<w<<endl;
            // for (int i = 0; i < WIDTH_9X9; i++)
            // {
            //     for (int j = 0; j < WIDTH_9X9; j++)
            //     {
            //         // cout << board[i][j] << " ";
            //     }
            //     // cout << endl;
            // }
            if (isValidSudoku(board, WIDTH_9X9))
            {
                // cout << ("isValidSudoku() before solving returned true.") << endl;
                solveBoard(board, 0, 0, WIDTH_9X9);
                // // cout << ("\nSolved board:");
                if (isValidSudoku(board, WIDTH_9X9))
                {
                    // cout << ("isValidSudoku() after solving returned true.") << endl;
                    if (isBoardSolved(board, WIDTH_9X9))
                    {
                        // cout << ("isBoardSolved() after solving returned true.") << endl;
                    }
                    else
                    {
                        // cout << ("isBoardSolved() after solving returned false.") << endl;
                    }
                }
                else
                {
                    // cout << ("isValidSudoku() after solving returned false.") << endl;
                }
            }
            else
            {
                // cout << ("isValidSudoku() before solving returned false.") << endl;
            }
            cout<<"IN WORKER "<<world_rank<<" DONE "<<ctr++<<endl;
            char res[81];
            for (int i = 0; i < WIDTH_9X9; i++)
            {
                for (int j = 0; j < WIDTH_9X9; j++)
                {
                    res[(i*9)+j]=board[i][j];
                    cout<<res[i*9+j]<<" ";
                }
                cout<<endl;
            }
            int test=1;
            // mtype=FROM_WORKER;
            // MPI_Request  request;
            MPI_Send(&res,81,MPI_CHAR,0,FROM_WORKER,MPI_COMM_WORLD);
            cout<<"SENT "<<w<<" TO MASTER FROM WORKER "<<world_rank<<endl;
        }
    }
    // Finalize the MPI environment.
    MPI_Finalize();
}
