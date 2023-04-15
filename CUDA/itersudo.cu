
IList<char> GetPossibleNumbers(char[][] board, int idx)
{
    int r = idx / 9;
    int c = idx % 9;
    IList<char> res = new List<char>(9);

    ISet<char> checkRepetitionRow = new HashSet<char>(9);
    ISet<char> checkRepetitionColumn = new HashSet<char>(9);
    ISet<char> checkBlock = new HashSet<char>(9);

    for (int i = 0; i < 9; i++)
    {
        if (board[i][c] != '.')
        {
            checkRepetitionColumn.Add(board[i][c]);
        }

        if (board[r][i] != '.')
        {
            checkRepetitionRow.Add(board[r][i]);
        }
    }

    int blockX = r / 3;
    int blockY = c / 3;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (board[blockX * 3 + i][blockY * 3 + j] != '.')
            {
                checkBlock.Add(board[blockX * 3 + i][blockY * 3 + j]);
            }
        }
    }

    for (char ch = '1'; ch <= '9'; ch++)
    {
        if (!checkBlock.Contains(ch) && !checkRepetitionRow.Contains(ch) && !checkRepetitionColumn.Contains(ch))
        {
            res.Add(ch);
        }
    }

    return res;
}

public
void SolveSudoku(char[][] board)
{
    int ? startIdx = null;

    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            if (board[i][j] == '.')
            {
                startIdx = i * 9 + j;
                break;
            }
        }

        if (startIdx != null)
        {
            break;
        }
    }

    if (!startIdx.HasValue)
    {
        return;
    }

    Stack<(int idx, IList<char> variants)> stack = new Stack<(int idx, IList<char> variants)>();
    stack.Push((startIdx.Value, GetPossibleNumbers(board, startIdx.Value)));

    while (stack.Count > 0)
    {
        var peek = stack.Peek();
        if (peek.variants.Count == 0)
        {
            stack.Pop();
            board[peek.idx / 9][peek.idx % 9] = '.';
            continue;
        }

        var lastChar = peek.variants[peek.variants.Count - 1];
        board[peek.idx / 9][peek.idx % 9] = lastChar;
        peek.variants.RemoveAt(peek.variants.Count - 1);

        var next = peek.idx + 1;
        while (true)
        {
            if (next > 80)
            {
                return;
            }

            if (board[next / 9][next % 9] == '.')
            {
                break;
            }

            next++;
        }

        var newVariants = GetPossibleNumbers(board, next);
        stack.Push((next, newVariants));
    }
}