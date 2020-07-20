import sympy, collections, copy
# represent the x timed in 'recurrent formula'
x = sympy.Symbol("x")

# define node and its methods for row and col lists
class node:
    def __init__(self, val, type_, list_):
        self.value = val
        self.type = type_
        self.child = list_
    def remove_child(self, del_child):
        self.child.remove(del_child)
    def add_child(self, new_child):
        self.child.append(new_child)


# example setup--------------------------------------------------------
## initialise row and col lists
rowlist = [node(i, "row", []) for i in range(1, 5 + 1)]
collist = [node(i, "col", []) for i in range(1, 5 + 1)]

c_r1 = [4, 5]
c_r2 = [2, 3]
c_r3 = [3, 4]
c_r4 = [3, 4]
c_r5 = [1, 5]

c_r = [c_r1, c_r2, c_r3, c_r4, c_r5]

## adding child to row ndoes
for i in range(len(c_r)):
    cur = c_r[i]
    for j in range(len(cur)):
        rowlist[i].add_child(cur[j])
# change row to a dictionary
rowlist = dict(zip([x for x in range(1, 5 + 1)], rowlist))
r_len = len(rowlist)

# adding child to col nodes
r_c1 = [5]
r_c2 = [2]
r_c3 = [2, 3, 4]
r_c4 = [1, 3, 4]
r_c5 = [1, 5]


r_c = [r_c1, r_c2, r_c3, r_c4, r_c5]


for i in range(len(r_c)):
    cur = r_c[i]
    for j in range(len(cur)):
        collist[i].add_child(cur[j])
# change column to a dictionary
collist = dict(zip([x for x in range(1, 5 + 1)], collist))
c_len = len(collist)

#------------------------------------------------------------------------


## initialise stack list
## check that the row node is deleted for those with empty child list
stack = []
for i in range(1, r_len + 1):
    if rowlist[i].child == []:
        rowlist[i] = []
        
for i in range(1, c_len + 1):
    if collist[i].child == []:
        collist[i] = []

stack.append([rowlist, collist])

res_r = 1


# Find the deleting cell for given row and col list of nodes
def delete(row, col):
    ## find the list of rows who has the max children
    size_r = []
    for i in range(1, r_len + 1):
        r = row[i]
        if type(r) == node:
            size_r.append(len(r.child))
        else:
            size_r.append(0)
    
    ## the row of the cell by which to decompose
    val_r = size_r.index(max(size_r)) + 1
    del_r = row[val_r]
    
    ## find the list of cols who has the max children 
    size_c = []
    set_c = del_r.child
    for i in range(1, c_len + 1):
        c = col[i]
        if type(c) == node:
            if c.value in set_c:
                size_c.append(len(c.child))
            else:
                size_c.append(0)
        else:
            size_c.append(0)
    
    ## the col of the cell by which to decompose
    val_c = size_c.index(max(size_c)) + 1
    del_c = col[val_c]

    return (del_r, del_c)

# Build basic block, 'Recurrent Formula', for recusion
    
# find the row and col list of inclusion board
def recurrent_i(board):    
    row_i, col_i = (board[0], board[1])
    ## find cell of the board by which to decompose
    del_r, del_c = delete(row_i, col_i) 
    ## update row list
    row_i[del_r.value] = []
    ## update element of col list's child list
    ## delete del_r in all child list of col nodes
    ## if child list empty, replace with empty list in col list
    for c in del_r.child:
        col_i[c].remove_child(del_r.value)
        if col_i[c].child == []:
            col_i[c] = []
        
    ## Similarly, update col list
    
    ## update the del_c
    del_c = col_i[del_c.value]
    if type(del_c) == node:
        col_i[del_c.value] = []
        ## update element of row list's child list
        for r in del_c.child:
            row_i[r].remove_child(del_c.value)
            if row_i[r].child == []:
                row_i[r] = []

# find the row and col list of exclusion board
def recurrent_e(board):
    row_e, col_e = (board[0], board[1])
    ## find cell of the board by which to decompose
    del_r, del_c = delete(row_e, col_e) 
    
    ## update elements of col and row list's child list
    
    col_e[del_c.value].remove_child(del_r.value)
    if col_e[del_c.value].child == []:
        col_e[del_c.value] = []
        
    row_e[del_r.value].remove_child(del_c.value)
    if row_e[del_r.value].child == []:
        row_e[del_r.value] = []



# recursion step
# always process the first level of the stack
def block_stack(stack):
    ## get seperate copies of row and col lists for inclu and exclu board
    board = stack[0]
    board_e = copy.deepcopy(board)
    board_i = copy.deepcopy(board)
    
    ## generate a new exclusion board
    recurrent_e(board_e)
    list_e = copy.deepcopy(board)
    ## update the row and col in the first level of stack
    list_e[0] = board_e[0]; list_e[1] = board_e[1]
    stack.append(list_e)    
    
    ## similar as above
    recurrent_i(board_i)
    list_i = copy.deepcopy(board)
    list_i[0] = board_i[0]; list_i[1] = board_i[1]
    ## add another x for inclusion board
    list_i += [x]
    stack.append(list_i)
    
    ## remove the 1st level stack
    stack.remove(stack[0])
    
    ## compute the length of children's list for each row node
    board_ = stack[0]
    row_ = board_[0]
    
    res_r = 0
    for i in range(1, r_len + 1):
        r = row_[i]
        if type(r) == node:
            res_r += 1
        else:
            res_r
    
    return(stack, res_r)
    
    
# recursion to find coeffs

## recursion
def build_stack(res_r, stack):
    ## run block_stack while row list not empty (res_r not 0)
    while res_r > 0:
        res_r = block_stack(stack)[1]
    
    i = 0
    ## if len_r is 0
    for level in stack:
        ## update the number of loopings
        i += 1
        
        row_ = level[0]
        
        list_node = []
        for k in range(1, r_len + 1):
            r = row_[k]
            list_node.append(type(r) == node)
        
        ## check if all r in row has only empty list
        if any(list_node) == True:
            ### update stack
            remove_level = copy.deepcopy(stack[0:i - 1])
            stack[0:i - 1] = []
            stack += remove_level
            break
        else:
            if i == len(stack):
                return stack
       
    ## update len_r
    for k in range(1, r_len + 1):
        r = row_[k]
        if type(r) == node:
            res_r += 1
        else:
            res_r
    ## restart the recursion for the new first level
    build_stack(res_r, stack)
    
    
## count the freq for each order of x
def count_ord(stack):
    ### remove the row and col in each level of the stacks
    for level in stack:
        level.remove(level[0])
        level.remove(level[0])
    
    ### store the coeff in a dictionary to access
    len_ = [len(i) for i in stack]
    counter = collections.Counter(len_)
    coeff = dict(counter)
    
    return coeff