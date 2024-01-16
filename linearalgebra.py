################
#input / output#
################

def get_matrix():
    """
    
    Helper function to take input from user
    
    takes input from the user of the size (m by n)
    takes m rows of space-separated input, parses it into a list of n elements
    
    Returns
    -------
    matrix: 2-D array
        a two dimensional array of size m by n
    """
    m = input("How many rows are in your matrix?\n")
    n = input("How many columns are in your matrix?\n")
    
    if not (m.isdigit() and n.isdigit()): #ensures the size is valid
        raise Exception("Error: Your dimensions are not integers")
        
    
    m = int(m)
    n = int(n)
    
    if m < 1 or n < 1: #ensures the size is valid
        raise Exception("Error: Your dimensions are not positive")
    
    else:
        matrix = []
        for i in range(m): #generates m input fields (one for each row)
            row = input(f"Enter row {i + 1}, with each entry separated by a space\n")
            row = row.split()
            if len(row) != n: #ensures the matrix is rectangular
                raise Exception(f"Error: Row {i + 1} does not have the correct number of entries")
            for j in range(n):
                row[j] = eval(row[j]) #convert fractional inputs to float
            matrix.append(row)
        return matrix

def print_matrix(Matrix, Max_decimal = 3):
    """
    Parameters
    ----------
    Matrix : 2-D array
        The desired matrix to be printed
    Max_decimal : int, optional
        Where the decimal should be truncated. The default is 3.

    Prints
    ------
    The inputted matrix in a readable format
    """
    max_length = 0
    for row in Matrix:
        for j in range(len(row)):
            row[j] = round(row[j], Max_decimal)
            max_length = max(max_length, len(str(row[j]))) #determine the longest entry for spacing purposes
    
    print(" " +"_" * ((max_length + 1) * len(Matrix[0]) + 1)) #decoration
    for row in Matrix:
        print("|", end= " ") #decoration
        for entry in row:
            entry = str(entry)
            while len(entry) < max_length: #ensure every entry is of equal length
                entry += " "
            print(entry, end = " ")
        print("|") #decoration
    print("|" + "_" * ((max_length + 1) * len(Matrix[0]) + 1) + "|") #decoration

def print_poly(deter_poly, max_decimal = 3):
    """
    
    Helper function to print char_poly()
    
    Parameters
    ----------
    deter_poly : list
        the input polynomial, powers in descending order (standard form)
    max_decimal : int, optional
        number of decimals to round to, the default is 3.

    Prints
    -------
    The polynomial in a readable format
    """
    for i in range(len(deter_poly)):
        deter_poly[i] = round(deter_poly[i], max_decimal) #round according to max_decimal
    n = len(deter_poly) - 1
    deter_str = str(deter_poly[0]) + f"x^{n}" #format first term
    for i in range(1, n + 1):
        if str(deter_poly[i])[0] == "-": #format signs
            deter_poly[i] *= -1
            deter_str += " - "
        else:
            deter_str += " + "
        deter_str += f"{deter_poly[i]}x^{n-i}" #format coefficients with variables
    deter_str = deter_str[:-3]
    print(deter_str)

def print_eigenspace(eigen_dict, max_decimal = 3):
    """
    Parameters
    ----------
    eigen_dict : dict
        dictionary of eigenvalues and their corresponding eigenspaces
    max_decimal : int, optional
        maximum number of decimals, the default is 3.

    Prints
    -------
    Eigenvectors and corresponding eigenspaces in a readable format

    """
    for (eigenvalue, eigenspace) in eigen_dict.items():
        print("=" * 30) #decoration
        print(f"Eigenvalue: {round(eigenvalue, max_decimal)}")
        print("Corresponding vector(s) of eigenspace:")
        print_matrix(eigenspace, max_decimal)


########################
#addition / subtraction#
########################

def add_matrix(matrix1 = None, matrix2 = None, sign = None):
    """
    Parameters
    ----------
    matrix1 : 2-D array, optional
        first input matrix, if no argument is given, calls get_matrix()
    matrix2 : 2-D array, optional
        second input matrix, if no argument is given, calls get_matrix()
    sign : str, optional
        + indicating to add matrices or - indicating to subtract matrices
    Returns
    -------
    output_matrix : 2-D array
        the sum or difference of given matrices
    """
    if sign == None:
        print("Would you like to add or subtract matrices?")
        sign = input('Type "+" to add or "-" to subtract\n')
    if sign != "+" and sign != "-": #ensure valid input
        raise Exception("Invalid sign input")
    if matrix1 == None:
        print("Enter the first matrix:")
        matrix1 = get_matrix()
    if matrix2 == None:
        print("Enter the second matrix:")
        matrix2 = get_matrix()
    if len(matrix1) != len(matrix2):
        raise Exception("Incongruent row sizes") #ensure matrices are same size
    if len(matrix1[0]) != len(matrix2[0]):
        raise Exception("Incongruent column sizes")
    if sign == "-":
        matrix2 = negate_matrix(matrix2) #convert matrix for subtraction
    output_matrix = []
    for i in range(len(matrix1)):
        output_row = []
        for j in range(len(matrix1[0])):
            output_entry = matrix1[i][j] + matrix2[i][j] #add term-by-term
            output_row.append(output_entry)
        output_matrix.append(output_row)
    return output_matrix

def negate_matrix(input_matrix):
    """
    
    Helper function for add_matrix()
    
    Parameters
    ----------
    input_matrix : 2-D array
        the input matrix

    Returns
    -------
    output_matrix : 2-D array
        the negation of the inputted matrix
    """
    output_matrix = []
    for row in input_matrix:
        output_row = []
        for entry in row:
            output_row.append(entry * -1) #multiply term-by-term
        output_matrix.append(output_row)
    
    return output_matrix

################
#multiplication#
################

def multiply_matrices(matrix_list = []):
    """
    Parameters
    ----------
    matrix_list : 3-D array
        matrices to be multiplied
    takes input of how many matrices are to be multiplied
    calls get_matrix() that many times
    calculates the product of the matrices

    Returns
    -------
    result_matrix : 2-D array
        result of the product of given matrices
    """
    if matrix_list == []:
        num_matrices = int(input("How many matrices do you want to multiply?\n"))
        for i in range(num_matrices):
            print(f"Enter matrix {i + 1}")
            matrix_list.append(get_matrix())
    else:
        num_matrices = len(matrix_list)
    while num_matrices > 1: #while there are still two matrices to multiply
        matrix_list[-2] = matrix_product(matrix_list[-2], matrix_list[-1]) #multiply rightmost matrices first
        matrix_list.pop(-1)
        num_matrices -= 1
    result_matrix = matrix_list[0]
    return result_matrix

def matrix_product(m1, m2):    
    """
    
    Helper function for multiply_matrices()
    
    Parameters
    ----------
    m1 : 2-D array
        left matrix
    m2 : 2-D array
        right matrix

    Returns
    -------
    new_matrix : 2-D array
        product of m1 and m2
    """
    m2 = transpose_matrix(m2) #transpose second matrix to convert row by column multiplication into row by row dot products
    new_matrix = []
    for i in range(len(m1)):
        temp_list = []
        for j in range(len(m2)):
            temp_list.append(dot_product(m1[i], m2[j])) #i,j will become new matrix location
        new_matrix.append(temp_list)
    return new_matrix

#############
#dot product#
#############

def dot_product(v1 = None, v2 = None):
    """
    
    Helper function for matrix_product()
    
    Parameters
    ----------
    v1 : list
        vector of integers
    v2 : list
        vector of integers

    Returns
    -------
    total : int
        the dot product of v1 and v2
    """
    if v1 == None:
        v1 = input("Enter vector 1 with each entry separated by a space\n")
        v1 = v1.split()
        for i in range(len(v1)):
            v1[i] = eval(v1[i]) #convert fractional entry into float
    if v2 == None:
        v2 = input("Enter vector 2 with each entry separated by a space\n")
        v2 = v2.split()
        for i in range(len(v2)):
            v2[i] = eval(v2[i])
    if len(v1) != len(v2):
        raise Exception("Error: vectors must be of the same length")
    else:
        total = 0
        for i in range(len(v1)):
            total += (v1[i] * v2[i]) #sum entry-by-entry products
        return total

###############
#transposition#
###############

def transpose_matrix(Matrix = None):
    """
    
    Helper function for matrix_product()
    
    Parameters
    ----------
    Matrix : 2-D array
        input matrix

    Returns
    -------
    new_matrix : 2-D array
        transpose of the input matrix
    """
    if Matrix == None:
        Matrix = get_matrix()
    new_matrix = []
    for j in range(len(Matrix[0])):
        temp_list = []
        for i in range(len(Matrix)):
            temp_list.append(Matrix[i][j]) #reverse indices to create transposed matrix
        new_matrix.append(temp_list)
    return new_matrix

###############
#row reduction#
###############

def reduced_echelon(input_matrix = None):
    """
    Parameters
    ----------
    input_matrix : 2-D array, optional
        the input matrix. if no argument is given, calls get_matrix()

    Returns
    -------
    echelon : 2-D array
        reduced echelon form of the inputted matrix
    """
    if input_matrix == None:
        input_matrix = get_matrix()
    echelon = [] #create new matrix, transfer echelon form rows from input_matrix into echelon one at a time
    print("Row reduction:")
    while len(input_matrix) > 0: #until all rows have been converted
        temp_matrix = echelon + input_matrix #used for printing purposes
        print("-" * 30)
        print_matrix(temp_matrix)
        input_matrix = sort_matrix(input_matrix) #order the matrix to be upper triangular
        input_matrix = scale_matrix(input_matrix) #scale leading coefficients to be 1
        for i in range(1, len(input_matrix)):
            if (num_leading_zeros(input_matrix[0]) == num_leading_zeros(input_matrix[i])): #if they are both 1 (same leading coefficient index)
                input_matrix[i] = subtract_row(input_matrix[i], input_matrix[0]) #replace the bottom one with the difference
        echelon.append(input_matrix[0])
        input_matrix.pop(0) #transfer top row from input_matrix to echelon
    #we are now fully in echelon form. we need to back substitute
    for i in range(len(echelon) - 1, -1, -1): #work backwards, from bottom to top
        print("-" * 30)
        print_matrix(echelon)
        for j in range(i): #for each row above row j
            pivot = num_leading_zeros(echelon[i])
            if pivot in range(len(echelon[0])): #if the row has a valid pivot position
                scalar = echelon[j][pivot] / echelon[i][pivot] #to determine what to scale our subtraction
                echelon[j] = subtract_row(echelon[j], echelon[i], scalar) #subtract scalar * bottom row from above row (leaving 0)
    
    epsilon = .00001 #convert integer-valued floats to integers
    for row in range(len(echelon)):
        for column in range(len(echelon[row])):
            if abs(echelon[row][column] - round(echelon[row][column])) < epsilon: #close enough to round float
                echelon[row][column] = round(echelon[row][column])
    print("=" * 30)
    return echelon

def sort_matrix(input_matrix):
    """
    
    Helper function for reduced_echelon()
    
    Parameters
    ----------
    input_matrix : 2-D array
        the inputted matrix

    Returns
    -------
    output_matrix : 2-D array
        the inputted matrix, rows rearranged to ensure upper triangularity
    """
    zero_rows = []
    for row in input_matrix:
        zero_rows.append([num_leading_zeros(row), row]) #to allow for sorting of rows
    zero_rows.sort()
    output_matrix = []
    for row in zero_rows:
        output_matrix.append(row[1]) #append the sorted rows in order
    return output_matrix

def scale_matrix(input_matrix):
    """
    
    Helper function for reduced_echelon()
    
    Parameters
    ----------
    input_matrix : 2-D array
        the input matrix

    Returns
    -------
    output_matrix : 2-D array
        the inputted matrix, rows scaled to have the first non-zero entry equal to 1
    """
    output_matrix = []
    for row in input_matrix: #for loop to scale each row, leaving leading coefficient of one
        leading_coefficient = 1
        leading_index = num_leading_zeros(row)
        if leading_index in range(len(row)):
            leading_coefficient = row[leading_index]
        output_row = scale_row(row, (1 / leading_coefficient))
        output_matrix.append(output_row)
    return output_matrix

def subtract_row(row1, row2, scalar = 1):
    """
    
    Helper function for reduced_echelon()
    
    Parameters
    ----------
    row1 : list
        first input row
    row2 : list
        second input row
    scalar : float, optional
        the scalar to multiply the second row by prior to subtraction, the default is 1

    Returns
    -------
    output_row : list
        result row of the row subtraction operation
    """
    output_row = []
    row2 = scale_row(row2, scalar) #scale by desired scalar
    for i in range(len(row1)):
        output_entry = row1[i] - row2[i] #new entry is difference of each corresponding entry from original rows
        output_row.append(output_entry)
    return output_row


def num_leading_zeros(Row):
    """
    
    Helper function for reduced_echelon()

    Parameters
    ----------
    Row : list
        matrix row

    Returns
    -------
    num_zeros : int
        number of leading zeros in the row

    """
    num_zeros = 0
    for entry in Row: #helps uskeeps track of pivot positions
        if entry == 0:
            num_zeros += 1
        else:
            break
    return num_zeros

def scale_row(input_row, scalar):
    """
    
    Helper function for reduced_echelon()

    Parameters
    ----------
    input_row : list
        the input row
    scalar : float
        the value to scale the row by

    Returns
    -------
    output_row : list
        the input row, scaled by the scalar value
    """
    output_row = []
    for entry in input_row: #scales each entry by desired scalar
        output_entry = entry * scalar
        output_row.append(output_entry)
    return output_row

###########
#inversion#
###########

def inverse(Matrix = None):
    """
    Parameters
    ----------
    Matrix : 2-D array
        the input matrix

    Returns
    -------
    Matrix : 2-D array
        the inverse of the original matrix
    """
    if Matrix == None:
        Matrix = get_matrix()
    Matrix = augment_identity(Matrix) #augment the identity of desired size
    Matrix = reduced_echelon(Matrix) #row reduce it
    Matrix = split_inverse(Matrix) #if the left side is now the identity, the right side will be the inverse
    return Matrix

def augment_identity(Matrix):
    """
    
    Helper function for inverse()

    Parameters
    ----------
    Matrix : 2-D array
        the input matrix

    Returns
    -------
    augmented_matrix : 2-D array
        the matrix augmented with the identity matrix of same size

    """
    if len(Matrix) != len(Matrix[0]):
        raise Exception("Error: The matrix is not square")
    else:
        size = len(Matrix)
        identity_matrix = identity(size)
        augmented_matrix = []
        for row_index in range(len(Matrix)):
            new_row = Matrix[row_index] + identity_matrix[row_index] #combine the original matrix row with the identity matrix row
            augmented_matrix.append(new_row)
        return augmented_matrix

def split_inverse(Matrix):
    """
    
    Helper function for inverse()

    Parameters
    ----------
    Matrix : 2-D array
        the input matrix

    Returns
    -------
    right_matrix : 2-D array
        the inverted matrix

    """
    if 2 * len(Matrix) != len(Matrix[0]):
        raise Exception("Error: The matrix is not of proper size")
    else:
        halfway = len(Matrix) #determine where to slice the augmented matrix
        left_matrix = []
        right_matrix = []
        
        for row in Matrix: #split each row into two variables for left and right side
            left_row = row[:halfway]
            right_row = row[halfway:]
            left_matrix.append(left_row)
            right_matrix.append(right_row)
        
        if left_matrix != identity(halfway): #ensures the left half is the identity matrix
            raise Exception("Error: Your matrix is not invertible")
        else:
            return right_matrix

def identity(N, c = 1):
    """
    
    Helper function for inverse()
    
    Parameters
    ----------
    N : int
        size of the matrix
    c : int
        scalar for the identity

    Returns
    -------
    id_matrix : 2-D array
        identity matrix of size N
    """
    id_matrix = []
    for i in range(N):
        id_matrix.append([0] * N) #make a zero matrix
    for i in range(N):
        id_matrix[i][i] = c #c remains 1 for all inverse operations, but it is helpful for A - lambda(I) calculations in eigenspaces
    return id_matrix

#############
#determinant#
#############

def determinant(Matrix = None):
    """
    Parameters
    ----------
    Matrix : 2-D array, optional
        the input matrix. if no matrix is provided, calls get_matrix()

    Returns
    -------
    total : float
        the determinant of the given matrix
    """
    if Matrix == None:
        Matrix = get_matrix()
    if len(Matrix) != len(Matrix[0]): #matrix must be square for a determinant to be calculated
        raise Exception("Error: The matrix is not square")
    n = len(Matrix)
    deter_terms = []
    for index_str in combinations(n): #every combination of row-column orders, without repeating indices
        product = 1
        col_index = 0
        for row in Matrix:
            product *= row[(int(index_str[col_index]))] #multiply them all
            col_index += 1 #go to the next combination index
        product *= determinant_sign(index_str) #determine sign then scale entry by 1 or -1
        deter_terms.append(product) #add term to the list
    total = sum(deter_terms) #output sum of the list
    return total

def combinations(N):
    """
    
    Helper function for determinant()
    
    Parameters
    ----------
    N : int
        number of indexes to be rearranged

    Returns
    -------
    combination_list : str
        list of all index arrangements

    """
    combination_list = []
    for i in range(factorial(N)):
        entry = ""
        for x in range(N):
            num = i // factorial(N - 1 - x) #count the multiplicity of the given factorial
            i = i % factorial(N - 1 - x)  #reassign n with the remainder
            entry += str(num) #append the num to the string
        combination = ""
        digits = list(k for k in range(N))
        for num in entry:
            index = int(num)
            combination += str(digits[index])
            del digits[index]
        combination_list.append(combination)
    return combination_list

def determinant_sign(Index_str):
    """
    
    Helper function for determinant()
    
    Parameters
    ----------
    Index_str : str
        string of column indexes of the determinant element

    Returns
    -------
    sign : int
        a multiplier to add or subtract the term in determinant calculation
    """
    sign = 1
    range_list = [x for x in range(len(Index_str))]
    for char in Index_str:
        if range_list.index(int(char)) % 2 == 1: # + - + -... pattern will tell us the term sign
            sign *= -1
        range_list.remove(int(char)) #ensures we don't use the same index twice
    return sign

def factorial(n):
    """
    
    Helper function for combinations()
    
    Parameters
    ----------
    n : int
        input number

    Returns
    -------
    factorial : int
        returns factorial of n
    """
    factorial = 1
    for i in range(1, n + 1):
        factorial *= i #1 * 2 * ... * n
    return factorial

###############
#cross product#
###############

def cross_product(v1 = None, v2 = None):
    """
    Parameters
    ----------
    v1 : list, optional
        The left vector of the cross product, calls for input if none provided
    v2 : _type_, optional
        _description_, by default None

    Returns
    -------
    cross
        The cross product of the two lists.

    Raises
    ------
    Exception
        If input is not two lists of size 3
    """
    if v1 == None:
        v1 = input("Enter vector 1 with each entry separated by a space\n")
    if v2 == None:
        v2 = input("Enter vector 2 with each entry separated by a space\n")
    v1 = [int(i) for i in v1.split()]
    v2 = [int(i) for i in v2.split()]
    if len(v1) != 3 or len(v2) != 3:
        raise Exception("Error: Your vectors are not of length 3.")
    imatrix = [v1[1:], v2[1:]]
    jmatrix = [v1[::2], v2[::2]]
    kmatrix = [v1[:2], v2[:2]]
    i = determinant(imatrix)
    j = -1 * determinant(jmatrix)
    k = determinant(kmatrix)
    cross = [i,j,k]
    return cross



############
#null space#
############

def null_space(Matrix = None):
    """
    Parameters
    ----------
    Matrix : 2-D array, optional
        the input matrix, if none provided calls get_matrix()

    Returns
    -------
    null_cols : 2-D array
        array representing the span of the null space of the input matrix
    """
    if Matrix == None:
        Matrix = get_matrix()
    Matrix = reduced_echelon(Matrix) #reduce the matrix
    pivot_list = find_pivots(Matrix) #locate the pivots
    null_indices = []
    for i in range(len(Matrix[0])):
        if i not in pivot_list: #columns without pivots
            null_indices.append(i)
    if len(null_indices) == 0:
        raise Exception("The matrix is linearly independent") #every column has a pivot
    print_matrix(Matrix)
    null_cols = []
    for j in null_indices:
        null_col = get_column(Matrix, j)
        while len(null_col) < len(Matrix[0]):
            null_col.append(0)
        null_col[j] = -1
        null_cols.append(null_col) #collect all columns in a matrix
    null_cols = transpose_matrix(null_cols) #transpose to display columns vertically
    return null_cols

def find_pivots(Matrix):
    """
    
    Helper function for null_space() and col_space()
    
    Parameters
    ----------
    Matrix : 2-D array
        the input matrix

    Returns
    -------
    pivot_list : list
        list of column indexes of the pivots
        None indicates that the column does not have a pivot
    """
    pivot_list = []
    for row in Matrix:
        if 1 in row: #the first nonzero term will be 1 because of scale_matrix()
            pivot_location = row.index(1)
            pivot_list.append(pivot_location)
        else:
            pivot_list.append(None) #all zero row
    return pivot_list

def get_column(Matrix, col_index):
    """
    
    Helper function for null_space()
    
    Parameters
    ----------
    Matrix : 2-D array
        the input matrix
    col_index : int
        index of column to be retrieved

    Returns
    -------
    column : list
        the desired column
    """
    column = []
    for row in Matrix:
        column.append(row[col_index]) #append the desired column index of every row to a list
    return column

##############
#column space#
##############

def col_space(Matrix = None):
    """
    Parameters
    ----------
    Matrix : 2-D array, optional
        the input matrix, if none provided calls get_matrix()

    Returns
    -------
    pivot_cols : 2-D array
        array representing the span of the column space of the input matrix
    """
    if Matrix == None:
        Matrix = get_matrix()
    matrix_cols = transpose_matrix(Matrix) #transpose for later
    ref = reduced_echelon(Matrix)
    print_matrix(ref)
    pivot_list = find_pivots(ref)
    pivot_cols = []
    for pivot in pivot_list:
        if pivot != None:
            pivot_cols.append(matrix_cols[pivot]) #choose the desired columns
    pivot_cols = transpose_matrix(pivot_cols) #transpose matrix to return to original layout
    return pivot_cols

###########################
#characteristic polynomial#
###########################

def char_poly(input_matrix = None):
    """
    Parameters
    ----------
    input_matrix : 2-D array, optional
        the input matrix. if no matrix is provided, calls get_matrix()

    Returns
    -------
    total : float
        the determinant of the given matrix
    """
    #modified version of determinant() - takes entries as lists
    if input_matrix == None: #see determinant() for more documentation
        input_matrix = get_matrix()
    if len(input_matrix) != len(input_matrix[0]):
        raise Exception("Error: The matrix is not square")
    n = len(input_matrix)
    Matrix = []
    for i in range(n):
        Matrix_row = []
        for j in range(n):
            if i == j:
                Matrix_row.append([-1, input_matrix[i][j]])
            else:
                Matrix_row.append([input_matrix[i][j]])
        Matrix.append(Matrix_row)
    deter_terms = []
    for index_str in combinations(n):
        product = [1] #define datatype as list
        col_index = 0
        for row in Matrix:
            product = poly_mult(row[(int(index_str[col_index]))], product) #poly_mult takes place of *
            col_index += 1
        product = poly_mult([determinant_sign(index_str)],product) #poly_mult takes place of *
        deter_terms.append(product)
    deter_poly = []
    for term in deter_terms:
        term = term.reverse() #reverse the terms for addition
    for i in range(n + 1):
        poly_term = 0
        for term in deter_terms:
            if i in range(len(term)): #add index by index
                poly_term += term[i]
        deter_poly.append(poly_term)
    deter_poly.reverse() #reverse again to return to standard form
    return deter_poly

def poly_mult(poly1, poly2):
    """
    
    Helper function for char_poly()
    
    Parameters
    ----------
    poly1 : list
        first polynomial to be multiplied. powers in descending order (standard form)
    poly2 : list
        second polynomial to be multiplied. powers in descending order (standard form)

    Returns
    -------
    poly_product : list
        product of poly1 and poly2. powers in descending order (standard form)
    """
    poly_product = [0] * (len(poly1) + len(poly2) - 1) #establish the desired length of the product
    for (i1, num1) in enumerate(poly1):
        for (i2, num2) in enumerate(poly2):
            poly_product[i1 + i2] += (num1 * num2) #add each product to the corresponding index
    return poly_product

#############
#eigenvalues#
#############

def eigenvalues(input_matrix = None):
    """
    Parameters
    ----------
    input_matrix : 2-D array, optional
        the input matrix, if none given, calls get_matrix()

    Returns
    -------
    values : list
        list of integer eigenvalues of the inputted matrix

    """
    if input_matrix == None:
        print("Warning: Feature currently only supports integer eigenvectors") #can only use rational root theorem on rational roots :( not sure how to get other roots of the characteristic polynomial
        choice = input("Proceed? y/n\n")
        if choice != "y" and choice != "n":
            raise Exception("Invalid choice")
        elif choice == "n":
            raise Exception(":/") # :( will attempt to resolve in the future
        input_matrix = get_matrix()
    values = []
    poly = char_poly(input_matrix) #get the characteristic polynomial
    for i in potential_roots(poly):
        if eval_poly(poly, i) == 0: #zero of the polynomial
            values.append(i)
    if len(values) == 0:
        print("No eigenvectors detected")
    return values

def eval_poly(poly, constant):
    """
    Parameters
    ----------
    poly : list
        polynomial to be evaluated
    constant : int
        the integer to evaluate the polynomial at

    Returns
    -------
    total : int
        the polynomial function evaluated at x = constant
    """
    total = 0
    poly_copy = poly[::-1] #reverse the polynomial
    for i in range(len(poly_copy)):
        total += poly_copy[i] * (constant ** i) #sum term-by-term
    return total

def potential_roots(poly):
    """
    Parameters
    ----------
    poly : list
        polynomial to be evaluated
    
    Returns
    -------
    roots_list : list
        potential roots of the polynomial, according to the rational root theorem
    """
    roots_list = [] #algorithm nearly identical to the rational root theorem
    poly_copy = poly[:]
    if type(poly_copy[-1]) != int:
       raise Exception("Non-integer entries are not currently supported")
    if poly_copy[-1] == 0:
        roots_list.append(0) #if we need to remove zero, we should add it to roots_list
    while poly_copy[-1] == 0: #we don't want 0 as the last term because of factoring issues
        poly_copy.pop(-1)
    for i in factors(abs(poly_copy[-1])):
        roots_list.append(i)
        roots_list.append(i * -1)
    return roots_list

def factors(num):
    """
    Parameters
    ----------
    num : int
        number to be factored

    Returns
    -------
    factor_list : list
        list of all positive integer factors of the number
    """
    factor_list = []
    for i in range(1, num + 1): #check each number from 1 to n
        if num % i == 0: #divisibility check
            factor_list.append(i)
    return factor_list

############
#eigenspace#
############

def eigenspace(input_matrix = None):
    """
    Parameters
    ----------
    input_matrix : 2-D array, optional
        the input matrix, if none provided, calls get_matrix()
    
    Returns
    -------
    space_dict : dict
        dictionary of integer eigenvalues and their corresponding eigenspaces

    """
    if input_matrix == None:
        print("Warning: Feature currently only supports integer eigenvectors")
        choice = input("Proceed? y/n\n") #can only use rational root theorem on rational roots :( not sure how to get other roots of the characteristic polynomial
        if choice != "y" and choice != "n":
            raise Exception("Invalid choice")
        elif choice == "n":
            raise Exception(":/")
        input_matrix = get_matrix()
    values = eigenvalues(input_matrix) #find the eigenvalues
    matrices = []
    for value in values: #calculate A - lambda(I) for each eigenvalue
        temp_identity = identity(len(input_matrix), value)
        temp_matrix = add_matrix(input_matrix, temp_identity, "-")
        print(f"For eigenvalue {value}")
        temp_ref = reduced_echelon(temp_matrix)
        matrices.append(temp_ref)
    space_dict = {}
    for i in range(len(matrices)):
        space = null_space(matrices[i]) #find the null space of A - lambda(I)
        space_dict.update({values[i]: space}) #create a dict of the eigenvalue : eigenspace
    return space_dict

##############
#main program#
##############

if __name__ == "__main__":
    options_dict = {
        "0": "add / subtract",
        "1": "multiply",
        "2": "row reduce",
        "3": "inverse",
        "4": "determinant",
        "5": "cross product",
        "6": "null space",
        "7": "column space",
        "8": "characteristic polynomial",
        "9": "eigenvalue",
        "10": "eigenspace",
        "q": "quit"
        }
    print("-" * 30) #decoration
    print("Select an option:")
    print("-" * 30) #decoration
    for (key, value) in options_dict.items():
        print(f"{key} - {value}") #list choices
    print("-" * 30) #decoration
    user_input = str(input())
    if user_input not in options_dict:
        raise Exception("Error: invalid option")
    else:
        max_decimal = input("How many decimal places do you want your solution rounded to?\n")
    if not max_decimal.isnumeric(): #ensure value can be converted to int
        raise Exception("Error: invalid decimal")
    elif user_input == "0":
        print_matrix(add_matrix(), int(max_decimal))
    elif user_input == "1":
        print_matrix(multiply_matrices(), int(max_decimal))
    elif user_input == "2":
        print_matrix(reduced_echelon(), int(max_decimal))
    elif user_input == "3":
        print_matrix(inverse(), int(max_decimal))
    elif user_input == "4":
        print(round(determinant(), int(max_decimal)))
    elif user_input == "5":
        print(cross_product())
    elif user_input == "6":
        print_matrix(null_space(), int(max_decimal))
    elif user_input == "7":
        print_matrix(col_space(), int(max_decimal))
    elif user_input == "8":
        print_poly(char_poly(), int(max_decimal))
    elif user_input == "9":
        eigen_list = eigenvalues()
        for eigenvalue in eigen_list:
            print(f"{round(eigenvalue, int(max_decimal))}", end = " ")
    elif user_input == "10":
        print_eigenspace(eigenspace(), int(max_decimal))
    elif user_input == "q":
        print("Goodbye!")
    else:
        raise Exception("Error: Invalid input")