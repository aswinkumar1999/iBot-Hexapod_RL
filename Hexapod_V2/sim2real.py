import numpy as np

# Initialise Transformation Matrix
trans_mat = np.zeros((18,18))
trans_mat_const = 600

# Fill up Transformation Matrix for Sim to Real

trans_mat[15,0]  = -1
trans_mat[3,1]   = -1
trans_mat[9,2]   = 1
trans_mat[16,3]  = -1
trans_mat[10,4]  = 1
trans_mat[4,5]   = -1
trans_mat[5,6]   = -1
trans_mat[17,7]  = -1
trans_mat[11,8]  = 1
trans_mat[14,9]  = -1
trans_mat[2,10]  = -1
trans_mat[8,11]  = 1
trans_mat[13,12] = -1
trans_mat[1,13]  = -1
trans_mat[7,14]  = 1
trans_mat[0,15]  = -1
trans_mat[6,16]  = 1
trans_mat[12,17] = -1

trans_mat = trans_mat * trans_mat_const

# Print Matrix

print(trans_mat)
