import numpy as np
import socket
import signal
import time
from xbox360controller import Xbox360Controller

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

## Real 7cm - 90cm

## Sim  9cm - 110cm


# Code

m_id = [1,2,3,7,8,9,13,14,15,18,19,20,24,25,26,30,31,32]
# m_val= [1533,1480,1540,1540,1490,1480,1580,1615,1500,1174,1400,1600,1651,1500,1530,1450,1552,1953]
# m_val= [1533, 1530, 1440, 1490, 1490, 1480, 1580, 1615, 1400, 1174, 1400, 1650, 1651, 1500, 1530, 1450, 1552, 1953]
# m_val= [1433, 1530, 1400, 1380, 1360, 1480, 1580, 1365, 1355, 1339, 1400, 1730, 1711, 1500, 1590, 1450, 1592, 2023]
m_val= [1433, 1530, 1400, 1380, 1510, 1480, 1580, 1365, 1355, 1339, 1400, 1730, 1811, 1500, 1710, 1450, 1592, 2023]
# m_val= [1433, 1530, 1350, 1380, 1460, 1480, 1580, 1365, 1305, 1339, 1400, 1730, 1811, 1500, 1710, 1450, 1592, 2023]
speed=[1,2,5,10,20,50,100]

# Hardcoded values for Tripod Gait
t1=[0.3,-0.3,0.3,0.3,-0.3,0.3,0.2,0,0.2,0,-0.2,0,0,0,0,0,0,0]
t2=[-0.3,0.3,-0.3,-0.3,0.3,-0.3,0.2,0,0.2,0,-0.2,0,0,0,0,0,0,0]
t3=[-0.3,0.3,-0.3,-0.3,0.3,-0.3,0,0.2,0,-0.2,0,-0.2,0,0,0,0,0,0]
t4=[0.3,-0.3,0.3,0.3,-0.3,0.3,0,0.2,0,-0.2,0,-0.2,0,0,0,0,0,0]
steps=[t1,t2,t3,t4]

t1_new = list(np.array(t1).dot(trans_mat))
t2_new = list(np.array(t2).dot(trans_mat))
t3_new = list(np.array(t3).dot(trans_mat))
t4_new = list(np.array(t4).dot(trans_mat))
steps_new=[t1_new,t2_new,t3_new,t4_new]

TCP_IP = '192.168.31.21'
TCP_PORT = 5080

MESSAGE = ''

for i in range(len(m_id)):
    MESSAGE=MESSAGE+'#'+str(m_id[i])+'P'+str(m_val[i])

MESSAGE = MESSAGE + 'T200\r\n'

print(MESSAGE)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
s.send(MESSAGE.encode())

print("Bot Setup Complete")
input("Enter Key to Continue")
while(True):
    for k in range(4):
        MESSAGE = ''
        for i in range(len(m_id)):
            MESSAGE=MESSAGE+'#'+str(m_id[i])+'P'+str(m_val[i]+int(steps_new[k][i]))

        MESSAGE = MESSAGE + 'T200\r\n'
        print(MESSAGE)
        s.send(MESSAGE.encode())
        time.sleep(0.4)
