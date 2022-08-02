from re import S
import torch
import sys
from metropolis import Metropolis
sys.path.append("./")

K = 4       # considered nearest Neighbors
SIZE = 10   # size of each repeated cell
REPEAT_TIMES = 10
J = -2

def move(unit):
    unit = -unit
    return unit

def fn(configuration,globally=False,unit=(0,1)):
    E = 0
    B = configuration.mean()
    def isCal(i,j):
        shape = configuration.shape
        s = configuration[i][j]
        S = 0
        if i > 0:
            S += s*configuration[i-1][j]
        if i < shape[0]-1:
            S += s*configuration[i+1][j]
        if j > 0:
            S += s*configuration[i][j-1]
        if j < shape[1]-1:
            S += s*configuration[i][j+1]
        return S, s
    if globally:
        Sigma_S = 0
        Sigma_s = 0
        shape = configuration.shape
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                S ,s = isCal(i,j)
                Sigma_S += S
                Sigma_s += s
        E = -0.5*J*Sigma_S + B*Sigma_s
    else:
        i = unit[0]
        j = unit[1]
        print(i,j)
        S ,s = isCal(i,j)
        E = -0.5*J*S + B*s
    return E

configuration = torch.full((SIZE, SIZE),1.0)

# Periodic boundaries
old_configuration = configuration.clone()
for i in range(0, REPEAT_TIMES-1):
    configuration = torch.cat([configuration, old_configuration], dim=0)
old_configuration = configuration.clone()
for i in range(0, REPEAT_TIMES-1):
    configuration = torch.cat([configuration, old_configuration], dim=1)

print(configuration.shape[0])

model = Metropolis(configuration,fn,move,2,1,273)

model.solve(1)

model.summary()