from network import MLP,Layer,Neuron
from ML import Value
#TMP here stands for Trained Model Parameters
TMP = []
with open("trained_model_parameters.txt","r") as f:
    for line in f:
        TMP.append((float(line[:-2])))
#TM stands for Trained Model
TM = MLP(4,[4,1])
TM.changeparam(TMP)
# print(TM.parameters()[0])
xs=[[2.0,3.0,0.0,1.0],
    [-1.0,0.0,2.0,4.0],
    [-2.0,1.0,4.0,6.0]
    ]
for i in xs:
    print(TM(i))