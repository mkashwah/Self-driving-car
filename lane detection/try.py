import numpy as np

# inpt_1 = input("inpot_1 : ")    #reading inputs
# # inpt_2 = raw_input("inpt_2 : ")    #reading inputs
#
# print (type(inpt_1))
# # print(type(inpt_2))
# # inpt_1 = int(inpt_1)
# print (type(int(inpt_1)))
# print (type(inpt_1))
##########################################################
# np.random.seed(0)
# synaptic_weights = 1* np.random.random((2,2))
#
# print ("random satrting synaptic weights")
# print(synaptic_weights)
##Classes

class TheClass():
    def __init__(self, var1, var2):
        self.vara = var1
        self.varb = var2
    def summ(self, var1, var2):
        return var1+var2


cc = TheClass(1,"abc")
a = cc.summ(5,6)

print(cc.vara)
print(cc.varb)
print(a)
