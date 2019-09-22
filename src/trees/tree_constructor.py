import sys
sys.path.append("..")
from src.kernels.kernels import *
from src.operators.operators import *
from src.utilities import instance_convert

class tree():

    def __init__(self,depth=2):
        self.OPERATION_LEMMAS  = [operator_affine,
                                  operator_multiplication,
                                  operator_polynomial,
                                  operator_exponential]
        self.SINGLE_OPERATORS = ['operator_polynomial',
                                 'operator_exponential']
        self.kernels = [Linear,
                        Polynomial,
                        RBF]
        self.constructed_kernels_dict = {}
        self.constructed_kernels_dict[1] = self.kernels
        self.depth = 1
        if depth > 1 :
            self.depth = int(depth)

    def construct_children(self,X):
        ls = []
        for Y in self.kernels:
            for operator in self.OPERATION_LEMMAS:
                ls.append(self.get_node(X,operator,Y))
        return ls

    def construct_tree(self,current_depth = 1):
        if current_depth < self.depth:
            ls=[]
            for X in self.constructed_kernels_dict.get(current_depth):
                ls+= self.construct_children(X)
            self.constructed_kernels_dict[current_depth+1] = ls
            self.construct_tree(current_depth+1)

    def get_node(self,X,fn,Y):
        k = Custom()
        if fn.__name__ not in self.SINGLE_OPERATORS:
            k.set_arguments(X,fn,Y)
        else:
            k.set_arguments(X,fn)
        # return k(X,fn,Y)
        return k

    def get_kernel_list(self):
        ls = []
        for key,value in self.constructed_kernels_dict.items():
            if key>0:
                ls+=value
        return [instance_convert(x) for x in ls]

