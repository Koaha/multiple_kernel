import numpy as np

class sketch():

    def simplify_kernel(self,support_set=None,error_set=None,remainder_set=None,dimension = 3,debug = True):
        if debug:
            support_size = 7
            error_size = 4
            remainder_size = 10
            support_set = np.array([[77, 79, 77],
               [26, 84, 33],
               [67, 25, 94],
               [76, 58, 79],
               [49,  9, 60],
               [30, 69,  1],
               [88, 48,  4]]) # Ss
            error_set = np.array([[98, 73, 79],
               [93, 86, 90],
               [14, 21, 96],
               [71, 75, 48]])  # Se
            remainder_set = np.array([[35, 85, 25],
               [61, 52,  3],
               [30, 34, 72],
               [17, 62, 44],
               [ 4, 68, 17],
               [30, 44, 11],
               [27, 44, 54],
               [57, 28, 37],
               [16, 18,  6],
               [57, 99, 36]]) # Sr

        full_set = np.vstack((support_set,error_set,remainder_set)) # Sf
        Kff = full_set @ full_set.T
        Kss = support_set @ support_set.T
        Krr = remainder_set @ remainder_set.T
        Kee = error_set @ error_set.T
        Kes = error_set @ support_set.T
        Krs = remainder_set @ support_set.T

        Vs = np.vstack((Kss, np.zeros((error_size,support_size)), np.zeros((remainder_size,support_size))))
        Ve = np.vstack((np.zeros((support_size,error_size)),Kee, np.zeros((remainder_size, error_size))))
        Vr = np.vstack((np.zeros((support_size, remainder_size)), np.zeros((error_size,remainder_size)),Krr))

        Vff = np.hstack((Vs,Ve,Vr))
        Vsf = np.vstack((Kss, Kes, Krs))

    def compute_delta_K_alpha(self,sample= 10, dimension = 3,debug = True):
        if debug:
            data = np.array([[35, 85, 25],
                                      [61, 52, 3],
                                      [30, 34, 72],
                                      [17, 62, 44],
                                      [4, 68, 17],
                                      [30, 44, 11],
                                      [27, 44, 54],
                                      [57, 28, 37],
                                      [16, 18, 6],
                                      [57, 99, 36]])  # Sr
            alpha = np.random.rand(sample,1)
            c = np.array([4,1,5]).reshape(1,dimension)
            alpha_c = 0.4
        else:
            data = np.random.randint(1,100,size=(sample,dimension))
            alpha = np.random.rand(sample, 1)
            c = np.random.randint(1,100, size=(1,dimension))
            alpha_c = np.random.rand(1,1)

        K_before = data@data.T
        K_alpha_before = K_before @ alpha

        data_after = np.vstack((data,c))
        alpha_after = np.vstack((alpha,alpha_c))

        K_after = data_after@data_after.T
        K_alpha_after = K_after @ alpha_after

        K_before_extra_zero_column = np.hstack((K_before,np.zeros((sample,1))))
        K_before_extra_zero_column_row = np.vstack((K_before_extra_zero_column,np.zeros((1,sample+1))))
        alpha_extra_zero = np.vstack((alpha,0))
        K_alpha_before_extra = K_before_extra_zero_column_row@alpha_extra_zero

        direct_delta_K_alpha = K_alpha_after - K_alpha_before_extra

        K__c = data@c.T
        K_c_ = K__c.T
        K_c_c = c@c.T
        K__c_apha_c = K__c @ alpha_c.T
        indirect_delta_K_alpha = np.vstack((K__c_apha_c,K_c_@alpha+K_c_c@alpha_c.T))



if __name__ == "__main__":
    sketch.simplify_kernel()
