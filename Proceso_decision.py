import numpy as np
from itertools import product
from random import choice
class MDP:
    
    def __init__(self,s,r,a,T,gamma):
        """
        Builds the MDP problem
        :param s: states
        :param r: rewards
        :param a: actions
        :param T: dictionary where keys are (s,a) pairs and
        values are probabilities
        :param gamma: the discount factor
        """
        self.s = s
        self.r = r
        self.a = a
        self.T = T
        self.gamma = gamma
        
    def policy_iteration(self,pi=None):
        """
            Policy iteration algorithm
        """
        #initial random policy
        if not pi:
            pi = [choice(self.a) for s in self.s]
        
        print('pi = '+str(pi))
        self.obtainT(pi);
        T = self.obtainT(pi)
        print('T(pi) = \n'+str(T))
        
        V = np.matmul(np.linalg.inv(np.eye(3)-self.gamma*T.T),self.r)
        print('V(s) = '+ str(V))
        
        pi_star = self.find_pi_star(V)
        print("pi*(s) = "+str(pi_star))
        
        if pi_star == pi:
            return pi
        else:
            return self.policy_iteration(pi_star)
        
    def obtainT(self,pi):
        """
        Obtains the transition probability matrix parametrized by the policy pi
        :param pi: the policy
        """
        return \
          np.matrix([[self.T[(s,t,pi[s])] for s in self.s] for t in self.s])
        
    def find_pi_star(self,V):
        """
        Finds the optimal policy for the given infinite horizon values
        :param V: the infinite horizon expected utility
        """
        
        print("\n".join([str((x,np.matmul(self.obtainT(x).T,np.array(V).T))) for x in product(self.a,self.a,self.a)]))
        
        return list(max(product(self.a,self.a,self.a),\
            key=lambda x: np.sum(np.matmul(self.obtainT(x).T,np.array(V).T))))

estados = [0,1,2]
print(estados)
acciones = [0,1]
print(acciones)
recompensas = [0,20,20]
print(recompensas)
gamma = 0.9
print(gamma)
T={
    (0,0,0):0.7,(0,0,1):0.5, (1,0,0):0.4,(1,0,1):0.2, (2,0,0):0.2,(2,0,1):0.1,
    (0,1,0):0.1,(0,1,1):0.3, (1,1,0):0.4,(1,1,1):0.7, (2,1,0):0.2,(2,1,1):0.1,
    (0,2,0):0.2,(0,2,1):0.2, (1,2,0):0.2,(1,2,1):0.1, (2,2,0):0.6,(2,2,1):0.8
}
print(T)

mdp = MDP(estados,recompensas,acciones,T,gamma)

pi_0 = [0,0,0]
print("T(pi_0) = \n"+str(mdp.obtainT(pi_0)))

politica = mdp.policy_iteration(pi_0)
print("Esto es lo que debes mandar:")

politica = mdp.policy_iteration([0,1,0])