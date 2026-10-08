# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 12:43:41 2025

@author: mathf
"""

import numpy as np
import matplotlib.pyplot as plt

def is_function(fct):
    tf = type(fct)
    return tf==type(tanh) or tf == type(fakereartanhtype) or tf==type(reartanh)

def apply(functions, values, deriv=False):
    if is_function(functions):
        return functions(values, deriv)
    if len(functions) != len(values):
        raise ValueError("{0} {1} Les tableaux de fonctions et de valeurs doivent avoir la même longueur.".format(len(functions), len(values)))
    dec = [f(x, deriv) for f, x in zip(functions, values)]
    results = np.array(dec)
    return results

## fonction
def tanh(z, deriv=False):
    t = np.tanh(z)
    return 1 - t**2 if deriv else t

def cos(z,deriv=False):
    return np.sin(z) if deriv else np.cos(z)

def sign(z,deriv=True):
    return 0 if deriv else np.sign(z)

def arctan(z,deriv=False):
    t = np.arctan(z)
    return 1/(1+z**2) if deriv else t

def MexicanRELU(z, deriv=False):
    if deriv:
        return np.exp(-z**2) * (1 - 2*z**2)
    return z * np.exp(-z**2)

def relu(z, deriv=False):
    return (z >= 0).astype(float) if deriv else np.maximum(0, z)

def sigmoid(z, deriv=False):
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s) if deriv else s

def swish(z, deriv=False):
    s = 1/(1+np.exp(-z))
    if not deriv:
        return z*s
    return s + z*np.exp(-z)*s**2

def softmax(z, deriv=False):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True) if not deriv else None

def leaky_relu(z, deriv=False, alpha=0.01):
    return np.where(z > 0, 1, alpha) if deriv else np.where(z > 0, z, alpha * z)

def linear(z, deriv=False):
    return np.ones_like(z) if deriv else z

class FakeRearFunction():
    def __init__(self, function):
        self.function = function
        self.pitch = 0
    def set_fonction(self,*args,**kw):
        pass
    def delta(self,*args,**kw):
        pass
    def train(self,*a,**k):
        return None
    def gradient(self,x, deriv=False, h=0.001):
        return 0*x
    def __call__(self,*args,**kwargs):
        
        if len(args)>0 and type(args[0]) == type('hey'):
            if args[0] == 'train':
                return self.train
            elif args[0] == 'grad':
                return self.gradient
        return self.function(*args,**kwargs)
        
fakereartanhtype = FakeRearFunction(tanh)
class RearFunction():
    def __init__(self, *l_functions, pitch_init = 0, moy='moyenneAr'):
        L = []
        r = sum([abs(k[1]) for k in l_functions])
        for i in l_functions:
            L.append((i[0], abs(i[1])/r))
        self.listeFunctions = L
        self.pitch = pitch_init
        self.set_fonction()
        
        if len(l_functions)==1:
            self.function=l_functions[0][0]
            k = FakeRearFunction(self.function)
            self.__call__ = k.__call__
            self.set_fonction = k.set_fonction 
            self.delta = k.delta
            self.train = k.train
        
        if moy==None:
            self.moy = self.moyenneAr
        else:
            self.moy = eval('self.{0}'.format(moy))
            
    def moyenne(self, v0, fct1, v1, fct2, z, deriv):
        return self.moy(v0,fct1,v1,fct2,z,deriv)
    def moyenneAr(self, v0, fct1, v1, fct2, z, deriv):
        return v0*fct1(z,deriv)+v1*fct2(z,deriv)
    def moyenneGeo(self,v0,fct1,v1,fct2,z,deriv):
        return fct1(z,deriv)**v0*fct2(z,deriv)**v1
    def moyenneQuadra(self,v0,fct1,v1,fct2,z,deriv):
        return np.sqrt(v0*fct1(z,deriv)**2+v1*fct2(z,deriv)**2)
    def set_fonction(self, seuil_borne=10**-6, borneplus=True):
        p = self.pitch
        self.current_i = 0
        last_i = seuil_borne
        for k in range(len(self.listeFunctions)):
            i = self.listeFunctions[k] 
            if i[1]-p >= 0:
                v0 = -p/i[1]+1
                v1 = p/i[1]
                fct1 = i[0]
                fct2 = self.listeFunctions[(k+1)%len(self.listeFunctions)][0]
                def fct(z,deriv=False,v0=v0,v1=v1,fct1=fct1,fct2=fct2):
                    return self.moyenne(v0,fct1,v1,fct2,z,deriv)
                self.function = fct
                # Calcul des gradients et problème de discontinuité dans ce calcul
                self.current_i = (i[1],fct1,fct2)
                if borneplus and p <seuil_borne:
                    self.delta(-seuil_borne)
                    return self.set_fonction()
                if not borneplus and i[1]-p < seuil_borne:
                    self.delta(seuil_borne)
                    return self.set_fonction()
                return 0
            else:
                last_i = i[1]
                p -= i[1]
        self.pitch=0
        return self.set_fonction()
    
    def delta(self, D):
        self.pitch = (self.pitch+D)%1.0
        self.set_fonction()
        
    def gradient(self,x, deriv=False, h=0.00000001):
        g= self.gradient_bis(x,deriv,h)
        return list(g)
        
    def gradient_bis(self,x,deriv,h):
        v, fct1, fct2 = self.current_i 
        return (fct2(x,deriv)-fct1(x,deriv))/v
    
    def gradient_direct(self,x,deriv,h):
        h1 = self.function(x, deriv)
        self.delta(h)
        h2 = self.function(x,deriv)
        self.delta(-h)
        return (h2-h1)/h
        
    def train(self,dp, deriv=False):
        self.delta(dp)
    
    def __call__(self,*args,**kwargs):
        if len(args)>0 and type(args[0]) == type('hey'):
            if args[0] == 'train':
                return self.train
            elif args[0] == 'grad':
                return self.gradient
        return self.function(*args,**kwargs)
reartanh = RearFunction((tanh,1))

if True:
    r = RearFunction((linear,2),(tanh, 4),(sigmoid, 3), (linear,3))
    T = [x/50 for x in range(-100,100)]
    plt.figure(1)
    for i in range(50):
        X = [r(t) for t in T]
        r.delta(0.01)
        plt.plot(T,X)
    plt.show()
    
#cost functions

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def get_anti_overfitting_version_cost_function(function, limit = 0.01):
    def fct(*args, limit_cst=limit, fct_cost = function):
        return max(limit, fct_cost(*args))

mse_anti_overfitting_001 = get_anti_overfitting_version_cost_function(mse, 0.01)

## Layers
class Layer:
    def __init__(self, *dim, **args):
        self.dim = list(dim)
        self.args = dict(args)
        
class _FeedForwardLayer(Layer):
    def __init__(self,*dim,**args):
        Layer.__init__(self,*dim,**args)
        if 'activation' not in self.args.keys():  
            
            self.args['activation'] = np.empty(self.dim,dtype=object)
            self.args['activation'].fill(linear)
        self.bias = None
        if 'bias' not in args.keys():
            self.args['bias'] = True
        
class InitFeedForwardLayer(_FeedForwardLayer):
    def __init__(self, *dim, **args):
        _FeedForwardLayer.__init__(self,*dim,**args)
        
class FeedForwardLayer(_FeedForwardLayer):
    def __init__(self, *dim, **args):
        _FeedForwardLayer.__init__(self,*dim,**args)

## Network 
        
class NetworkLayer:
    def __init__(self, *liste):
        self.listeLayers = list(liste)
        for i in range(len(liste)):
            if i!=0:
                liste[i].previous_layer = liste[i-1]
            if i!=len(liste)-1:
                liste[i].next_layer = liste[i+1]
        self.init_weights()
    def init_weights(self):
        pass

class FeedForwardNetwork(NetworkLayer):
    def __init__(self,*liste, use_gear=True, interval_gear = None):
        NetworkLayer.__init__(self,*liste)
        self.use_gear = use_gear
        self.interval_gear = interval_gear
    def init_weights(self):
        
        self.listeWeights = []
        self.listeActivations = []
        self.listeBias = []
        for i in range(len(self.listeLayers)):
            current_layer = self.listeLayers[i]
            if i!=0:
                previous_layer = self.listeLayers[i-1]
                W = self.rule_init_weights(np.zeros(previous_layer.dim+current_layer.dim).T)
                current_layer.previous_weights = W
                self.listeWeights.append(W)
            #traitement act
            act = current_layer.args['activation'].copy()
            if True:
                for _ in range(len(act)):
                    a = act[_]
                    try:
                        a.pitch 
                    except:
                        a = FakeRearFunction(a)
                    act[_] = a
                current_layer.args['activation'] = act
            self.listeActivations.append(act)
            #traitement bias
            if current_layer.bias == None:
                current_layer.bias = np.zeros(current_layer.dim)
                v = tuple(list(current_layer.bias.shape)+[1])
                current_layer.bias = current_layer.bias.reshape(v)
            self.listeBias.append(current_layer.bias)
        self.listeWeights = [np.eye(self.listeLayers[0].dim[0])] + self.listeWeights
    
    def rule_init_weights(self,matrice):
        return matrice
    def forward_propagation_process_complete(self, vector_0):
        vectors = [vector_0]
        onevalue = (vector_0.shape==tuple(self.listeLayers[0].dim))
        values = []
        for i in range(len(self.listeWeights)):
            w = self.listeWeights[i]
            value = np.dot(w, vectors[-1])+self.listeBias[i]
            values.append(value)
            vector_i = apply(self.listeActivations[i], value)
            vectors.append(vector_i)
        return vectors, values
    def forward_propagation_process(self,vector_0):
        return self.forward_propagation_process_complete(vector_0)[0]
    def forward_propagation(self, vector_0):
        return self.forward_propagation_process(vector_0)[-1]
    
    # Training part
    
    #Backward_propagation_models
    def gradient_descent_model(self, difference, activation, vec, value, weights=None, bias=None, alpha = 0.01, epaisseur = 1, profondeur_time = 0):
        delta_a = difference*apply(activation, value, True)
        return -alpha*delta_a.dot(vec.T), -alpha*delta_a.mean(1).reshape(-1,1)
    
    def gradient_descent_model_pitch(self,difference, activation, vec, value, weights=None, bias=None, alpha = 0.001, epaisseur = 1, profondeur_time = 0):
        #return self.gradient_descent_model( difference, activation, vec, value, weights=None, bias=None, alpha = 0.01, epaisseur = 1, profondeur_time = 0)[0]
        grad_pitch = apply(activation, ['grad' for _ in range(len(activation))])
        gp =difference*apply(grad_pitch,value)
        gp = gp.mean(1).reshape(-1,1)
        return alpha*gp
    
    def backward_propagation(self, difference,vecs, vals, model=None, model_pitch=None):
        if model==None:
            model = self.gradient_descent_model
        if model_pitch == None:
            model_pitch = self.gradient_descent_model_pitch
        Weights = self.listeWeights.copy()
        Bias = self.listeBias.copy()
        Activations = self.listeActivations.copy()
        
        deltaW, deltab, deltapitch = [], [],[]
        for i in range(len(Weights)-1,-1,-1):
            dW, db = model(difference, Activations[i], vecs[i], vals[i], Weights[i], Bias[i], epaisseur=len(Weights), profondeur_time=len(Weights)-i)
            deltaW.append(dW)
            deltab.append(db)
            if self.use_gear:
                dp = model_pitch(difference, Activations[i], vecs[i], vals[i], Weights[i], Bias[i], epaisseur=len(Weights), profondeur_time=len(Weights)-i)
                deltapitch.append(dp)
        
        deltab, deltaW, deltapitch = deltab[::-1], deltaW[::-1], deltapitch[::-1]
        
        for i in range(len(deltaW)):
            self.listeWeights[i] += deltaW[i]
            self.listeBias[i] += deltab[i]
            if self.use_gear:
                l_act = self.listeActivations[i]
                st = apply(l_act, ['train' for _ in range(len(l_act))])
                apply(st, deltapitch[i])
    
    def train(self, training_set_in, training_set_out, cost_function, iterations = 10,gradient_function = None, gradient_function_pitch=None, validation_set_in = None, validation_set_out = None, interval_gear=None):
        self.interval_gear = interval_gear
        couts_train = []
        _valbool = validation_set_in != None and validation_set_out != None
        couts_val = []
        for iteration in range(iterations):
            if self.interval_gear != None:
                self.use_gear = self.interval_gear[0] <= iteration <= self.interval_gear[1]
            
            #Training
            vecs,vals = self.forward_propagation_process_complete(training_set_in)
            y_train = vecs[-1]
            couts_train.append(cost_function(y_train, training_set_out))
            if _valbool:
                couts_val.append(cost_function(self.forward_propagation(validation_set_in), validation_set_out))
            self.backward_propagation((y_train-training_set_out),vecs=vecs, vals=vals, model=gradient_function, model_pitch=gradient_function_pitch)
        couts_train.append(cost_function(y_train, training_set_out))
        if _valbool:
            couts_val.append(cost_function(self.forward_propagation(validation_set_in), validation_set_out))
        return couts_train, couts_val
    
    def train_and_plot(self,*args, **kwargs):
        va = self.train(*args, **kwargs)
        couts_train, couts_val = va
        
        plt.plot([t+1 for t in range(len(couts_train))], couts_train, label="Training")
        plt.plot([t+1 for t in range(len(couts_val))], couts_val, label="Validation")
        plt.legend()
        plt.show()
        
        return va
        
if __name__ == '__main__':
    
    def entrainement(training_set_in, training_set_out, mdl_de=0.01, mdl_pitch=0.01):
        # Entrainement O
        f1 = InitFeedForwardLayer(2, activation = [RearFunction((relu,1),(leaky_relu,1)), RearFunction((relu,1),(leaky_relu,1))])
        f2 = FeedForwardLayer(3,activation = [RearFunction((tanh,1),(sigmoid,1), (relu,1)), RearFunction((relu,1),(tanh,1),(sigmoid,1)), RearFunction((sigmoid,1),(relu,1),(tanh,1))])
        f3 = FeedForwardLayer(1,activation = [tanh])
        FFN = FeedForwardNetwork(f1,f2,f3, use_gear=False)
        print("Out : ", FFN.forward_propagation(np.array([[1,1]]).T))
        
        
        print('Training')
        
        def mdl(*args, **kwargs):
            kwargs['alpha'] = mdl_de
            return FFN.gradient_descent_model(*args,**kwargs)
        
        def mdl_pi(*args,**kwargs):
            kwargs['alpha'] = mdl_pitch
            return FFN.gradient_descent_model_pitch(*args,**kwargs)
        
        ct, cv = FFN.train(training_set_in, training_set_out, mse, iterations=3500, gradient_function=mdl, gradient_function_pitch=mdl_pi)
        print("Out : ", FFN.forward_propagation(np.array([[0,0]]).T))
        print("Out : ", FFN.forward_propagation(np.array([[0,1]]).T))
        print("Out : ", FFN.forward_propagation(np.array([[1,0]]).T))
        print("Out : ", FFN.forward_propagation(np.array([[1,1]]).T))
        
        #Entrainement 1
        f1 = InitFeedForwardLayer(2, activation = [RearFunction((relu,1),(leaky_relu,1)), RearFunction((relu,1),(leaky_relu,1))])
        f2 = FeedForwardLayer(3,activation = [RearFunction((tanh,1),(sigmoid,1), (relu,1)), RearFunction((relu,1),(tanh,1),(sigmoid,1)), RearFunction((sigmoid,1),(relu,1),(tanh,1))])
        f3 = FeedForwardLayer(1,activation = [tanh])
        FFN1 = FeedForwardNetwork(f1,f2,f3)
    
        ct1,cv1 = FFN1.train(training_set_in, training_set_out,mse, iterations=3500, gradient_function=mdl, gradient_function_pitch=mdl_pi)    
        
        #Entrainement 2
        f1 = InitFeedForwardLayer(2, activation = [RearFunction((relu,1),(leaky_relu,1)), RearFunction((relu,1),(leaky_relu,1))])
        f2 = FeedForwardLayer(3,activation = [RearFunction((tanh,1),(sigmoid,1), (relu,1)), RearFunction((relu,1),(tanh,1),(sigmoid,1)), RearFunction((sigmoid,1),(relu,1),(tanh,1))])
        f3 = FeedForwardLayer(1,activation = [tanh])
        FFN2 = FeedForwardNetwork(f1,f2,f3, use_gear=False)
    
        ct2,cv2 = FFN2.train(training_set_in, training_set_out,mse, iterations=3500, gradient_function=mdl,gradient_function_pitch=mdl_pi, interval_gear=[500,600])    
        
        print('')
        print('Pitchs FFN1:')
        print(FFN1.listeActivations[0][0].pitch)
        print(FFN1.listeActivations[0][1].pitch)
        print(FFN1.listeActivations[1][0].pitch)
        print(FFN1.listeActivations[1][1].pitch)
        print(FFN1.listeActivations[1][2].pitch)
        print('')
        print('')
        print('Pitchs FFN2:')
        print(FFN2.listeActivations[0][0].pitch)
        print(FFN2.listeActivations[0][1].pitch)
        print(FFN2.listeActivations[1][0].pitch)
        print(FFN2.listeActivations[1][1].pitch)
        print(FFN2.listeActivations[1][2].pitch)
        print('')
        

        plt.figure(0)
        for ct,cv,label in [(ct,cv,'No rear'),(ct1,cv1,'Rears'),(ct2,cv2,'Rears timed')]:
            plt.plot([t for t in range(len(ct))], [np.log(c) for c in ct], label=label)
        plt.legend()
        plt.show()
        
    training_set_in = np.array([[0,0],[1,0],[0,1],[1,1]]).T
    training_set_out = np.array([-1,-1,-1,1]).T
    
    entrainement(training_set_in, training_set_out, mdl_pitch=0.0001)
    
    training_set_in = np.array([[x/100,y/100] for x in range(100) for y in range(10)]).T
    training_set_out = np.array([(x+y) for (x,y) in training_set_in.T]).T
    
    bruit = np.random.normal(0,0.1,training_set_out.shape)
    training_set_out += bruit
    
    entrainement(training_set_in, training_set_out, mdl_pitch = 0.0001, mdl_de=10**-5)
