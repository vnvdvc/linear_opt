import scipy,time,sys
from scipy.integrate import quad
import numpy as np
from scipy.stats import norm
#global parameters&variables
#cutoff is the highest order of hermite functions used as basis functions, so the matrix has rank cutoff+1

#return the basis function divided by (1/2pi)**0.5*exp(-1/4*x**2)
def basis(n):
    pref = 1/((2**n)*np.math.factorial(n))**0.5
    hnz = scipy.special.hermite(n).c
    xn = np.zeros(n+1)
    for order in range(n+1):
        xn[order] = (np.sqrt(2))**(-n+order)
    return pref*np.poly1d(hnz*xn)

# 1st argument is np.poly1d, the polynomial is in the order of (x-x0)
# turn np.poly1d into a real polynomial function
def polynomial(x,poly,x0=0):
    coeffs = poly.c
    rank = len(coeffs)
    results = 0
    for order in range(rank):
        results += coeffs[order] * (x-x0)**(rank-order-1)
    return results

# integrate two basis functions, or derivatives of basis functions
def basis_integral(fun1,fun2):
    def __poly(x):
        return polynomial(x,fun1*fun2,x0=0)
    distribution = norm()
    expectation = distribution.expect(__poly)
    if expectation < accuracy:
        return 0.0
    else:
        return expectation

# return the linear coefficients of fun as expansion of basis functions,
# the fun has to have the same exponent as the basis function
def linear_expansion(fun,cutoff):
    coeffs = []
    for n in range(cutoff+1):
        coeffs.append(basis_integral(fun,basis(n)))
    return np.array(coeffs)

# calculate the nth order derivative of a gaussian with center=0, and sigma
# return the polynomial of the derivative
def derivative_of_gaussian(sigma,n):
    hnz = scipy.special.hermite(n).c
    xn = np.zeros(n+1)
    for order in range(n+1):
        xn[order] = (np.sqrt(2)*sigma)**(-n+order)
    return np.poly1d((-1)**n*(np.sqrt(2)*sigma)**(-n)*(hnz*xn))
    
def liouville(tauD,cutoff):
    harmonic = np.poly1d([0.25,0.0,-0.5])
    matrix = np.zeros((cutoff+1,cutoff+1))
    for n in range(cutoff+1):
        base = basis(n)
        rhs = (-1)*np.polyder(base,m=2) + harmonic*base 
        rhs += (-2)*np.polyder(base,m=1)*derivative_of_gaussian(np.sqrt(2),1)
        rhs = rhs + (-1)*(derivative_of_gaussian(np.sqrt(2),2) * base)
        matrix[:,n] = linear_expansion(rhs,cutoff)
    return matrix/float(tauD)

# param_list is in the order of nu,alpha,xc
def fet(param_list,cutoff):
    if len(param_list) == 3:
        nu,alpha,xc = param_list
    else:
        raise ValueError("fet: Wrong number of parameters! Check whether the right model is used!")
    matrix = np.zeros((cutoff+1,cutoff+1))
    norm_const = nu*np.exp(-alpha*xc**2/(2*alpha+1))/np.sqrt(1+2*alpha)
    dist = norm(2*alpha*xc/(2*alpha+1),1/np.sqrt(2*alpha+1))
    for n in range(cutoff+1):
        rbase = basis(n)
        for m in range(cutoff+1):
            if m < n:
                continue
            lbase = basis(m)
            matrix[m,n] = dist.expect(lambda x:polynomial(x,poly=lbase*rbase,x0=0))
            if matrix[m,n] < accuracy:
                matrix[m,n] = 0.0
            if n != m:
                matrix[n,m] = matrix[m,n]
    return matrix*norm_const

#Return the bases of the model, and constant vectors, W0j and Ibj
class Initialize(object):
    def get_bases(self):
        bases = []
        for n in range(self.cutoff+1):
            bases.append(basis(n))
        return bases
    def get_W0j(self):
        W0j = np.zeros(self.cutoff+1)
        dist = norm()
        for n in range(self.cutoff+1):
            W0j[n] = 1/(2*np.pi)**0.25*dist.expect(lambda x: polynomial(x,self.bases[n],x0=-self.x0))
            if W0j[n] < accuracy:
                W0j[n] = 0.0
        return W0j
    def get_Ibj(self):
        Ibj = np.zeros(self.cutoff+1)
        dist = norm()
        for n in range(self.cutoff+1):
            Ibj[n] = (2*np.pi)**0.25*dist.expect(lambda x: polynomial(x,self.bases[n],x0=0))
            if Ibj[n] < accuracy:
                Ibj[n] = 0.0
        return Ibj
    def __init__(self,x0,cutoff):
        self.x0 = x0
        self.cutoff = cutoff
        self.bases = self.get_bases()
        self.W0j = self.get_W0j()
        self.Ibj = self.get_Ibj()

#param_list = [nu,alpha,xc] for fet
class Model(object):
    def get_qt(self):
        lams,eigVecs = scipy.linalg.eigh(self.h)
        coeff_vec = np.dot(self.W0j,eigVecs)*np.dot(self.Ibj,eigVecs)
        def qt(t):
            results = 0
            for idx,lam in enumerate(lams):
                if lam.imag != 0.0:
                    raise ValueError("Model.get_qt: Imaginary eigenvalue encountered!")
                results += coeff_vec[idx]*np.exp(-lam.real*t)
            return results/np.sum(coeff_vec)      
        return qt
    def get_loss(self,data):
        self.loss = np.sum((self.qt(data[:,0])-data[:,1])**2)
#idx in the restricted set will have zero value in the corresponding idx of grads
#For example, if the value of nu is fixed for this mutant, then restricted = [0]
    def get_grads(self,data,restricted=[]):
        times = data[:,0]
        grads = []
        temp = self.qt(times)-data[:,1]
        for idx in range(len(self.param_list)):
            if idx in restricted:
                grads.append(0.0)
            else:
                params = self.param_list
                dparam = params[idx]*epsilon
                params[idx] += dparam
                qtNew = Model(self.tauD,params,self.initialize,et_model=self.et_model).qt
                dqt = 2*(qtNew(times)-self.qt(times))/dparam
                grads.append(np.dot(temp,dqt))
        self.grads = np.array(grads)
    def __init__(self,taud,param_list,initialize,et_model=fet):
        self.tauD = taud
        self.param_list = param_list
        self.et_model = et_model
        self.initialize = initialize
        self.cutoff = initialize.cutoff
        self.h = liouville(taud,self.cutoff) + et_model(param_list,self.cutoff)
        self.W0j = initialize.W0j
        self.Ibj = initialize.Ibj
        self.qt = self.get_qt()

#param_lists is a 1-dimensional list with length 3*num_of_mutants
#param_lists are in the order of (nu,alpha,xc) for each mutant
#initParamValues are the initial values of param_lists
#bounds are the lower and upper bounds of param_lists, with lower = upper meaning the parameter is fixed
#datasets: real data for all mutants
#initialization: list of class Initialize, [Initialize(+),Initialize(-)]
#x0s for mutants.
class Opt(object):
    def __init__(self,taud,initParamValues,bounds,datasets,initialization,x0s,et_model=fet):
        self.num_of_mutants = len(datasets)
        self.num_of_param_per_mutant = len(initParamValues)/self.num_of_mutants
        if len(initParamValues)%self.num_of_mutants != 0:
            raise ValueError("Opt.init: Length of param_lists,{} does not match number of datasets,{} !"
                             .format(len(param_lists),self.num_of_mutants))
        if len(bounds) != len(initParamValues):
            raise ValueError("Opt.init: Length of bounds,{} does not match number of parameters,{} !"
                            .format(len(bounds),len(initParamValues)))
        self.plusInit = initialization[0]
        self.minusInit = initialization[1]
        self.initValues = initParamValues
        self.datasets = datasets
        self.bounds = bounds
        self.tauD = taud
        self.et_model = et_model
        self.x0s = x0s
    def get_loss(self,param_lists):
        total = 0.0
        self.models = []
        param_num = self.num_of_param_per_mutant
        for idx in range(self.num_of_mutants):
            if self.x0s[idx] > 0:
                model = Model(self.tauD,param_lists[param_num*idx:param_num*idx+param_num],self.plusInit,et_model=self.et_model)
            else:
                model = Model(self.tauD,param_lists[param_num*idx:param_num*idx+param_num],self.minusInit,et_model=self.et_model)
            model.get_loss(self.datasets[idx])
            self.models.append(model)
            total += model.loss
        return total/self.num_of_mutants
#The gradient of nu of each mutant is the same, since they have the same value of nu
    def get_grads(self,param_lists):
        start = time.clock()
        param_num = self.num_of_param_per_mutant
        grads = np.zeros(param_num*self.num_of_mutants)
        dnu = 0.0
        for mutant_idx in range(self.num_of_mutants):
            restricted = []
            for idx in range(param_num):
                l,u = self.bounds[param_num*mutant_idx+idx]
                if l == u:
                    restricted.append(idx)
            self.models[mutant_idx].get_grads(self.datasets[mutant_idx],restricted)
            grads[param_num*mutant_idx:param_num*mutant_idx+param_num] = self.models[mutant_idx].grads
            dnu += self.models[mutant_idx].grads[0]
        dnu = dnu/self.num_of_mutants
        for mutant_idx in range(self.num_of_mutants):
            grads[param_num*mutant_idx] = dnu
        if len(grads) != len(param_lists):
            raise ValueError("Opt.get_grads: Length of grads,{} does not match number of parameters,{} !"
                            ).format(len(grads),len(param_lists))
        return np.array(grads)




def scenario(x0s,index):
    taud = 2.6
    num_of_mutants = 4
    initParams = np.zeros(3*num_of_mutants)
    bounds = [(0.01,3.76),(0.01,1.0),(0.001,10.0)]*num_of_mutants
    initialization = [Initialize(x0,cutoff),Initialize(-x0,cutoff)]
    threshold = 0.0001
    max_epoch = 5
    results = []
    loss = 100.0
    epoch = 0
    while loss > threshold and epoch < max_epoch:
        for idx in range(num_of_mutants):
            rands = np.random.random_sample((3,))
            if idx == 0:
                initNu = rands[0]
            initParams[3*idx:3*idx+3] = np.array([initNu,rands[1],rands[2]*5])
        if epoch == 0:
            if index == 0:
               initParams = np.array([0.9487,0.5159,0.6198,0.9487,0.97,0.92,0.95,0.55,1.26,0.95,0.23,1.86])
            if index == 5:
               initParams = np.array([1.51,1.00,1.07,1.51,1.00,0.70,1.51,0.17,2.96,1.51,0.60,1.57])
        opt = Opt(taud,initParams,bounds,datas,initialization,x0s)
        mins, loss, __ = scipy.optimize.fmin_l_bfgs_b(opt.get_loss,initParams,fprime=opt.get_grads,bounds=opt.bounds,factr=1e7,iprint=0)
        results.append((loss,mins))
        with open(folder+"results_{}_2nd.txt".format(index),"a+") as out:
            out.write("Initial condition x0s:{}\n".format(x0s))
            out.write("Parameters:{}\n".format(mins))
            out.write("loss:{}\n".format(loss))
        print "initial condition x0s:{}".format(x0s)
        print "Parameters:{}".format(mins)
        print "loss:{}".format(loss)
        epoch += 1
    return results

if __name__ == "__main__":
    accuracy = 1e-10
    epsilon = 1e-8
    #Implementation
    #Trial 1:loss = 0.264, nu=0.5420, alpha=0.01, xc=0.001,  x0=0.99 x0s=[++++]
    #Trial 2: loss=0.281 [0.517,0.01,0.001,0.517,0.01,2.73,0.517,0.01,0.001,0.517,0.01,0.001] x0=0.86 x0s=[++++] 
    #read data
    cutoff = 10
    folder = "./" 
    mutant_files = ["y98r.txt","y98h.txt","y98a.txt","y98f.txt"]
    datas = []
    for filename in mutant_files:
        with open(folder+filename,"r") as inp:
            data = []
            for line in inp:
                nums = line[:-1].split()
                data.append((float(nums[0]),float(nums[1])))
        datas.append(np.array(data))
    x0 = 0.86
    x0s = [x0]*4
    x0s_list = []
    for i in [+1,-1]:
        for j in [+1,-1]:
            for k in [+1,-1]:
                for l in [+1,-1]:
                    x0s_list.append([i*x0,j*x0,k*x0,l*x0])
    index = int(sys.argv[1])
    scenario(x0s_list[index],index)

