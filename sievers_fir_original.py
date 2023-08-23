import numpy as np
from scipy.linalg.blas import zgemm
from matplotlib import pyplot as plt
import numba as nb
import time

plt.ion()

@nb.njit(parallel=True)
def apply_filt_quadratic(adc,nleft,nright,coeffs):
    out=0*adc
    offset=coeffs[nleft+nright+1]
    linear=coeffs[:nleft+nright+1]
    quadratic=coeffs[nleft+nright+2:]    
    for i in nb.prange(nleft,len(adc)-nright-1):
        tmp=offset
        for j in range(nleft+nright+1):
            val=adc[i-nleft+j]
            tmp=tmp+linear[j]*val+quadratic[j]*val**2
        out[i]=tmp
    return out



pulse=np.loadtxt('data/pule_shape.txt')
adc=np.loadtxt('data/adc_out.txt')
signal=np.loadtxt('data/signal.txt')
pileup=np.loadtxt('data/pileup.txt')
wiener=np.loadtxt('data/wiener_out.txt')
npad=len(adc)-len(pileup)

signal=np.hstack([signal,np.zeros(npad)])
pileup=np.hstack([pileup,np.zeros(npad)])
pp=0*pileup
pp[:len(pulse)]=pulse
pred=np.fft.irfft(np.fft.rfft(pp)*np.fft.rfft(pileup+signal))
i1=30000
i2=31300
if False:
    plt.clf()
    plt.plot(pred[i1:i2])
    plt.plot(adc[i1:i2])
    plt.show()

nleft=25  #how many samples to the left of our data point we are allowed to use
lag=8 #how many samples to the right of our sample we are allowed to use
nright=lag+1 #offset by 1 is because of python indexing
nn=nleft+nright+2
mat=np.zeros([len(adc),nn])
ncut=10000 #snip this many samples off the ends to avoid edge effects in simulations.  
mm=np.zeros([len(adc)-2*ncut,nn])
for i in range(nn-1):
    mm[:,i]=adc[ncut-nleft+i:-ncut-nleft+i]
mm[:,-1]=1 #add a constant offset as a parameter, since samples were not zero mean

#mm_big=np.hstack([mm,mm[:,:-1]**2,mm[:,:-1]**3]) #let us use the square of the ADC samples
mm_big=np.hstack([mm,mm[:,:-1]**2,mm[:,:-1]**3]) #let us use the square and the cube of the ADC samples
if False:  
    #you can do a bit better, but only a bit, if you look at correlations of adc samples with their neighbors.
    #so, we will skip it.  Depending on how much you want to squeeze out, you could play with this and code
    #will work correctly
    nneighb=2
    for ii in range(1,nneighb+1):
        tmp=np.zeros([mm.shape[0],nn-ii-1])
        for j in range(nn-ii-1):
            fwee=np.roll(adc,j-nleft)*np.roll(adc,j-nleft-ii)
            tmp[:,j]=fwee[ncut:-ncut]
        mm_big=np.hstack([mm,tmp])

#do the least squares fits, both for standard FIR, and using ADC squared as a value
ss=(signal+pileup)[ncut:-ncut]
lhs=mm.T@mm
rhs=mm.T@ss
coeffs=np.linalg.inv(lhs)@rhs
pred1=mm@coeffs
print('linear rms is ',np.std((ss-pred1)[ncut:-ncut]))

lhs2=mm_big.T@mm_big
rhs2=mm_big.T@ss
coeffs2=np.linalg.inv(lhs2)@rhs2
pred2=mm_big@coeffs2
print('nonlinear rms is ',np.std((ss-pred2)[ncut:-ncut]))

plt.figure(1)
plt.clf()
plt.plot(ss[i1:i2],'o')
plt.plot(pred1[i1:i2],'.')
plt.plot(pred2[i1:i2],'.')
plt.legend(['data','linear','nonlinear'])
plt.title('Reconstruction w/ Matrix Multiply')
plt.show()
plt.savefig('nonlinear_reconstruction.png')

#plot the residuals with true signal on x-axis.  We *hope* that residual will not really
#depend on true signal, since that means our reconstruction where we care about it is
#unbiased.
i2=i2+20000
plt.figure(3)
plt.clf()
plt.plot(ss[i1:i2],ss[i1:i2]-pred1[i1:i2],'.')
plt.plot(ss[i1:i2],ss[i1:i2]-pred2[i1:i2],'.')
plt.title('residuals, plotted against true signal value')
plt.legend(['linear','nonlinear'])
plt.show()
plt.savefig('nonlinear_residuals.png')
           
if False:  #this repeats the reconstruction but explicitly as a FIR filter
    #if you don't want to import numba, set this to False and trues that I haven't lied.
    #I also didn't code up the cubic FIR, so this will only work if you set to quadratic

    #first do a short call so numba compiles and we get valid timing
    out=apply_filt_quadratic(adc[:1000],nleft,nright,coeffs2)
    for i in range(1): 
        t1=time.time()
        out=apply_filt_quadratic(adc,nleft,nright,coeffs2)
        t2=time.time()
        print('reconstructed series in ',t2-t1)

    truth=signal+pileup
    plt.figure(2)
    plt.clf()
    plt.plot(truth[i1+ncut:i2+ncut],'.')
    plt.plot(out[i1+ncut:i2+ncut],'.')
    plt.title('Reconstruction w/ FIR')
    plt.show()

    print('sanity check - scatter in matrix vs. FIR (should be zero): ',np.std(out[i1+ncut:i2+ncut]-pred2[i1:i2]))
