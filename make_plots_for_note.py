import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
plt.ion()

def make_convmat(n,pulse):
    nn=len(pulse)
    mat=np.zeros([n,n+nn])
    for i in range(n):
        mat[i,i:i+nn]=pulse
    return mat



pulse=np.loadtxt('pule_shape.txt')

pp=np.zeros(1000)
pp[:len(pulse)]=pulse
ppft=np.fft.rfft(pp)
plt.clf()
plt.loglog(np.abs(ppft[1:]))
plt.xlabel('k')
plt.ylabel('Pulse FT Amplitude')
plt.savefig('pulse_ft.png')


adc=np.loadtxt('adc_out.txt')
signal=np.loadtxt('signal.txt')
pileup=np.loadtxt('pileup.txt')
npad=len(adc)-len(pileup)
signal=np.hstack([signal,np.zeros(npad)])
pileup=np.hstack([pileup,np.zeros(npad)])

pp=0*pileup
pp[:len(pulse)]=pulse

pred=np.fft.irfft(np.fft.rfft(pp)*np.fft.rfft(pileup+signal))
mypars=np.polyfit(adc[10000:-10000],pred[10000:-10000],1)
print('before rescaling, scatter is ',np.std((adc-pred)[10000:-10000]))
adc=adc*mypars[0]
sig=np.std((adc-pred)[10000:-10000])
print('after rescaling, scatter is ',sig) #np.std((adc-pred)[10000:-10000]))
N=sig**2
ideal_err=np.sqrt(N/np.sum(pp**2))
print('ideal err is ',ideal_err)



convmat=make_convmat(500,pulse).T
lhs=convmat.T@convmat
err_mat=np.linalg.inv(lhs/N)
deconv_err=np.median(np.sqrt(np.diag(err_mat)))
print('expected deconvolution error is ',deconv_err)

pp=np.roll(pp,0)

i1=15000
i2=i1+200
lhs=pp@pp
rhs=np.fft.irfft(np.fft.rfft(adc)*np.conj(np.fft.rfft(pp)))
ofilt=rhs/lhs
oerr=signal+pileup-ofilt
print('optimal filter RMS ',np.std(oerr[10000:-10000]))
plt.clf()
plt.plot((signal+pileup)[i1:i2])
plt.plot(ofilt[i1:i2])
plt.xlabel('Samples after '+repr(i1))
plt.legend(['Signal','OF Reconstruction'])
plt.ylabel('Signal Amplitude')
plt.title('Optimal Filter Reconstruction')
plt.show()
plt.savefig('of_reconstruction.png')


deconv=np.fft.irfft(np.fft.rfft(adc)/np.fft.rfft(pp))
#plt.plot(deconv[i1:i2])
derr=signal+pileup-deconv
print('deconvolution RMS ',np.std(derr[10000:-10000]))
plt.clf()
plt.plot((signal+pileup)[i1:i2])
plt.plot(deconv[i1:i2])
plt.xlabel('Samples After '+repr(i1))
plt.ylabel('Signal Amplitude')
plt.title('Deconvolution Reconstruction')
plt.legend(['Signal','Deconvolution'])
plt.savefig('deconv_reconstruction.png')

maxent=np.load('maxent_fit_'+repr(i1)+'.npy')
plt.clf()
plt.plot((signal+pileup)[i1:i2])
plt.plot(maxent[:(i2-i1)])
enterr=(signal+pileup)[i1:i1+len(maxent)]-maxent

plt.clf()
plt.plot(ofilt[i1:i2])
plt.plot(deconv[i1:i2])
plt.plot(maxent[:(i2-i1)])
plt.plot((signal+pileup)[i1:i2])
plt.legend(['OF Reconstruction','Deconvolution','Bayes','Signal'])
plt.xlabel('Samples after '+repr(i1))
#plt.legend(['Signal','OF Reconstruction'])
plt.ylabel('Signal Amplitude')
plt.savefig('reconstructions.png')

plt.clf()
plt.plot(oerr[i1:i2],'.')
plt.plot(derr[i1:i2],'.')
plt.plot(enterr[:i2-i1],'.')
plt.xlabel('Samples after '+repr(i1))
plt.ylabel('Error')
plt.legend(['OF Reconstruction','Deconvolution','Bayes'])
plt.title('Reconstruction Residuals')
plt.savefig('residuals.png')
plt.ylim([-2,2])
plt.title('Reconstruction Residuals Zoom')
plt.savefig('residuals_zoom.png')


jl=30
jr=8
k1=10000
k2=len(adc)-k1
D=np.zeros([k2-k1,jl+jr+2])
D[:,0]=1  #we'll add an overall offset as a fit parameter
for i in range(k1,k2):
    D[i-k1,1:]=adc[i-jl:i+jr+1]
s=(signal+pileup)[k1:k2]
c=np.linalg.inv(D.T@D)@(D.T@s)
pred=D@c
plt.clf();
plt.plot(pred[i1-k1:i2-k1]);
plt.plot(s[i1-k1:i2-k1]);
plt.plot(deconv[i1:i2])

plt.legend(['FIR ['+repr(jl)+','+repr(jr)+']','Signal','Deconvolution'])
plt.show()
plt.savefig('fir_'+repr(jl)+'_'+repr(jr)+'.png')
print('linear RMS is ',np.std(pred-s))

DD=np.hstack([D,D[:,1:]**2,D[:,1:]**3]);npow=3
cc=np.linalg.inv(DD.T@DD)@(DD.T@s)
pp=DD@cc
print('nonlinear RMS is ',np.std(pp-s))

nplot=15000
plt.clf()
plt.plot(s[k1:k1+nplot],(pred-s)[k1:k1+nplot],'.')
plt.plot(s[k1:k1+nplot],(pp-s)[k1:k1+nplot],'.')
plt.plot(s[k1:k1+nplot],deconv[2*k1:2*k1+nplot]-s[k1:k1+nplot],'.')
plt.legend(['Linear','Non-linear','Deconvolution'])
xx=plt.xlim()
plt.plot(xx,[0,0],'k')
plt.xlim(xx)
plt.xlabel('True Signal+Pileup')
plt.ylabel('Reconstruction Error')
plt.savefig('nl_fir_error_'+repr(jl)+'_'+repr(jr)+'.png')



#let's fit residuals to a stretch where the
#nonlinear seems to be missing for (25,6) filter
if False:
    smin=7
    smax=26
    mask=(pp>smin)&(pp<smax)
    fitp=np.polyfit(pp[mask],(pp-s)[mask],2)
    pp2=pp.copy()
    pp2[mask]=pp2[mask]-np.polyval(fitp,pp[mask])
    plt.plot(s[k1:k1+nplot],(pp2-s)[k1:k1+nplot],'.')



sthresh=10
mask=s>sthresh
D2=D[mask,:]
s2=s[mask]
c2=np.linalg.inv(D2.T@D2)@(D2.T@s2)
pred2=D@c2
plt.clf()

#plt.plot(s[k1:k1+nplot],(pred2-s)[k1:k1+nplot],'.')
#plt.plot(s[k1:k1+nplot],(pp-s)[k1:k1+nplot],'.')
#plt.plot(s[k1:k1+nplot],deconv[2*k1:2*k1+nplot]-s[k1:k1+nplot],'.')

plt.plot(pred2[k1:k1+nplot],(pred2-s)[k1:k1+nplot],'.')
plt.plot(pp[k1:k1+nplot],(pp-s)[k1:k1+nplot],'.')
plt.plot(deconv[2*k1:2*k1+nplot],deconv[2*k1:2*k1+nplot]-s[k1:k1+nplot],'.')

plt.legend(['Debiased Linear','Non-linear','Deconvolution'])
xx=plt.xlim()
plt.plot(xx,[0,0],'k')
plt.xlim(xx)
plt.xlabel('Recovered Signal+Pileup')
plt.ylabel('Reconstruction Error')
plt.savefig('debiased_fir_error_'+repr(jl)+'_'+repr(jr)+'.png')

yl=plt.ylim()


thresh=10
mask=pred2>thresh
best_pp=pp.copy() #the nonlinear FIR
best_pp[mask]=pred2[mask]
print('high-energy scatter: ',np.std((best_pp-s)[mask]))


plt.clf()

#plt.plot(s[k1:k1+nplot],(pred2-s)[k1:k1+nplot],'.')
#plt.plot(s[k1:k1+nplot],(pp-s)[k1:k1+nplot],'.')
#plt.plot(s[k1:k1+nplot],deconv[2*k1:2*k1+nplot]-s[k1:k1+nplot],'.')

plt.plot(s[k1:k1+nplot],(best_pp-s)[k1:k1+nplot],'.')

#plt.legend(['Debiased Linear','Non-linear','Deconvolution'])
xx=plt.xlim()
plt.plot(xx,[0,0],'k')
plt.xlim(xx)
plt.xlabel('True Signal+Pileup')
plt.ylabel('Reconstruction Error')
plt.ylim(yl)
plt.title('Hybrid Error, Thresh='+repr(thresh))
plt.savefig('hybrid_error_'+repr(jl)+'_'+repr(jr)+'.png')


#k1=500000-100
plt.clf()
nplot=300
fac=3
plt.plot(s[k1:k1+nplot])
plt.plot(fac*(best_pp-s)[k1:k1+nplot])
plt.legend(['True Signal+Pileup',repr(fac)+'*reconstruction Error'])
xx=plt.xlim()
plt.plot(xx,[0,0],'k')
plt.xlim(xx)


nn=(len(cc)-1)//npow
cc_rect=np.empty([nn,npow])
for i in range(npow):
    cc_rect[:,i]=cc[1+i*nn:1+(i+1)*nn]*(50**i)/2
plt.clf()
plt.plot(np.arange(jr+1),pulse[:jr+1],':')
pvec=0*s
pvec[:len(pulse)]=pulse
pft=np.fft.rfft(pvec)
pcoeffs=np.fft.irfft(1/np.conj(pft))
pcoeffs=pcoeffs/pcoeffs.max()
to_plot=np.zeros(jl+jr+1)
to_plot[:jl]=pcoeffs[-jl:]
to_plot[jl:]=pcoeffs[:jr+1]
plt.plot(np.arange(-jl,jr+1),to_plot)

plt.plot(np.arange(-jl,jr+1),cc_rect)



myleg=['Pulse Profile','Deconv']
for i in range(npow):
    myleg=myleg+['d**'+repr(i+1)+' coeffs']
plt.legend(myleg)
plt.title('Nonlinear FIR Coefficients')
plt.xlabel('dt (samples)')
plt.ylabel('FIR Coefficients (rescaled)')
plt.savefig('coeffs_'+repr(jl)+'_'+repr(jr)+'.png')



mask=s<sthresh
DD2=DD[mask,:]
ss=s[mask]
cc2=np.linalg.inv(DD2.T@DD2)@(DD2.T@ss)

pp2=DD@cc2
ptot=pred2
ptot[mask]=pp2[mask]
