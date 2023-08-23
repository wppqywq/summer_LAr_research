import numpy as np
import time
try:
    import numba as nb
    have_numba=True
except:
    have_numba=False
    print("Numba unavailable - some things may be slower.")

def extract_vec(vec,jl,jr):
    if jl==0:
        return vec[:jr+1]
    else:
        v1=vec[-jl:]
        v2=vec[:jr+1]
        return np.hstack([v1,v2])
def insert_vec(vec,jl,jr,n):
    out=np.zeros(n)
    if jl==0:
        out[:jr+1]=vec
        return out
    else:
        out[-jl:]=vec[:jl]
        out[:jr+1]=vec[jl:]
        return out

    #@nb.njit(parallel=False)
def vec2dense_nb(vec,jl,jr,k1=10000,k2=-10000,mat=None,istart=0):
    """make a dense version of the FIR matrix for one vector.  Allocate a matrix
    or optionally take in a pre-existing matrix.  copies centered on k1:k2 with shifts
    going from -jl to jr will go into mat."""
    n=len(vec[k1:k2])
    if mat is None:
        mat=np.zeros((n,jl+jr+1))
        istart=0
    for i in nb.prange(0,jl+jr+1):
        mat[:,istart+i]=vec[k1+i-jl:k2+i-jl]
    return mat

@nb.njit(parallel=True)
def vec2dense_trans_nb(vec,jl,jr,k1,k2,mat,istart):
    for i in nb.prange(-jl,jr+1):
        mat[istart+i+jl,:]=vec[k1+i:k2+i]
        
def vec2dense_trans(vec,jl,jr,k1=10000,k2=-10000,mat=None,istart=0,use_nb=True):
    """make a dense version of the FIR matrix for one vector.  Allocate a matrix
    or optionally take in a pre-existing matrix.  copies centered on k1:k2 with shifts
    going from -jl to jr will go into mat."""
    n=len(vec[k1:k2])
    jtot=int(jl+jr+1)
    if mat is None:
        mat=np.zeros((jtot,n))
        istart=0
    #mat=np.zeros((jtot,n))
    if have_numba&use_nb:
        vec2dense_trans_nb(vec,jl,jr,k1,k2,mat,istart)
    else:
        for i in np.arange(jtot):
            mat[i,:]=vec[k1+i+jl:k2+i+jl]
            
    return mat


class AdjustFit:
    def __init__(self,pred,targ,thresh,order=3,use_thresh=None):
        self.order=order
        self.thresh=thresh
        if use_thresh is None:
            use_thresh=thresh
        self.use_thresh=use_thresh
        mask=pred>thresh
        self.coeffs=np.polyfit(pred[mask],targ[mask],order)
    def apply_adjust(self,pred):
        out=pred.copy()
        mask=pred>self.use_thresh
        tmp=pred[mask]
        out[mask]=np.polyval(self.coeffs,pred[mask])
        return out
class FIRmat:
    def __init__(self,vecs,jl,jr,k1,k2):
        if isinstance(vecs,list):
            self.vecs=vecs
        else:
            self.vecs=[vecs]
        if isinstance(jl,list):
            self.jl=jl
        else:
            self.jl=[jl]
        if isinstance(jr,list):
            self.jr=jr
        else:
            self.jr=[jr]
        self.k1=k1
        self.k2=k2
        if isinstance(vecs,list):
            self.nvec=len(vecs)
        else:
            self.nvec=1
        self.n=len(self.vecs[0][k1:k2])
        self.fts=None
        self.coeffs=None
        self.adjust=None
    def add_vec(self,vec,jl=None,jr=None):
        self.vecs.append(vec)
        if jl is None: #default to the previous jl/jr if none specified
            jl=self.jl[-1]
        if jr is None:
            jr=self.jr[-1]
        self.jl.append(jl)
        self.jr.append(jr)
        self.nvec=self.nvec+1
        
    def nrow(self):
        n=0
        for i in range(self.nvec):
           n=n+(self.jr[i]+self.jl[i]+1)
        return n
        
    def dense(self):
        nr=self.nrow()
        mat=np.empty([nr,self.n])
        istart=0
        for i in range(self.nvec):
            vec2dense_trans(self.vecs[i],self.jl[i],self.jr[i],self.k1,self.k2,mat,istart)
            istart=istart+(self.jr[i]+self.jl[i]+1)
        return mat
    def get_fts(self):
        self.fts=[None]*self.nvec
        for i in range(self.nvec):
            self.fts[i]=np.fft.rfft(self.vecs[i][self.k1:self.k2])
    def vecmult(self,vec):
        vecft=np.conj(np.fft.rfft(vec[self.k1:self.k2]))
        vals=[None]*self.nvec
        for i in range(self.nvec):
            tmp=np.fft.irfft(self.fts[i]*vecft)
            vals[i]=extract_vec(tmp,self.jl[i],self.jr[i])
        return np.hstack(vals)
    def sqr(self):
        tmparr = [[None for j in range(self.nvec)] for i in range(self.nvec)]
        for i in range(self.nvec):
            for j in range(self.nvec):
                tmp=np.fft.irfft(self.fts[i]*np.conj(self.fts[j]))
                block=np.empty([self.jl[i]+self.jr[i]+1,self.jl[j]+self.jr[j]+1])
                for ii_tmp in range(self.jl[i]+self.jr[i]+1):
                    ii=ii_tmp-self.jl[i]
                    #this loop could be a *lot* more efficient.
                    #it doesn't add to runtime, though so I'll be lazy and leave it.
                    for jj_tmp in range(self.jl[j]+self.jr[j]+1):
                        jj=jj_tmp-self.jl[j]
                        block[ii_tmp,jj_tmp]=tmp[ii-jj]
                tmparr[i][j]=block
                tmparr[j][i]=block.T
        tmp2=[None]*self.nvec
        for i in range(self.nvec):
            tmp2[i]=np.hstack(tmparr[i])        
        return np.vstack(tmp2)
    def get_coeffs(self,s):
        lhs=self.sqr()
        rhs=self.vecmult(s)
        self.coeffs=np.linalg.inv(lhs)@rhs

    def set_adjust(self,targ,thresh,adjust=AdjustFit,order=None):
        if order is None:
            order=len(self.vecs)-1
        self.adjust=None
        pred=self.get_pred(adjust=False)
        self.adjust=adjust(pred,targ,thresh,order)
    def get_pred(self,coeffs=None,adjust=False):
        if coeffs is None:
            coeffs=self.coeffs
        pred=0.0
        icur=0
        for i in range(self.nvec):
            nn=self.jl[i]+self.jr[i]+1
            tmp=insert_vec(coeffs[icur:icur+nn],self.jl[i],self.jr[i],self.n)
            icur=icur+nn
            tmpft=np.fft.rfft(tmp)
            pred=pred+np.fft.irfft(self.fts[i]*np.conj(tmpft))
        #print('=====pred len:    ', len(pred))
        if adjust:
            if self.adjust is None:
                print('adjusted fit requested, but adjust has not been set. skipping.')
            else:
                pred=self.adjust.apply_adjust(pred)
        pout=np.zeros(len(self.vecs[0]))

        pout[self.k1:self.k2]=pred
        return pout
                    
def read_dir(dirname,subdir='0_digitization'):
    adc=np.loadtxt(dirname+'/'+subdir+'/digits_out_sequence_eT.txt')
    pileup=np.loadtxt(dirname+'/'+subdir+'/hit_eT_bck_sequence.txt')
    sig=np.loadtxt(dirname+'/'+subdir+'/hit_eT_sig_sequence.txt')
    pulse=np.loadtxt(dirname+'/'+subdir+'/ideal_output_sequence.txt')
    return adc,pileup,sig,pulse

