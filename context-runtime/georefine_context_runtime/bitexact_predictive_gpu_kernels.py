import triton
import triton.language as tl

@triton.jit
def _rans(state, ip, stream, freq, cum, context, active, nbytes,
          VOCAB:tl.constexpr, LOGV:tl.constexpr, BLOCK:tl.constexpr):
    slot=(state&65535).to(tl.int32); lo=tl.full((BLOCK,),0,tl.int32); hi=tl.full((BLOCK,),VOCAB,tl.int32)
    for _ in range(LOGV+1):
        mid=(lo+hi)//2; searching=lo<hi
        boundary=tl.load(cum+context*(VOCAB+1)+mid,active&searching,0)
        go=boundary<=slot; lo=tl.where(searching&go,mid+1,lo); hi=tl.where(searching&~go,mid,hi)
    sym=lo-1; valid=(sym>=0)&(sym<VOCAB)
    f=tl.load(freq+context*VOCAB+sym,active&valid,0).to(tl.uint32)
    c=tl.load(cum+context*(VOCAB+1)+sym,active&valid,0).to(tl.uint32)
    bad=active&(~valid|(f==0))
    state=tl.where(active,f*(state>>16)+slot.to(tl.uint32)-c,state)
    for _ in range(3):
        need=active&(state<(1<<23)); avail=ip<nbytes
        byte=tl.load(stream+ip,need&avail,0).to(tl.uint32); bad=bad|(need&~avail)
        state=tl.where(need,(state<<8)|byte,state); ip+=need.to(tl.int64)
    return sym,state,ip,bad|(active&(state<(1<<23)))

@triton.jit
def decode_words(main_stream,main_states,main_offsets,main_freq,main_cum,
                 res_stream,res_states,res_offsets,res_freq,res_cum,
                 labels,flips,thresholds,reference,out,error,n,cols,main_bytes,res_bytes,
                 MODE:tl.constexpr,STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    lane=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); active_lane=lane<(n+STRIDE-1)//STRIDE
    block=lane.to(tl.int64); start=(block*STRIDE).to(tl.int64); row=(start//cols).to(tl.int64); col=(start%cols).to(tl.int64)
    state=tl.load(main_states+block,active_lane,1<<23).to(tl.uint32); ip=tl.load(main_offsets+block,active_lane,4).to(tl.int64)
    if MODE==1:
        rs=tl.load(res_states+block,active_lane,1<<23).to(tl.uint32); rip=tl.load(res_offsets+block,active_lane,4).to(tl.int64)
    badall=tl.full((BLOCK,),False,tl.int1)
    for step in range(STRIDE):
        ix=start+step; active=active_lane&(ix<n)
        word=tl.load(reference+ix,active,0).to(tl.uint32); flip=tl.load(flips+row,active,0).to(tl.uint32)
        aligned=word^(flip<<15); key=tl.where((aligned&32768)!=0,~aligned,aligned^32768).to(tl.uint16).to(tl.int32)
        lo=tl.full((BLOCK,),0,tl.int32); hi=tl.full((BLOCK,),15,tl.int32)
        for _ in range(4):
            mid=(lo+hi)//2; th=tl.load(thresholds+mid,active&(lo<hi),0).to(tl.int32)
            go=key>=th; lo=tl.where((lo<hi)&go,mid+1,lo); hi=tl.where((lo<hi)&~go,mid,hi)
        binid=lo; lab=tl.load(labels+row,active,0).to(tl.int32); ctx=lab*16+binid
        sym,state,ip,bad=_rans(state,ip,main_stream,main_freq,main_cum,ctx,active,main_bytes,2048,11,BLOCK); badall|=bad
        if MODE==0:
            bit=ix*5; by=bit//8; a=tl.load(res_stream+by,active&(by<res_bytes),0).to(tl.uint32); z=tl.load(res_stream+by+1,active&(by+1<res_bytes),0).to(tl.uint32)
            residual=(((a<<8)|z)>>(11-bit%8))&31
        else:
            residual,rs,rip,bad=_rans(rs,rip,res_stream,res_freq,res_cum,sym,active,res_bytes,32,6,BLOCK); badall|=bad
        value=((sym&1024)<<5)|(((sym>>2)&255)<<7)|((sym&3)<<5)|residual
        tl.store(out+lane*STRIDE+step,value.to(tl.uint16),active)
        col+=1; cross=col>=cols; row+=cross.to(tl.int64); col=tl.where(cross,0,col)
    has=(block+1)*STRIDE<n
    ns=tl.load(main_states+block+1,active_lane&has,1<<23).to(tl.uint32); no=tl.load(main_offsets+block+1,active_lane&has,main_bytes).to(tl.int64)
    badall|=active_lane&((state!=ns)|(ip!=no))
    if MODE==1:
        ns=tl.load(res_states+block+1,active_lane&has,1<<23).to(tl.uint32); no=tl.load(res_offsets+block+1,active_lane&has,res_bytes).to(tl.int64)
        badall|=active_lane&((rs!=ns)|(rip!=no))
    if tl.sum(badall.to(tl.int32),0)>0: tl.atomic_or(error,1)
