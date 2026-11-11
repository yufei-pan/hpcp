#!/usr/bin/env python3
# /// script
# requires-python = ">=3.6"
# dependencies = [
#     "xxhash; python_version >= '3.7'",
#     "multiCMD>=1.56",
# ]
# ///
# A Ctrl-C while Python imports this module: Python's own handler can raise it inside importlib's weak-reference callbacks,
# which drop it, and the run went on (a move to its removal). SIGINT is held in the importing thread for every import of
# the module (_signal: a builtin, loaded before any module runs), and the finally lets it in on every path, a failed import
# too, only where it was not blocked before (a forkserver or spawn pool worker imports hpcp with it blocked, until
# _init_worker). One that came meanwhile is raised there, at the finally. Below the header the module imports nothing more
# and runs no subprocess (_fat_ioctl_numbers).
import _signal
_IMPORT_MASK = _signal.pthread_sigmask(_signal.SIG_BLOCK, (_signal.SIGINT,)) if hasattr(_signal, 'pthread_sigmask') else None
try:
	import argparse
	import array
	import concurrent.futures
	import concurrent.futures.process
	import contextlib
	import errno
	import fnmatch
	import functools
	import glob
	import hashlib
	import itertools
	import locale
	import math
	import multiprocessing
	import multiprocessing.connection
	import multiprocessing.util
	import os
	import pathlib
	import platform
	import queue
	import random
	import re
	import shutil
	import signal
	import stat
	import struct
	import sys
	import tempfile
	import threading
	import time
	import unicodedata
	import uuid
	import weakref
	from collections import Counter, deque, namedtuple
	from concurrent.futures import ProcessPoolExecutor
	from concurrent.futures.process import BrokenProcessPool
	from math import log
	from pprint import pformat
	# (multiCMD's embed goes here with multiCMD 1.60, one tab in, so the mask covers it; until then it follows the finally,
	# its imports outside the mask)
	try:
		import xxhash  # type: ignore
		hasher = xxhash.xxh64()
		xxhash_available = True
	except ImportError:
		import hashlib
		hasher = hashlib.blake2b()
		xxhash_available = False
	try:
		import resource  # the open-files limit (below _PARENT_FLAGS)
	except ImportError:
		resource = None
finally:
	if _IMPORT_MASK is not None and _signal.SIGINT not in _IMPORT_MASK:
		_signal.pthread_sigmask(_signal.SIG_UNBLOCK, (_signal.SIGINT,))

try:
	import multiCMD  # type: ignore
	if tuple(int(x) for x in multiCMD.version.split('.')) < (1, 59):
		raise ImportError('multiCMD too old')
except Exception:
	import sys,types,base64,lzma   # noqa: E401
	multiCMD = types.ModuleType("multiCMD")
	sys.modules["multiCMD"] = multiCMD
	_SRC_B85=r'''
rAT%)=o\O*k16n/!WXAE('"=7F>)aNi:=rV**iOVLdFUh,;[5]%l=/aQgL]knEMJj2Td\)1[`bH(.0)JfPPs3eIh)]U<,WTZEDeSe:(YNb%NdN`;%MLm
ps(uYZLt9N^(d>j;?gF2fq.JS8E^d=!D`Sq$Gt&.k_&Jr(6<fc-."_L#%?6QoJ:i3GQHb>?A/=XqGZdH'Nu>j"g6i-cqUX1Al)9;U'hMS[PsbB-tdCiI
VVRN65*?GL<kIO7Z=mU4u4VKNr5CflUSB48`AZQ<O!o56.5)A@KI42W:)mWjP.7V"FYI88TLE(J'j#AVe8Eeh%CF,$8]XM46g9;,T/C$'$!;/WY^tg;T
%)kt>#SHo/!XLPa/XSKstQ>:g+Jr/ZquR/FXc]G[-+/8afANn_AP$KHf5"LG_IAO2cb6&`?)g?nA9]NDNU@b279H&#bIZc)hnE@N.%)Qj:?bT'+RlES'
)dIB8I!qbXLEp@iE!#0eM$@f"Zj!CtT&G&VIl)4TZZ+AAq&:IR[@iO:,*DR);QR1C[[i_bKWO$N3J9@Hu&A+^3FJqh1$hr4Vq8!iSAAoH^b:O<V0(RSZ
8q[X/@mLA!r%\rD`KI&9iAr])7>cqFP>m6fq[?dq?Pgi2q#h*q<D8l\.9^W,?fEBm2Z7EB+MH^f:;,meT6B+qgq"?\1\7e<J-e>NnaBAIB`gNU2mpA`F
%a&Br'i+0ZFD\u78;uC=B?$a?6%_"'#=j2-K6RR=mu?UERMoWM6kOIV=tN%CDGn++d+s6M6daN/(mkurXoP,F5T,(6LCgh?>*bciUUUB)h#=Y\H,>N=.
bFV_i@!NPM.ODgA&DTaQ$e<[eCO9_(LYg".rfE\7nYm[%OYK$XC'9aL[O;g.eCZ_$kuo97[t23U(lC'I\7k_'e?>T3g(n$[!F!=(A22+[gG'k8fJrNp?
);CY8$2hls5697k,8$fghWV:op^XtoLdr:#V>46[k:hKRL^4j.CmbqlXgI<&^^!Q>47GleEg-cf<H'j/%8M,;#A%A91[*Qg7`TQl!hoZW-PH@[8MiTJ<
L[<,k;h]O7Z@fRD[K-Hn"#l]>^W,o_8?W4Up`mZ+b_Lf8T:*G9Doau;%FeUg?NJP[Q&oIfEEQt1)=oGFbPhaXOpoFp@Q6@&HB%t.2986QgG,rHQcS_.E
UZ@(ZLjd>q9H9NS0JWb@)K%Y'.sc'$kOu=n6Z+Nuo];Bm,T/2i",8K(q3_8<=8+egfgfoe2DCMNNR3ju4)'?QUpXrn+%J:Or2\KE_)A!I;94U)fE2[V:
q'&lHI>=i;us]@R?9SO9-Gn0Yl(K]G8!3AS6^>V]:rIML-/qc_3QbhEQ*c2M"h+jU=hJujEtEQkp_XGWeeVaqDIjk(B4K1k($mZb)3fK^kLs4[a#;XO@
GuW>k66DN#Y=hYOKeg`UD&Who+n1EeV]n-3.7nJXV4$B#0*H!J]$D\[t1G(d3Ah,_J-\Nu9f!PI6Kt;GDiZqhZUalBO3OJod8Ke]Vrq]Yd'YfjGU*QH>
;ph:UA',pO?ZbI`.h'9/I;`:^."HD;4M:cXLP""Pk[Nu+es/<$glXNe^8Ip!+L;;U`.nY,PF$Pj]0gQKH0jFX$L;MKL/n=rEr7U`UhLA5Hh#u%6F1uc1
1fe$JjU1V;r),28+Sdd>]bV%^<mAM&acW0lI32t*@qN*0je@H[s2#]:V>1!^N!nK3@cX[S9:s^qmlP<&Q%_q_(g@X4AHnk>G41H^ES4@DIC$M^';4;'c
'Cdh;Qn9(",^fs`::uNfjV<YDiM754o3`ka^"olUG1qSA>H\>Di55%t=k$i:ZuVBnjo,OZD5'M5eJcXL;;u!<A1[Ep!D9s^IWaV5`-"#u2arNo=h!rkC
*Zh3J]b)9*4DLJP&ji7RSCqGc[^*Sl<$'-6<kNYf<(;Z$`m_n@%NR1K7"+=]u'](g(^+f2S#e!]oZl9D!I$[1KmfVDTr@'CSEc4m.Z]8+O3_Z-JYNidW
C0?:k%?S-aH]E0pTZ[oT-_r_0#W;!X.tbK3PK[)8OA+BF4Fo4;>tmidWEZI+7REnTdhI+Ph/uUpLfn\c#Tb)F%M]NG\pHi939+H>4<i?'o&B=\?.tlJQ
e;dj"]>]J,Lh7/MhW`lK#7)0s]gca6>4LA*OM3XD\]_.>lL,*:e7$*muG\eBcHmPXaL.uXTQfuqj)r=i8hR6DE<KB^@<UBG%5.<Ug%Mp0'\B3DE9`i`*
I1VOm5!bIZcp2+5@3B0k5[W[EKf>RS'@YLgGl_IT`a,cK3/4F'_DccX^k458[?=F\4qL.*eT8UR/T.<dR1uXtMfEl9?5$)a6jnP+Z+Wp,b.`\C!-L6o3
H:#p=F/82gbK=Zcdd<Y;;+p+S"F`c.hUaRFAn\OE2Zk;I69MbaSpcMZVr$03q[BmDZXoOCB/\3)@eqrW"qFT897#.7*kf<D=@)9fq40Pe3MA[]!Q2*^+
aJMf<THX#+'C;Qn^1e9gE$er3Q%Y9=nE3\)B14AL7E1sB;T07>lUt!<=F6MEGpd,Vg;/8<L,k0(#btbQ7]ig\77E<02I6:]&RWCpJJ%"LtfA%`dG^M\o
>'fK//lWCYbR*p4gk'2Y;T\M^amXS0:j84g;\iEq9-W"Bu?007rr4m%bNECI!CR03o#!eA8I#Qq,8Q;`T6h/a0HfMga*?obDaC"ZDurL+CjInel7Z".N
cg*\Dc[8+i'u7Bki$O"pX_D3OD)#BN[E^q-9%rJQndqWJP#()BWS_$MM@q[Gg:'0jIlEO4JVf.)!@eBnG2S#/r;B8ZEZT@`Z3+R,adLP*ksV:BRJUj.2
>ImW:L^&@9``F&fglc8L*b:=D+Y!nd`UU$oi)Eo\5URQCpZ]:2Zio;E8L-?,@7E5r(SQhh%S@C(6C/BqUQW$Y%h*Ge(P?9[7AbPW7]7_\4.NY.Cpmu3q
k#54P'%7.i"\VOeBn>Yq..S+R)[u_?>=?,Z6MJ-J"+6Pd1q5`_BnguO8a>7;$KY.'Fu38d'?`/hTJ.:'L^i(8nrfk/i`%O^ALqAbhi,PBrH"#NYc&e>W
Qf`[=FjLQ^K,4o"ojb*QGm!?Q,!RF*QM9!\kkbR(J$`.ZZ72+X*TTdr=BfM/*%bP+Kp$?-EPH"+jeuus';\sAhuo9/&bcg%O6HbOq.d7O2[NhkeP5cd"
RI_j]?OhK_Oh_2tp3L8hEO+?n1(gV^3rm/_j:H9RX1pN.9pk9B?cfEjpq]XEmR]i&Wj*799J,'FF_]B4\D_BQk?`H*SDUG"]T"D&GPhdr%f"HdJiin\[
:G@/FW+6i+k*WUjcbR+?BcasgJgE/,=lF9]_C7$;f:).&$S8^KR[Nq3Quj6PaZ^kD3[e'd%f7S*K@b7pEC:7,m`LKXp_s71SHk<=VT:e(OpmRK#d/,et
:Ln&6s&Gq%d\PqQF06Y;VT(&m$1_I,.]22@-E`T^akP(JY$@&kDE7f(lMN[^MN9A+ah'6[)<Obt-f@*p3Ij3!'m.t')&Ln]pFT;I7mi,<>Zo?"m_.`2e
<BMiI!NlHT:1@X1f1R%b/+qEP=Zm537U=j1[T'DVW)Hhdg4O:5Y.ZN;Dn)#t(8#n-2(g'5+e?AN>SU8TIiVK<>QV<GS_lOLDJ,_-kYD(=:g!%5WL=?k-
WWG;1qDe;.=5Ob%kJgN]*ND6C[kA1Z"ENo'E'9^:Sc7>QF+Tf+cX-`,VD:M(YW+6_'kb'J;^a?6O6B%SHt.3IPd24'9I+3k7>X>nIkRan%)GY3d,3DP<
hP#k<bm99V'^\K%h$TO`Y',k+hCU1d-kg`fcm'-fTZ5jl*%HoE7qt`F#O).SXA^Rh63^_6Fo@JH-.JVIr4%KPAosmC"Fu`^)`l$<2X^AjZ^rWidT,\>5
SYJCrq<<i#?Ii-:>PUIHlZ2G3#rpF1Aqfm>T1&l8`fi,![@3(`X:X)"NGf6u=,2?r$r1gsWmaR!+tj,iEnD.)O4iX1-6%q3mhpmNm_ST:nGE\_S?;b_(
i3L+fTr4Z#mgJM__go3S,%>Bsei5iNR)/idO1R*lV[<cooaDH3(5u(Ni(E;:t!)e*q1$^\i/.'\Y\A@e2Agt;`6]Q1U+9p!iG.1[iH/8@j+bX3U!akpZ
q.=H>NIT8=j09C+c')`%iqlL6g;nk6-Sc&1nX;jdC\L\PT/-:UPib0D)*+B,-SO=T&G>jXXrI!)[A-38E]kfeI$9`b#4b6qcG>NB8"Y:/R=S(G$'=.I(
3#/brbC>Ifp\V-T^m):Mhc"UTfV+2*_"h-%Y&)&"C$@WlK`skn=*c?[6!OT0>N7!2NA82@75L&fo=!DqnutjVJI^R(uaZcrfG>@ah7:G[q[-14]:O3gY
bd7A#iks=5[HfcF^"pY/"Q8Np&<6#JO9&\2dHLb+%f3$G"0_i0Fs3jIUZ0m#agK1a.;1TS<IAOd`r<bX('X/PM3]W,ld3k1*XU>:,49(r@tm6$u47/e=
J"(S,_J#Ptrq\9kA*jc=f3cI@D'g,WeAZYp[+8tHh->F1Sg%4].o^XJ8`c)-WCcQcD!oSnL!V#F=E0B0WVfa4`*_=TuC5Y\_D_CGH]#F%#=:s65i`XrA
.+5\;"fNflP4+esG+^MtIS]QY9$UVl3L>lTm/`kbcLNi<Pm,\G7=]_0$Wb9KAoo\qU-B]ULb9h`.F\og^Ad,0ITs,$Zb$NtpPNFOF_f3$d+/&V6-C7ec
^@I9g9X!e#b5T3N1SgCK.s'q.1HBp4"?!r#e#B^]ls=jnF'=0Nn:?&&P-%i?'n1Q?$Y'o#E@KWa[/bIb6t\qf9)UrAIjFZ`#dPue5uFgVVMfH[F;-=/T
/=u=3NO5rp$)b&])hG?Z&:PSe.Il2S1&Q#]GVU,`)-Um)<m1iOE'j+T;!4Y9Bmq.FGfB03qhgi\\!G[LSpA]]$7%;.SOBVg<!*L]j(Gjh;XW?!Z?%9L1
7-/[MpW6Oda5R>sIOhq5"fdN@=hk.bbrA3jm+KL<0)YDfkG%`ubn+l5%cF;GB4_D"d`P`o)K<,pZBU;"sCE,1*C:%c+LU:`rPfXCM:gU"HL7/f?:NB,X
/mQ[5X8P2-!moe0VKnlDE9EI,F;WQB"LoHg&7>6W!k!*R=!Fq4tS4%!.u]*!8SVF?APc+(W\.\PNn05-&l`.p)W0g)Aq)]@Z:V8ic/e\Dk_RoXGKA#JA
Rk6^8C.t6g!"#!$ECa)CRT'_(H^TO4a@5kBncUmRe:ckDkY:L"1Bd$B-]WkCj'nJr'D?;7Y5KDId8[#eK8G=r>8PeoINDXjEi_4ptY!4s0e!2E(,:r(&
p]a6rk5sIM*PK8T%Fu7+D<SE^(:(,1"Pe?@D[u6cIV1;#)3bB'Jo9fE_&6-M\\k%Il]*'GKhe<i"9*CgLJcRn,SgP^pX6XfKDgoL;T1QK)9)i!Srp6e+
'd$\N/dLkWB<0aEh8mVHg>.Z7+XcOXtt)c<A<VO7V*I:'9o8S[C`"b;(DgO3h"&r/U?bHRu.gCSuD):S$Rjq0kMVG7+$$lH\MEY`Wa-XbL$^7HWV!+^#
1)kEE!"?WA\[0_.n/A=%fAM?FlWRV,7'qmRQ\/:;C:5T+rFjOM71rN>[pBD(;^):K;bC9#4sg[K)?5_][6B[`!3-Y$85-1&ad9]e/L_F`GFWXD4q#>0g
7T2ihMC/LH:W?J`adl7]3r#tm(L&k,]W:s7*0\/bf0JA..?/TO9tNNSSlCFlMWp-]`2*ekm^3HH@Y"6U*4ch@N2]mi@gJD6M'AQDB'Vus;)G(1lF.8Ea
6NUF)669<hpisuhp;G`rf[nA'jbm&PT4$l)k6L,_kMJY;1-qS\W0a%V)JP2W5&'%[lLI9iA]eRMFbKi_DFK,f#?=ZV0ne5edFPXiuO;7hr+Fs:q[H@oN
;!+B3TS,ka)s?"DOCm@Z4L``$(^$js+^F'j8Q;`EHH?-/rXHVFeHIKRq@n!c09HLom7QRW#i(pQZgOl^[;>0bX'g_6@XKpigqBERE@O;tJe-,Dm\P"Rm
.qE\b4gJ<Q]b3i"ea<'R=sSR5nb5o[&m7V3IUUs*H[gH;`:Gb<Ep:^<n1(u"+GoZLQC-[(MY=n>L+K82+HB`[j`OcP(;3dZE3k!PE;5!)Bi;Dp"Ru3`'
>(2[5HZ^8-&b*>0u7H<DHU<J(YoJd)rMJG^bb-"9i[aX[bBM,G%Isi3@c^7GW[?i>7e$<H2E4bo8=-m]^pF]lnB`VpDEF_#BF4I-E0>q/d#Vn]F)84(K
\8k_l,p)VME=@9WMaYs5ZN`a:K2IhEC>HXNs,BU,+hW2)#.=E5Q9EE7?ALP/5fgF9_tV-%O>(3R^VD1<i8d8!=>2'AEQH-6kU[C=P/77K4TgeOHI&Pol
u5L`9El)buaD<fUt?.qm-SoK\*hga)\'G#hKjL8Uirj)abNW-ZJfV)L_8j0uK1ht+u[$d_#F"m:YB1#enPIr:rFT!CV<GhPtSi8N$TCUL;?ZU":iWN"m
\X-"0g!(^e.[WYd;tA2`bP;>f;V[8VXl;!9O7K\">fU.LJc-nSl/=ED:a":>prkS<*0[<3n!3A(=@EP6D7A9>)'EF49P.?Kr;gtcSAr.tf:A&i!8W$D0
SujM,Ii\;1_GcC:TVJYR"\j.qFR)Q;)L::Ac1(&(`ZP\#-"(()ShkF$0*j+Vc6XNOatCKeqHpE3f*CAL@NYrXcH&3T6E06&Asj9\Ql;:R"&mP?E_7nYe
hU)72951s(3R<Mj\b337@^N20Bf.K!Sk&?"JNBJn*9RF"l",=8&"J=nMZ][$q:0^cE2:kKuj/k5e+Sn7)Q*Z9#FO^:>/MlP';NJC^S9"7Tb2F9aP77`8
[ak_0GAE3@9GQRWD1*[`kI^BEOHoq"\XKG-r5ps)M="f(AGqq_OuA\1`!Wn6(]/+J",$i^*A@Qc0$;^Ko?l>fPCpE&h-<@(c:nnMYl;O%FbM.%3HRojH
5e"8,H-`&1G.R_b=l4gXB\?X(SAmY:/]PUGF]V&(u8eRNA8#te3L6)6Gj^fsf!r3kthd5]n1.&$QNnes^ETGtcLtN,d<"RTHj`>N4+C'ae51d)-GfQrQ
^pS4bj/jNT,kcY+B-\cjKJ*qCLq'6pLlcl!Xh*$)T$'c2#DdYMMuG!U`B9Xi=[2:b!9mY)nI[:KYZk>U<Ig-h2!.(l)VWOq!?e`tjWM]6PE,Y7(qa%S4
Oe39!Z!X,GQK0[F&(A?P*TH/q<0.$Z)D7KpsQ9'%\b8K*;>/ZF\CsK215mn/Q@2l(+KG38+XiG*\#Y85.qsK7Vp%hnbuSjSl2r)i$dn)^Q-'Mi+_P>kX
qWJ[UD5m?Tdma(@@dsAkM]:1NYJ'Q*RjJ.$G=?[Wi&TY:]ZN[GKI6jDa[L-S<8uPUZ0t\-U=R<kd+p@+0-68<^7P>#NW8Ym-q(=2Hm(G,G"b*2"4X47)
(m;[Hh\nE:m&'CoO5n;V.U%PI+:',AF2_k+Q)8`$Y=khRQ3E*iQD]OPrP6XOl\p-_Q#08$k;XXU.m2io*]K?IhljC=*h]5i*<5(2Z`Zf1e"flM[4dQ6\
J9@ijZrTMDF80:>Wo,"ttLp$(A9HG[`TnQBlP+6>S0ZGTNeXZB.4GQU>LS4Og/>:a%hP>f(TS)EI:PSVh_(Ri4jZi*(S#;!5n,S)+,bP2\j1L5_h]>Al
N4L^+#-aC"Gm*_r3.(Dajt/EhYbNLWPQJai+ZWP"\@:>b>^`r^W/Y@=0tP4BM"s<&kQ#VW$Hf6'O!p%E+jb,o;mL09Bsa6V?bfscn<^A:jE+`8]tPY%A
h9I7dN=7O.E`rpc";7EUjWEX<H<"Y'e><-P^;F>`#GX<pa+D\!C'KM?fPm3m;L]L-ZT=__<Z8gbP$+CATN2`kG[S1kU?Y+^OR2F!cV)=nN?j0c:sDS-r
)o3BOte#[1:]AJ?2tun@4SW$AI4:jt_Vke<ot#%8>5H<c"Me,*7=LhZd_bkf+K2ll(/Ph^;RAX:lr(?g!5&<^\C4G\:9M7fM1DX(*9TYb!V2_0!=g$f_
0<D<",6m&/h'(,F5p64HMt7Jd.Ec-]h_.GCL3(6pAcJ8:IuAJ:>3,Pt"90N6OfK6,_]Wd4I`)5nrg`Q]%7rB<,;SnT<<N*[kC@njt+X[1IjeSsF)H?o:
DY.X;/bf+Wo0C\i\q"u/)'tsKG,!O&CW`dFBX^PG:`+dIJr9.`M81<f^")QaMY]S;n%_>qUaC9\`&L=r!B?-9^-;]jNNNZpb)(BEEiXB*u#>BCco^"rU
J"bO7[/[6dF[DPM/1MIN]DG)!Y`0pp/"]6i+S-0<Jh(@#eF9f2^300G'I=l-"o1J`chV9(\KAsVi^8`J%p#_'Eu_!=h95;niEF[>d[><V,I3-qMIjT[r
nQ+7L`7bP!3A@bJ17*8Z*o$[M6j?&A(1BqWFu.hp2T.k%HRGsSWc<+>J1;9$KhJ?'%%"3#jYV!\gV@_>_ZliKb8/QI+,rU$qcoNn.pF86#0$g4a#qo,`
0t,SVRQnO.NZc3M$eAKTV3.UN9(J"5]j&R;,W=1rVc"%,fAq35KfX,b\\?H/cEt4&!#kocg0OJH!I.Ni;IeCl&6QQW3nP4'S_;3H$:TkGr33&@r41E0[
Ni&8VXC2Yg:ZUGQJt:+M.8cYQhB7lVC7fh!D%(3/@ojPl)H&-)d.lcQkEpY/$f^quL;9FTLXf&,&dI-*4o^D@rAaAhgh6a<8M@+-JiGog\*-LXAqp:gd
+Lb1S9n&KT*&I[r&:3US`Z)T>98gQsNii8ardtrenRgW`g-L7KD'>,5eMS6hFPQHrLc@qY7*YJ)OpRb6#-ShUcPq64cEb3MIG6ipUC2]/\A.uDM_/=BC
jT\Mim`B/1`Xe?)??q<U'@D=];@89o-uSgEo<lC1.>@g*"JH0@_*=W\&=e+-aZ'C-IPnFFFAiL&[J*"W'$fjO!JA#iW&[qn[h.a)_^J,TZ3%Wrm1C2BI
3rZXA0i/TebZT0k1VR'O\T+/I1:'.P#?Bb4,njU(q@pLA?m)n?4_311`c:LJ[@sUUE](OpXiZb?t4VmEubIOk0W.>J!\Ak2f401-/ke9',(P[iDMZsmC
b4H=hCU*RgugQm&;V[cD8O0]3Nq]0RAq#WQ9/P[=e_%oGpN%>U<0h4X"3^2>5Dc')%<k%)btiq'!!V)65g"HKqjY&^;n,Dq'\L#cQn(DV;bP.W2C"F?D
fte]%]M-KhEnK1":%FrN$?I"8Pr).SNpD'+g0.s6s\R%&`m/Oq!">qb$G2EMbs)@!mQ6+BN#gs(BuXpqe?F(uVj)[hJeg#m'BUo=8iDC]%%5jB%q=LjS
^Y!!_\PF0J#?kI#i+hDPUS38lm%$b=jaJN^k9,e$k3fE`"buq-9n$#UO>`@CK_:(MGOq(+`=pc3&CY6i(i:CoV'rC"c]n(V"a=Jr'o#SSh+p=M-:&H[<
5n5ko:2DLj63Mi$Oqg0Q;)$,&78]PJY58iecVp7SB&BqUe4R#6`k3t#g)cTi9XTKGeAXoJL=&aCfs4M_@0&4AReGs/6Bo:\cG'mJ&L;hPIaEl4mRp=2r
Y%m(W0$*$!,F,^hM+tS`?O,-U9K9E*%ProS>a[f(FPMkSZG&L+[!Y@NAd%s>TORJj([ekhd,>GISSd(KX,:;?^OPSk3X5tm>2tYYtbBA5Z!be[ma$moS
1F$jd#a;Si("4hEh=>BPcC[7XJlK'o28:dfq*/B1HInd-G,@,B3hn402V7XjVU^V"OGg#3aGVN_pk<JD,_?YWD]Z:7:Y2hOU:Jm=F6I?$UC^7or7\2?(
ES6[nqY/mo@["H&26-VW2Fcj>2@561V!5ZQ=2F;t8&"PU2e700$J+fZ=sO&"*`!Y!e)AN])\a+QND_[`nki&Q"G8N%Fh;+,sSM@!Fa6TonhGRo(r#tCM
r?.8LBaJG+.bCO5f-D&EWqlADZa%A7EeK<s4of%uEL&eJEV+;#P^U81qo4l,Xj-OWt8i@"-BFBo2aOQ<tJIES+fpj.G.WIdfLa3O,@a_5h$J(GubQ5\,
N_4=h.`;]]fdD7jIBN%\6k'Ri`Y0qn,\@ik#2luC&/Yn_8EE'<>3beON*a`".HO/1=JD=HUro^d0L,L"@_8G)R@/PM'rFF0k'7%Y:ou#?n)'Sr(7-is4
W)8l_epMXTN\^cP\)^L<#GNaDCPXSkIs+'a.7b0'sujTUGf3-c3%bJ2S_6@P[Vc5A)fPOR^`qq!>+NB+:L@r%#asrMoW\a&H:mrH7UXS=4HM7b5Y80bf
Y\eH8ST!kdN5UDTK*rZEprs?k3I0L'5gD(.uW2O@P/IH_rR\gc[[[!HcP)a.K9[Ek&'F:[##*.\s<;RZ;:SE8SF^^`[JtNdf&(#4sje-P$W;On7Y@52p
nJ%VU!Y!gk24's59^>3r/8X]2-!Cmgc6@-_@EbOSr;bigP<NU<#O#['Rmg!rY_.Ua(/fQjjYq!cgn]!;7d<R!a/5q[0s:Ka3__HLSKaO_lbQ-fS,lBPL
sq[4RDFXLK_5.i=qi)_G+V1>&TLF::UHk>pdj8\Jnk>%WF\iQF:8mpC3,X,pNA$^d<o6n*c.I7:GBR"K(=&n19@^&?RQ.W-!S'nO%PNfopR^dhB!GtNr
PBX:HhCT<!g):2sO%;$[=?2Y/1/r+V$K0<!M+m=T6#Eirrh`$PB[G!q_/e%phb@qR(DKV18W99=4m2XFd+FKi!,]N1]p%8f*i(YB1W2u_PCo'jiOsBo3
hKc/oQgePZ2Gp0/GW6g_+TVN&&SgLMJ.T6on.Z9l[(B.@e#?),'GD:FacMs:cWCfKiXJM(,YE$jXTYo#H'66-T\;tAF/Id>GHKo2s*1uh09!K%V]MBYf
T;uJ0a;&!#_OMa=Cle+^RYuL0oY#6)[XhlsY[Yb*Z;r%J6L=nPAXhPnV`.CpUdD[^Jeh])+%f`Or#XN7P':Jd@jN;?jhY_C<5t)T1o^nYusZ[6;r4+&c
12,!jfq.9+jrHY2tCYBlPl+O-j9]RX53k@>CrVae(ZJ+0S,CT\i$97bHl_Yf#c,\&"`m:U')W,kM[1#KWZk4W'=L'</JYSUt4XO`2P(dSXDcU\'E1(IS
a2eQCYLMTd<:]G$4m;UP#D>9L=UZp9a8"QO7A#/a7#=C'jp2tiq:WuhKAgN8ProZP!g[Q[`Q3i<MNP/p*'Vn:jH8]=ma_gE_[],.-/risc&)ql@%k^S%
Nu;$'\Vna#^Le^R%8'foo`l[-=4o!'kfMp46L>\)(p]1gecW_N7P7na^@p'XF[+2\mGaC(H0?T-DlB/0c@3;0fq&81[mq4.3Ul;qG%K7]lRS<DQ%5=\2
f1S%g^9hn1)REBT+ei]G(q"jRX?U/?;&3V;bY@U(CLq@!Z0sJ-"ca.gDNX;;4nArb7@O)%)H?5i.sa:"hf[E8+,K'1Fo^HW]qF0DlO_U-KlINdS!-=s"
U%DA=+6=iN2""dF@B_l[dk-*8q2NDsYbNN\C(;W0+CPK.%hrnWe)B*2a[`XTDI6rSDK+YL]?+"8rn48T(Ic=@fK4Dau`A`$)>j&uTkLlH4TjV%:&#HY!
c;?nEF%\E9p0Y>fc>=j@UDCV=(pFU[3JS/ANIb>H6npgJoCn"@\cb6)c_;C$dMD\W2gi5SNBpNL<famDfGr5hJqo!*)tJfCAtf?)[EM^L-)n;]tV2CuA
7Eo\>6,u1X6:RjJBW)^SY2\f_kj*,EZ1OMSG@Zue.)XoJ@RQk2Gn_elC/H=>5?Q-5AXWI2Fb<UR^8G\6==Q^I4QF5bD"nBAZBaa`[[PhRgCeY=_AF2>\
*o!!eRhnaOWLD41r0<BO(>M,Vkh@.Blp8=Mp',.CAVCISl>3#k0?sEHqUYL%]$k]<jA$'Y'D)aIlUf9kR)(lZXdP-6:5ngPU:@WJm6=ba%oo=[LeI2`k
?,8^4a^4k;SP26ZT!8Q'P55Q(l\Rbl8d":)U6YKn\Ma,Ph_?GacrM+>eG(N,h?bP]P*)n8s\3O%5?qkUk!S$<ZP6,_DK?hM,o25rJJ5FS]^LN_gN._kh
5hL!qe'qlZuDFB(HP=!(<Q1%5S+$I#0^W$Im:,We-06>@Z>\#aRCC['!"X2pT&`Q)48&D=^*)''#$m'p2ktT_I"(PIca%UN^lcj^RJKK?!8^D:5GnM>p
$f"A/8ACGV6#:<@mmZDs5_/a1Fd\:m_3J9A$KM^euq5%l8mU3JLn#N\nR<j,'AI[eJa\_qdTB(98?@?J`p.&-=hBfXtC4qg,q.(3a*(UF,FS<7oW;'Uq
<M''@=&Sa>0/65l4Kj.!`&=$(a75Bk+n5"HCjG_$S@X;XTZpCF+'bfN&^sW.=XrnppXGPJAW>^h$P-O%u>iZ]'+1<,cFA*bn&l!Pl#=p'2C1uUo*f[Z+
_VDp_k0]^GT!Y;:!5^D7>Z0rR+*=9T*58o7R1mBQ`Clp!#RM7)k`Mr0/K+q=ZOZ]ni!B5KhIJ7JKtdGM?GVU<Vh9q:UU4GZRZ>Z/$.%-1id76*n]eC&C
2')7[k8727ZA#Z0ss$Z]kk(]kUD2l/cF%[>9Z6eTtFe4j36];7.9=k%ioN6)DQ-1[.?sSo;DF2?_=&2-97F3M*/7tFl;Drb_MC6<8V]V7DDR<;41C#K&
(lFo:A_/\KcVF\(]0%GM/#P&P>D3DhA0[9fECP#3KJUcG__^)B"%MO>fRfPM.8Ff_58<#gKXmNPs\V.1b!O:NT\&:eVaeMb7:1c!sU2eZ&8D9Gp5T=0L
=-dho1k:D4A[Vr"LdN&Hi.a6ZQ7-MZ\&>9Uo`h6(\+![JK-#aEm>!;bK=9pQp6XOZ"8C=ce5]FDGFLEY`qlOH\[M<QsShc6g3m'5PRF=;:,_n.?)B1e#
6Eb-kQ#oc"!TnM17)85MMOjognrHcLn4#><O&[p']b2%.G?>@.pU8b7T5Z_t:++mF`,2%24,Ju)`gIXcWnupqDpa<G;NZqt"ZngD7\FnM\#!THs&-XT9
_%o+':sf,iDg--]Nja&_>4QV62)g)h67cC*03N%G8T.&FJuo;q4%t<9!<OYL(4%hjR/,&6!6"HJMKmJ&O=t4kZ$2L8nK.0)bW0]3i(E1nT&n5?I?m3/Y
&do.q;#\aeL]Q@gn'[bK[LE$)"4u:EaLrOI9f%)`LX6t6,QiFE)A.C.!18g)nE1"mZuW36IF@WTQ2Zr+f5&6\0=,e^>JNALcDq!rm2Pd:P98ldn)7E=H
j9C=-_ShaB3NX4C)PIl`\o:c#i3IR>lZ%5hukn^=ecJ5s+W(^/tNNHp+\CK;m4H]%^EITd(\6(&blF36T@/@]p12O2p;)@l]<,@4>SBGB$9nBHnGS&\o
U.?2Y7srqY#FY(fnRVo``CCt@f/TAd$Vg.u@lIm/ahBLZTqljVqPU4>bB>#^!Y`Gq).k7u3.M3IZ7n5&gE:@'0r=IpM'X+;=NPF3,XKe6g4ge#I!Z8o3
Q2uNPoQ[+Rd%.T.[mat=]hGppJR%$7+ngK`/2ZqiB443'SMW(<sUT?.XcbeIqQrb[aq`j'V6mE:I,A5>BPa]*Qd]0O`%Y4*,()i0-[d*5H?L5PW%L'Y[
j>/iCq;l7HYiY#rg52NT#i3S`b]W:cMSA3b9L9<'2#sdrPGV5i#6"-/<TG&1pMV2OKH4ja:Q,ajIdm-0=G3JD"34j@WcUqoa3c>$6Vc`)Q`S+h.-so@i
-_Qu,Ns<jd42>rp!1C2d9iN&FBQneSi)H#mToj3R'1Mk'JP##$->hng74%_cKjToK2$ktlfc%?&NIkMC'4I82NnN;B%5ATdA\%`k/[8^.dYceOk\r"Rl
#CZ)nPaa9R=dm>%$Pg"AKChIAX.*rUIg)$MgOB"beF6-d#o4q9nIq,e(spNcFF9?7b<12,>]WFf0\>^PN4Jm.U,_%+7Pc+eK2u:mpkVEGaV=Wel*6d@V
KbnC0(aFi.2?Td@Wn%hPS?nSQHO'03RA^1T91#2]OCCc-;'r4KVs5)AEX>\tBoo+am)m!C(0"pN0:#1H-^HV#kPmJ8;6O@AGXTT$:*D4$_Ks.A8rU_!-
.,+?3@>d!),N!U^/N6bJne2$]k/:oBiMcC!_J7_/KX+1n5DY4V!bIF*3CD;;p[$]BKcpO>!]3`8<@R^W<l-p.u$OI<YLbY$L$j5BLmk$CuHRn_,EH/V#
_oGd'H0S:H]6lu`\8Ksb[V+]pV.T"</+b1:C1:q7#SR'3&cE9s@Y6NG!_7HGp'`kSOdL<tO6<(E5#PIknY7g][k)>73%IrY>JL]Lc2jhf1!Wk5oEA!PX
BusDLAof1nmlK!2-O(r=acJ'%)R(oJ]'e_p%h?OC`I8e58@YRO9G6p#g&3[\eq0s(,0e;G-O0LZHeq(_Q6*@)r>Ld!Sp"L.bH>7Hu1F+8S9a:?pt^N>[
=<$3Xtrs`_R$8qm#DkZGqf@O(B`<SWcQE['h)91Om^MH3aK[oC;Ko*`Y#\5n=22CtJPKCjJ_)A;##%*G<!?a?u-Q;+Ak!f`nG_++bm9ZMqBs-o<3H<MC
+tNHJA3.@-B39X)1d]>+Z\MQHgm9;(MS+a:j"a4PF836_=T4u:5%/28<.m@WKWO$2B1IFZrPIIPLe6rQZ/GAMYd`]G?-cSp0b^2ti!S&2RH;kuhH!Kec
1OQ(EL>N?_PG#HbW?L)H=1!qKO<PX7-DCjgd=,ONG-;%L_*MT_>qX@ekLaQifH1GQB=Lgi!h@km5)KKq.!lf8]LOsS6@T$2:l)ZS]U_md)(mh;52Ua`C
R#,-Hf41M2(40/@"G>WAUUXq7K>[u*<.s^V.OL]coCiFUNoPTJ>ZHT$*H^Emg>_.@U@5Zs),-X2mQ-7Eb(*XKke3@Ps/F3:K[umDRq))i;D-=tgaJ5Yl
r,bUP05Gq*\"K="e)#O#%Pr,Roe5<DET[PoI31Ndd!93g"[ACd';ZK(u-so6J1+?C\Ltgl2),Ono;;Qn1-rng,jqscOClAFPFQSEX26\hp*8]QuLFagR
fAqSkdLq&`5%G-0lZcggE#F@Q*+-0\4Hb-+I@$Q4!Xg)\O^j7`gt+$WkjPIt@fh$$Z3J%@CukZ#l7;0L/Ec=mf_95k;j%&'QVm?I*uEEF#f014\DrZ3%
R@6k`SA=5Z\9BI,pMR7#)[Kbf+t9OhVkPT8uYO"d8%&("$PNPJS\3CLOlFJEok!K4[_6LIg?6#$Y+)$0gq:W-h3lHi1_W0\FIbG#g=6Ju.u.&ht!:]&.
<4[,`.@Zr1(=a'`Q<RN?ZM(b?:ID]up%a/^f@(#HUWX-n;8(KhbJs4"&0.*OdhnUa5Ga-17CH>R;H#6?,bQhDk69@8YlriF.gK.j]jsKRq[>6s3"_VbR
FuhB>%iQ`G\UZqp7:\0%Gh`>_QM[im*C>ni6hH1<dr\:"AjlM#k1YC1Qit!jV[Sk;l_Kk6;iVk7@p&Eef7A82)a2c1s%uI1aSo?%^4GjCE;t7T>f9%MV
l#--Oqp?:#&o^EAcER">eltW,&u=)0EsEV'OGrf\u=YoR@T5PE*6rhm*VUieqjM^`]K$)kFg4p%Lj$>kXU2d<Dqnc=mC=])]]s#%bI(qo.r1cXZO8`.J
uD%4sUib-8/(]ESMhi2=N"I56$\F?_D4g4JqA>r"sUu]NK-U:BQ=9PUlX`\gd:?aU4p<@K'G'lq<3o^)#8OFDlo:mtFApQY+MjM>;U2UW$L$$Mi\#Z+U
6Y#.60S>D'PDl/GD4V(pnuRB_P"k9qKW9bHW;3]P42Bdi(u'k+P8CZa^/(bBR^A7u53Ig''[-HX3(,<,+D=Vo@4>:`7Xe8+cr?GC`tH8^L$,uO(`W;?/
4i'2a/)GL7amVOT6ipS?8#7N.Y"!t6nGEAHR;fLPeFD?Lo1o#A"+h@E.bX&^'P^PnIV>b4T5Vk^nSff(=iHZV&5r0:_Lr+t67j[Yk.sgKZACM';YeUac
ZD8.\lN,J)$le,+aum\LOrn/^70/$XG*0u*s!,o_6H$IlkfcOQ[&(tMd7Y7,*TG&PTa04,(I"D6<b@+C"a8\l>[+aPfLU:&kAj.a]'^%1rs0\*0+LZRl
`AF#kTO_]@(0E2=kghGEI8Rg9ukJO8RI1)C0!QN2e+n:2q[/GDT>mQE`Z=]68P]e0G_I*.ig(!amf`b"-"Z%]FD86Dpo1Di%tW2*<k>r:0"b>9KCnTI)
C*bmPt?fHX^'PMr@TqLYPifYthV**%$f&Ljr`qD8l9nD^0/F4IV'IX5'oJm/eUKCQl!NhVlgm,Ck&3EtM;hon.q4gRt#:FpCWDH\r\'_8'!*_XJJ?l4i
O+SD,MfdXeMCjB;-,c-W:6Uc)Lc\p$H&e"SK:ZXecb!b)V24Y]HNbCPqSY==!5hM2I%6L'?_)F"cF<P\Nap:[JSVVH+T%j<on#gq>IA_A8M*[=u?pgG%
o)S'EB@(`(XY[j"7]VG7WS3B/@B=I',#:Ud^ho\SPdU4qBq_LTJ@ep41@lhH?i#X8R:t6Dc,Ha67^,n*0Y7O&NI3Fs`eu:M0V#C'ZXjrHc2Z6r6&q6\C
-`&riXf$jTD'(s;<Um&ndQT^#6GYsH5o'!Hs-,"gf<N>Xi!!OS/_=D(H;]B)EJ)kHYpC)V!lZ"KK]_FpW#(Lq5abu^aNP;^ScPu;m>*`]8bLs^O,,M`L
2+867mr8kpY`hS,*q.[QO5?,:mA<dbk7=FfRI8"jeX]2o<ulu4CXcBPf"2%+?9f/e_#bZG\>=;+5iQS5@oUhGHdU6qVR/,gHA"LZNU7^<R@kV/-R2tr*
.$s_2CR?pPR-_EqU&\+[URcX1$@R,CjKDQ\?t9ngV:8n=-gtre!t/X[ED%.<8o)\u@^d&"^[IH;gk,dTabJ7ba%U-Utg/1>IT<MuOm2!n(tD[-!>^2eq
fFr*6NZ_0)WGl1s`'nf_VdboA8G<NBaF:r)GQp^';AAfp0#57_8GFUf^68kr0E7/_035"#2ko8@3KDk?N@3Yq[pVe.&51GKgc2c1cTO:]%`SopfNGk8D
i@FcPm+"A$loG;'YA20V+XPG#mA*X%!AM*:7P6a(l%uM->R![)R!K.qAB)NBS0YQS>NL`s4"a`bLkj]hQngl9%!gd+<fPJ-]5=T0Y9?!MVPKf:3%"[QW
h`'uFF6J*p)r[ct_p'<)bnA7e(<YFCS[:.J;tbte]]%2QAHMg4g&G][e_4,%&=WQI0Kp^*q3Hm0+<O.T2o=N)_(E))?id\)I6ZB1Pg980Ma:YL7W'k/4
-203$G?U%-/:1ZZ+o+r4R'68ro>#+$pLa'#-@&DpW1"9*Ck['`Zgk/HfT%5k;WQ,Aj4k"MJb#D4=6$+m]2o+RB>bIcZhbV2:^96kj*niVtd4)q$B\FhK
AQ#hdAZC81e.Bb7:U9p#ZTcmU85)d/'FGc":bg>DdPVF\&GT"=%l9k+_Eje(jWQfMJ2k^d;;Ifg\@RMF8D<7q7qV@eabA$%laB$'*Z\6d$$/qU4OFns/
LPV[Gp&J0+$D;jI^-h7G,g>e.>`eFL*!!WAC4\X2&g;_5BR1NMg%He95.!-j%Sm$'f]0H9q>6E3f/;_/9RbUp=$AeZTqX\IQJrs"7q^)b\[,bIj>]Xgg
D'jb;lj\L7pT8.g>5TrL[-#fcPU0#=]MjArgYJl(2U:%9rB&:aQ+)fj(kSLGVLpGL<KP/g_i+/!ofg#6mX*>#/eZC_^f@Crc`-Mfp8^o)(67l/cEUorq
i(OOA%MRLa.\@Qa"k`d#[olHATA@jgn$ZrH:ahWCGLr;9^p*[=K44DmY7W91%T?M,F\MIq*J>MWVkB+:HWZL-)"#!*,)_cs[GMF]XM90Qe!_*0KDu5t)
'u2*E2F>MF2nVb6m[CKR]Ke6@72Du'hgqgM^A(?Ep;hBR<!(jZ\$V3><I87O#o2&2Y*W7iVJ=Z&XD$`_GQ$9$bkcK,!A.b'*'%UJu#G:,kS$@V.-@A,?
d"g!D9Adj:u"7Y,<?3f6j:$"9N\'=p_,VWcd6f735TV&@86W[<]H"3fanbLgZ3AQ_j;\+U>f:[n0'-,V:C4@]Z7H1+`J_r>a4]g^]>:Lp>3\jqNY8@m@
%_9`"/mY,bZI,3RG]Hi2!GUXYUhpR[&d:cMSU1S.Sr+o!r6<tpLa:ggl4P:J,ESiF$N!i[jNHF$Xqn)K:gq0WkLZeQuqU8uRUGTU?k,h`ph]j<Z&(,r.
M!Dp=fikdeYg*n1j,a;.9kJ<em84urZT+s6m4VB,p7J?-rU$dXJ5N#)(;*meD)Srb:gM(Id2mq@.c\<ur#LdAlfmO5Q39ZNqK\O9HZuV:fHajmIXtY8C
VM)R%D,Y_aoA<SIp4aD2=rTD:e&E4oK0nc=EVS95;sKb+S)2,h*nh\SA+k_Q\Zo0cTH61UfHGKa1%4r3L3&g'77&1@3$MV"F#Gm$`D1_;8Djel4bL:b7
$7GPYUA*t"h1N#TO4%#ZGnm<(V-ts5T$LjQJ%MF'\as)f**%^NDD01B5@h>HsDe8Q9pZjP*dDbkQL:9U^2I;1_]QV]4u_5dbQV:Q2HbnA=@J%s"&q!/l
EgNP>)hg73i*A1:?>P]cIebDfqk\d<aIXn%HA<`+rIO^ZVom19lu($B&DY":F2*9%%.PT?a0[-_CbK=.o,=q/re,UJ:kk3R`JEYDI"eJ,oigHTO^0jPt
.l`(P`fQsh0s&M.7:;'1-sa*)Ci:#EIS%o;t/ap?5>mHh5&e@#]!Wflg.SQ*2<bZm?:?d,r'k>pdQ5p0a@QJk\mk@0$IVi7=%orT-+cnYu&Ld=HtGShi
,l;p^#fl@%8"u0mop.,uA][DqBHI@c6CHuApB<=QjFA>BZ@nfuHO\ht4nl73q^47\X/8&87SBNa:lP"_'/].]C$u+9'PRj0fS0i5j`L'X`pSZ@V;N![H
37h7sXi9('@g,H!X(NJc<%)XqJsguS_W!iK34POGdD9*%GFL_#ZJmZQX2.&inrh.=]==/SD!J0gWl=;gn\hTiHBC^pO2JP2Qh`<tZ%UN3!V@n01i?a1.
>+uH?D'E:q9mX<_&`e*%VciO-'Mi`F39i]o%%=O0_\tEUsMiU925bpV93$#o<1]@@Ep/0JjIpMK:J[%$R5K@i^sfETX5Z2%cP*K.5:]1b1WT"k(e<n*K
f#l!3N]A+K/-^.!LuYr2WO)lA)4QQ75\Lang5/S26LF=TfkOMgj<`/:u6+:$Op@M6t%PM3*@an)8^5I#\B*M.oT+1""5.=N:Jl3KKhsIP7!,T;$Va!mg
pkPeol)aQ0k8kjhkW*8C#7&$gI0f\+^mPc<Wm5,9"X3(`.\Z;Co1S.i3t&?t+0T-F<jhbuZHS2G\BoJ()4Ffq0bp?fg<F6Q,)h:l6b('"Ir9Y)Q*#/sB
u&%$JAhfCko5V&rUg%^EAk^=SBd\*0F9I>22A_SqgmVLJ7<*rK,\hAhLZqS]\5,2>Wl)]hPJ@%K3cI2Wb7ZP]U42YM%aGN"!V<rU'+`n&L]Vf/t4;!uT
#ns$dL?)lUr4lPY,*.;r('"9+=HFdk1fQr#o7'L)RBW"/MIo$A4p'XFlrp#elNhtl0lepDr],p<(H8i$,:h@44VNgs['9_#_`;U^o'$/*/5T831A-b4h
j40M]1t#UHDPBFeMqo&UIe,C!an9=pHXE'EjT,uIg7)W[m$roD>lK4J!:pY<L_l9U'k(9kXhO=57=C*eNj]`48=n6`tc[qV/l#h"7l2s!72)UbXe*V:9
_(cRq6N16(Y`!FB,/fZjJWH$<5oIdt7uUPK,/1fVpm?65IXH4YCY&lW*NI+Hc=iL,_;F8k#<a&<^\74<=*ETC"##n/Nc>8d>&jInt1"C;mEsBEDLp/P;
`:.7#h<Y=AoFNh\83?:s;/DpM(Wq=OAamcl[(oJi!t1E">!P/T!%#Zg8q&i0[-@)YM!-l&ZDSU";7-H0J.blfHs;L1FC]V\9?1`G26X3>4rj4mm5%bEB
Q33Br#X&ZeMa\*[T5(rcq$ROE'TOucdi$Tl0chSPJ)4>F`,TqG582qo<TqpmOR`n!1lOS#grU+FL6kpBCmXk"`K"`eHC=]g.0ANsW/b_juHY3)n!h]oE
Cd=FrWAa?>S`&2(E</;72J]!CSYPiD0\.?D%.r8TPlof50"q-f3T%c6d8Rul(IRQi^U+"%5k!&qoX//NJbL!89TTr,5iGMRX<M'\DtBrb_ssaP2Bjb&>
@1^*^.=noOD-$$>q"X2kLD(,$V.;8o^Xmcq\flEGq7Q7/V]!T8EDJdaJ//GJB83O-M(0Wrfp%>=u"'62sJ!;52W:N1f_aeE8spC6Z/mFfimWk7.Ut1Rq
pf![W#uMY+U*YXE<mq\;HkYYgP@:&J>+N"3"@NFPfKf<emGXbLT15Ej!g0HE1`9@/)f_rE:eQA^`li)Ch743WklQjBYBd6+f:MlZB@#ZL(75Y1G4op2,
LQ#FNDc_Db9:j_<Stf!;<0`>::!Yhj9eK:(NrK;R#VUp98F^O*%Xm6lhg`F_aPIqK2ESuL6Cr$WWY=H=g[P$uTR;8:o.0qY1IScZG].!X67OPuj\U6Q1
)&V(k=?<M&MH8:04>XG3ghqIg,+bk:e*75oUo#bVFNgra_Ooc/@]:4kP1X&/09stHi/Pm[ZQ^3E'P@=o(%hBu$9#l27T]#G;qS:gSj<j@NZ<,jT"o>m9
><kZ@3=CFTI,Mkr7b?\BJ/q,-qY3>:21D[:p5)oa``LL2Zc1b^^h`]LoFJW/W@#ub_M)rSi.sWT6cATE_^Y=MT(sh^"!0;I22&$q'r%L3SV$s<nAOXri
QrAh:Y.BBii>.-m$5>aW)%8>]n[-I.b0Rmkd6PbCjk1mGGM>^V!b2pU0U]8T3F(9J>GklXA49(E&jr\?85`X#+ZBc)u)6*A9b'Z)b-o?rA">sF:9dPAg
O$Sgl#-';,Yh-@ET/!X4\I^""fqS]F:u('sl>S?jrX*^?QZ-.i4KTh,O3?'0*(L-Tj+DE%0aLloC$\b94UDB'NiOnH!^).7LpUf3WVU)^CE1U]qXYG80
a+0O8ETdOZ;Z1&oSi9FT7o>XonDOq[P4c-=7C\dXQ<]F0@33rcuU@qB+OV$);E8M?LE+%^^o5D%hr8IBWc23ZPP*?`nM;SQ=GhGS06G0$^g7n5NRQAt4
+:QHW4EM_^;oTij2',5+L\VDm_bV<d*=gfg],=qcRCZA*Zf=;j8KA6O.9Y)u"1R&^lms*M+?L[9ITh(2B45Lg)Bupc$O#SsL7u"jB9<cJmFB/3&Hok,`
f/AI'K1>s^_k?)3cRoYu&h:otA)7g8@Pf>>:G_KX_gbdLXcls[[:+`t.fXMp;J&]@8D0(oNm5Xt8fNh_^HMo;_AX7S4aC1]f/Kf>5si7LS?@i%ng&[)`
)E(eD[:=0T%fP1gsuA8(M75iQj^+hU03FU8q;-WE,`356#)T<%^u)sOVM+E&'MV3A$h)cUY&*/])Q>E][*R+P!@WsP)9+]pMpQ:m6qP$2YkP4?D@OnJ$
!)5(@A6VP9$>Y[D:d4.H.Sk.r3hjcVr;c'01BC6<_d['gdkK[h58,JrAt3eoAHgS.@*X24kBb,>oO3Y@lu7.(U;/.+q"H*dL5Yn1?b0h]3/NP'$XlKO\
nO>aFpPc(P(LkpfLXW4<0.$u'MKUCKBp=(@Ap.Q/Sd+5JnqjdhUh%N!V(STm7XBQ7kMb(S<#_dqjr5,e`<3ND)k,'DTJEeD;;oETfg?aA_%ekQ'+YL#^
B9YICo[2b3g-4U"8O-YB*56]Y&SPV9`_Zj<J4jJ52^YQ?AS:.Gb5!'u+@7FQr6&DC:!'*;^Pb4)fK(2>:,Wma_`!,/:igdZ!c_t)m0lQSjcMRVr[9"k7
-loH'd'<'*0sG.1qg,;$@q/`W>?DX$:0mum?q---dZH&_<u$j84j[(,+\31$4LUtJ!!0Ta!QjF*:7S8LZ,C;P!WW3#!!HG.'''
	exec(lzma.decompress(base64.a85decode(_SRC_B85)).decode("utf-8"), multiCMD.__dict__)

version = '9.87'
__version__ = version
COMMIT_DATE = '2026-11-10'

MAGIC_NUMBER = 1.61803398875
RANDOM_DESTINATION_SELECTION = False

BYTES_RATE_LIMIT = 0
FILES_RATE_LIMIT = 0

COMMAND_TIMEOUT = 0
NO_CREATE_DIR = False
CONTENT_ONLY = False
WRITE_BESIDE = False  # -wb: every rewrite goes beside the old file, renamed over it once whole (_cp_beside)
READ_FIRST = False  # -rbw: the source is read to its end before a rewrite in place (_read_through)
# True in a move (-rm with a destination): a copy is on disk before it is reported copied (_sync_landing), so no source goes
# while its copy may be only in memory.
SYNC_COPIES = False
MIRROR_FS_PARAMS = True
ERRORS = []

ERROR_TO_RETURNCODE_TABLE = {
	'Success': 0,
	'Runtime Error': 1,
	'Command is empty': 64,
	'General Exception': 70,
	'Permission warning': 77,
	'Platform error': 78,
	'Command timeout': 124,
	'Command not found': 127,
	'Task return code error': 128,
	'Keyboard Interrupt': 130,
	'User requested stop': 147,
	'Fix fs Error': 150,
	'Not supported fs for fixing': 151,
	'Cannot set fs uuid for ntfs': 152,
	'Cannot set fs uuid for hfs': 153,
	'Cannot set fs uuid for ufs': 154,
	'Cannot set fs uuid for bfs': 155,
	'Cannot set fs label for minix': 156,
	'Cannot set fs uuid for minix': 157,
	'Create symlink os error': 160,
	'Create symlink general error': 161,
	'Failed to create symbolic link': 162,
	'Exclude file read error': 170,
	'Exclude file not exist': 171,
	'Partition not found error': 180,
	'Partition copy error': 181,
	'Copy partition info error': 182,
	'Cannot create zfs file system': 183,
	'Cannot create cramfs file system': 184,	
	'Cannot create iso9660 file system': 185,
	'Not supported fs for creating': 186,
	# _run_mkfs_with_fallback() has exhausted every tier (mirrored params, the
	# -O-stripped retry, and bare defaults) and the destination partition has
	# no filesystem at all - a real partition-level failure, not a degrade,
	# so unlike the FS param warnings below it keeps a non-zero code.
	'Create fs error': 187,
	'Copy failed': 190,
	'Copy process crashed': 191,
	'Remove process exception': 192,
	'Directory sync error': 193,
	'Source path type error': 200,
	'Source image mount error': 201,
	'No valid destination': 202,
	'Source path length error': 203,
	'Destination path length error': 204,
	'All destination paths are too long': 205,
	'Source image not found': 206,
	'Source path not found': 207,
	'File list not found': 208,
	'Listing error': 209,
	'Remove file failed': 210,
	'Remove directory failed': 211,
	'Remove skipped': 212,
	# A file named directly (an argument, a list entry) and another source land on one destination file: the first stays,
	# the other is not copied (cp fails the same way). Directories that merge are not a clash; a file landing on a source
	# file is, and refuses the whole run: see _settle_destinations.
	'Destination clash': 213,
	'Dest path DataType error': 220,
	'Destination image broken': 221,
	'Cannot fix destination image': 222,
	'Destination image size too small': 223,
	'Cannot mount destination image': 224,
	'No destination image path': 225,
	'Destination path does not exist': 226,
	'Destination fs type mismatch': 227,
	'Not enough space': 228,
	'No DD source': 230,
	'Too many DD sources': 231,
	'DD source not found': 232,
	'DD source type error': 233,
	'Source partition larger than destination partition': 234,
	'No source paths': 240,
	'Invalid arguments': 241,
	'Diff image error': 242,
	'Patch format error': 243,
	'Incomplete patch': 244,
	'Patch conflict': 245,
	'Patch verify error': 246,
	'Entry state error': 247,
	'Disk layout mismatch': 248,
	'Raw area error': 249,
	'Identity change error': 250,
	'Disk in use': 251,
	'Patch error': 252,
	# FS param mirroring warnings: each of these reports a successful copy that
	# degraded to mkfs defaults (a probe/build failure, or mkfs rejecting the
	# mirrored parameters), not a failed one, so they must not inflate the
	# overall exit code. (Create fs error is a real failure, not a degrade -
	# it keeps a non-zero code in the partition-failure family above.)
	'FS param probe warning': 0,
	# A FIFO, socket or device node: not copied (a note, rc 0); a move (-rm) leaves it in place.
	'Special file skipped': 0,
	# A copy that may not fit, tried anyway as cp tries (_space_warning): the copy's own result decides the exit code.
	'Space warning': 0,
	# -rme met another filesystem below the destination while removing (one mountinfo did not list): left alone.
	'Other filesystem left': 0,
	# A move left source entries the copy did not list (written during the move, special files): a note, rc 0.
	'Move left entries': 0,
	'FS param build warning': 0,
	'FS param warning': 0,
}
RETURNCODE_TO_ERROR_TABLE = {v: k for k, v in ERROR_TO_RETURNCODE_TABLE.items()}

REMOVE_FILES_WHILE_LISTING = True

# set limit to 1m files
SINGLE_PASS_FILE_LIMIT = 1_000_000

#%% ---- Helper Functions ----
# Whether this run shows a progress bar (not under -v, which turns it off): a pool task's error line may then land on the
# bar's line (_bar_line_start). A run setting, handed to the workers (_TASK_SETTINGS).
_BAR = False
# Whether the parent's progress bar drew a frame since its line last ended (Adaptive_Progress_Bar sets it, stop() clears it).
_BAR_DRAWN = False
_LINE_START = [None, '']  # (pid, _line_start's answer)

def _line_start():
	"""What starts an error line that may land on a progress bar's line, once per process: '\\033[2K\\r' when stderr is a
	terminal (it clears the bar's line and starts there, also where the bar reaches the terminal through | tee); '\\n' when
	stdout and stderr are one file (> log 2>&1), which ends the bar's line; else nothing (no bar is in stderr's file)."""
	pid = os.getpid()
	if _LINE_START[0] != pid:
		start = ''
		with contextlib.suppress(Exception):
			if sys.stderr.isatty():
				start = '\033[2K\r'
			else:
				out, err = os.fstat(sys.stdout.fileno()), os.fstat(sys.stderr.fileno())
				if (out.st_dev, out.st_ino, out.st_rdev) == (err.st_dev, err.st_ino, err.st_rdev):
					start = '\n'
		_LINE_START[:] = [pid, start]
	return _LINE_START[1]

def _bar_line_start():
	"""_line_start where a progress bar may hold the line: in a pool task of a run that shows one, or in the parent while its
	bar has drawn a frame since its line last ended; else nothing."""
	return _line_start() if (_IN_TASK and _BAR) or _BAR_DRAWN else ''

def eprint(*args, **kwargs):
	global ERRORS
	out = kwargs.get('file', sys.stderr)
	start = _bar_line_start() if out is sys.stderr else ''
	# What stdout holds goes first: a redirected stdout is buffered, and an error printed past it lands mid-line (after a
	# progress bar's carriage return).
	try:
		sys.stdout.flush()
	except Exception:
		pass
	try:
		# One write, flushed: the line start and the line land together, and at once. 3.6 to 3.8 buffer a redirected stderr,
		# and a line left in its buffer lands later, after the bar frames drawn meanwhile and on the last one's line.
		out.write(start + kwargs.get('sep', ' ').join(map(str, args)) + kwargs.get('end', '\n'))
		out.flush()
	except Exception as e:
		print(f"Error: Cannot print to stderr: {e}")
		print(*args, **kwargs)
	stringRep = ' '.join(map(str,args))
	if ':' in stringRep:
		ERRORS.append(stringRep)

def get_rc_from_error():
	global ERRORS
	global ERROR_TO_RETURNCODE_TABLE
	if not ERRORS or len(ERRORS) == 0:
		print('Done.')
		rc = 0
	else:
		# Unknown errors must fail too; only explicitly registered warnings are 0.
		rc = max((ERROR_TO_RETURNCODE_TABLE.get(error.partition(':')[0], 128) for error in ERRORS), default=0)
		print(f'Error occurred: {ERRORS}, return code: {rc}.')
	ERRORS.clear()
	return rc

def _is_failure(error):
	"""True for a recorded error whose prefix has a non-zero return code (unregistered ones count: 128)."""
	return ERROR_TO_RETURNCODE_TABLE.get(error.partition(':')[0], 128) != 0

def _validate_paths(src, dests):
	"""Validate path lengths (4096 char limit). Returns filtered dest list, or None if invalid."""
	if len(src) > 4096:
		eprint(f'Source path length error: {src} is too long')
		return None
	valid = []
	for d in dests:
		if len(d) > 4096:
			eprint(f'Destination path length error: {d} is too long')
		else:
			valid.append(d)
	if not valid:
		eprint(f'All destination paths are too long: Skipping {src}')
		return None
	return valid

def _print_operation_summary(apb, elapsed, unit='file', show_bandwidth=False):
	print(f"\nTime taken:             {elapsed:0.4f} seconds")
	print(f"Average speed:          {format_bytes(apb.size_counter / elapsed)}B/s")
	print(f"                        {apb.item_counter / elapsed:.2f} {unit}/s")
	if show_bandwidth:
		print(f"Average bandwidth:      {format_bytes(apb.size_counter / elapsed * 8,use_1024_bytes=False)}bps")

class Adaptive_Progress_Bar:
	def __init__(self, total_count = 0, total_size = 0,refresh_interval = 0.1,last_num_job_for_stats = 10,custom_prefix = None,
			  custom_suffix = None,process_word = 'Processed', suppress_all_output = False,bytes_rate_limit = 0,files_rate_limit = 0):
		self.total_count = total_count
		self.total_size = total_size
		self.refresh_interval = refresh_interval
		self.item_counter = 0
		self.size_counter = 0
		self.scheduled_jobs = 0
		self.stop_flag = suppress_all_output
		self.process_word = process_word
		self.start_time = time.monotonic()
		self.last_n_jobs = deque(maxlen=last_num_job_for_stats)
		self.custom_prefix = custom_prefix
		self.custom_suffix = custom_suffix
		self.last_call_args = None
		self.quiet = suppress_all_output
		self.bytes_rate_limit = bytes_rate_limit
		self.files_rate_limit = files_rate_limit
	def print_progress_thread(self):
		while not self.stop_flag:
			# sleep for refresh_interval
			time.sleep(self.refresh_interval)
			self.print_progress()
	def print_progress(self):
		if self.quiet:
			return
		if self.total_count == self.total_size == self.item_counter == self.size_counter == 0:
			return
		if self.total_count > 0 and self.item_counter >= self.total_count:
			self.stop_flag = True
		# job: (num_files, cpSize, cpTime, files_per_job)
		if not self.custom_prefix and not self.custom_suffix:
			job_count = len(self.last_n_jobs)
			if job_count > 0:
				last_n_files_sum, last_n_size_sum, last_n_time_sum , files_per_job_sum= [sum(x) for x in zip(*self.last_n_jobs)]
				files_per_job = files_per_job_sum / job_count
				last_n_time = last_n_time_sum / job_count
				last_n_file_speed = last_n_files_sum / last_n_time if last_n_time_sum > 0 else 0
				last_n_size_speed = last_n_size_sum / last_n_time if last_n_time_sum > 0 else 0
			total_time = time.monotonic() - self.start_time
			total_file_speed = self.item_counter / total_time
			total_size_speed = self.size_counter / total_time
			if total_file_speed:
				remaining_time = (self.total_count - self.item_counter) / total_file_speed
			elif total_size_speed:
				remaining_time = (self.total_size - self.size_counter) / total_size_speed
			else:
				remaining_time = '∞'
		if self.custom_prefix:
			prefix = self.custom_prefix
		else:
			prefix = f'{format_bytes(self.item_counter,use_1024_bytes=False,to_str=True)}F ({format_bytes(self.size_counter)}B) {self.process_word} |'
			#if self.scheduled_jobs:
			prefix += f' {self.scheduled_jobs} Scheduled'
			if job_count > 0:
				prefix += f' {files_per_job:0>3.1f} F/Job '
		if self.custom_suffix:
			suffix = self.custom_suffix
		else:
			suffix = f'{format_bytes(total_size_speed)}B/s {format_bytes(total_file_speed,use_1024_bytes=False,to_str=True)}F/s |'
			if self.bytes_rate_limit > 0 or self.files_rate_limit > 0:
				if self.bytes_rate_limit > 0:
					suffix += f' {format_bytes(self.bytes_rate_limit)}B/s Limit |'
				if self.files_rate_limit > 0:
					suffix += f' {format_bytes(self.files_rate_limit,use_1024_bytes=False,to_str=True)}F/s Limit |'
			elif job_count > 0:
				suffix += f' {last_n_time:.1f}s: {format_bytes(last_n_size_speed)}B/s {format_bytes(last_n_file_speed,use_1024_bytes=False,to_str=True)}F/s |'
			suffix += f' {format_time(remaining_time)}'
		callArgs = (self.item_counter, self.total_count, prefix, suffix)
		if callArgs != self.last_call_args:
			global _BAR_DRAWN
			self.last_call_args = callArgs
			multiCMD.print_progress_bar(*callArgs)
			# The frame leaves its line open for the next error line to start off, but the frame at the total ends it
			# (print_progress_bar's newline).
			_BAR_DRAWN = not (self.total_count > 0 and self.item_counter == self.total_count)
	def stop(self):
		global _BAR_DRAWN
		self.stop_flag = True
		_BAR_DRAWN = False
	def update(self, num_files, cpSize, cpTime= 0 , files_per_job=1):
		self.item_counter += num_files
		self.size_counter += cpSize
		self.last_n_jobs.append((num_files, cpSize, cpTime, files_per_job))
	def under_rate_limit(self):
		if self.total_count == self.item_counter:
			return True
		if not self.bytes_rate_limit and not self.files_rate_limit:
			return True
		duration = time.monotonic() - self.start_time
		if self.bytes_rate_limit > 0:
			estimated_max_current_copy_size = self.bytes_rate_limit * duration
			if self.size_counter > estimated_max_current_copy_size:
				return False
		if self.files_rate_limit > 0:
			estimated_max_current_copy_files = self.files_rate_limit * duration
			if self.item_counter > estimated_max_current_copy_files:
				return False
		return True
	def rate_limit(self):
		self.print_progress()
		while not self.under_rate_limit():
			# sleep for 0.1 seconds
			time.sleep(self.refresh_interval)
			self.print_progress()


_binPaths = {}
_badOverrides = {}  # tool name -> the <TOOL>_PATH value check_path ignored
def _bin(program_name):
	"""The program hpcp starts for a tool, through the command runner too: its <TOOL>_PATH override if that is an
	executable, else the one found on PATH, else the bare name (which then fails as a missing command does). A tool
	hpcp did not look up at import is looked up now."""
	if program_name not in _binPaths:
		check_path(program_name)
	return _binPaths.get(program_name, program_name)

def _warn_bad_overrides():
	"""Name, once at the start of a run, every <TOOL>_PATH that is no executable and is ignored: also cp's, which runs
	in the workers, and a tool's that is missing everywhere."""
	for program_name, override in sorted(_badOverrides.items()):
		print(f"Warning: {program_name.upper()}_PATH={override} is not an executable file; using {program_name} from PATH.", file=sys.stderr)

@functools.lru_cache(maxsize=None)
def check_path(program_name):
	"""True when program_name is found, and _binPaths then holds the program to start: its <TOOL>_PATH override (a
	path, or a bare command name as which() takes it), else the one on PATH."""
	override = os.environ.get(program_name.upper() + '_PATH')
	if override:
		# Recorded, not printed: importing hpcp prints nothing. main() names it at the start of a run
		# (_warn_bad_overrides).
		# Every start of the tool uses it, so a stray variable that names an executable file,
		# MOUNT_PATH=/usr/bin/true, makes hpcp run that file as mount.
		resolved = shutil.which(override)
		if resolved is None:
			_badOverrides[program_name] = override
		override = resolved
	program_path = override or shutil.which(program_name)
	if program_path:
		_binPaths[program_name] = program_path
		return True
	return False

_binCalled = {'lsblk', 'losetup', 'sgdisk', 'blkid', 'umount', 'mount','dd','cp', 'xcopy',
			  'truncate', 'df', 'partx', 'fsutil', 'mkfs.ext4', 'mkfs.hfs',
			  'mkfs', 'mkfs.btrfs', 'mkfs.xfs', 'mkfs.f2fs', 'mkfs.ntfs', 'mkfs.vfat', 'mkfs.exfat', 'mkfs.hfsplus',
			  'mkudffs', 'mkfs.jfs', 'mkfs.reiserfs', 'newfs', 'mkfs.bfs', 'mkfs.minix', 'mkswap',
			  'e2fsck', 'btrfs', 'xfs_repair', 'xfs_info', 'fsck.f2fs', 'dump.f2fs', 'ntfsfix', 'ntfsinfo',
			  'fsck.fat', 'fsck.exfat', 'dump.exfat', 'fsck.hfsplus',
			  'fsck.hfs', 'fsck.jfs', 'fsck.reiserfs', 'fsck.ufs', 'fsck.minix',
			  'tune2fs', 'dumpe2fs', 'xfs_admin', 'exfatlabel', 'udflabel', 'udfinfo', 'jfs_tune',
			  'reiserfstune', 'debugreiserfs', 'swaplabel', 'udevadm', 'e2label', 'fatlabel', 'ntfslabel', 'btrfstune', 'f2fslabel'}
[check_path(program) for program in _binCalled]

# The environment of a command whose output hpcp parses or matches against text. Tools translate section names
# and messages (xfs_info in pl_PL prints metadane/dane/nazwy, mke2fs its 'Invalid filesystem option set'), and
# hpcp looks for the English ones. multiCMD sets it for that command alone: it never reaches another command,
# and the user's locale stays for every command whose output is shown or not read.
_C_LOCALE = {'LC_ALL': 'C'}

def run_command_in_multicmd_with_path_check(command, timeout=...,max_threads=1,quiet=False,strict=False,extra_env=None,pass_fds=()):
	"""run_commands_in_multicmd_with_path_check for one command: its stdout lines."""
	return run_commands_in_multicmd_with_path_check([command], timeout=timeout, max_threads=max_threads, quiet=quiet, strict=strict, extra_env=extra_env, pass_fds=pass_fds)[0]

def run_commands_in_multicmd_with_path_check(commands, timeout=...,max_threads=1,quiet=False,strict=False,extra_env=None,pass_fds=()):
	"""
	Run commands (lists, or strings split on whitespace) with multiCMD, each tool as _bin finds it.

	A missing tool, a timeout (COMMAND_TIMEOUT unless timeout is given) and a non-zero exit are recorded in ERRORS
	(the exit code) and printed unless quiet (a missing tool always is); e2fsck's exit 1-3 is a success. strict raises
	on each of them. extra_env: variables set for these commands only, on top of the inherited ones (a None value
	removes one); pass _C_LOCALE for a tool whose output is parsed. pass_fds: descriptors the commands inherit, at the same
	numbers (multiCMD 1.59, as subprocess's pass_fds).

	Returns:
		list: The stdout lines of each command.
	"""
	if timeout is ...:
		timeout = COMMAND_TIMEOUT
	newCommands = []
	tools = []
	for command in commands:
		command = command.split() if isinstance(command, str) else list(command)
		if not command:
			eprint("Command is empty: The input command is empty")
			continue
		tool = str(command[0])
		if tool not in _binPaths and not check_path(tool):
			eprint(f"Command not found: Command '{tool}' not found. Please consider installing it then retry.")
			if strict:
				raise RuntimeError(f"Command not found: Command '{tool}' not found. Please consider installing it then retry.")
		# judged as the tool the caller named: an override may be a program of any other name
		tools.append(os.path.basename(tool))
		newCommands.append([_bin(tool)] + command[1:])
	tasks = multiCMD.run_commands(newCommands, timeout=timeout, max_threads=max_threads, quiet=quiet, return_object=True, extra_env=extra_env, pass_fds=pass_fds)
	for tool, task in zip(tools, tasks):
		# multiCMD appends 'Timeout!' and then sends SIGINT, so whatever the tool prints while dying lands after it
		if task.returncode == 124 and any(line.strip() == 'Timeout!' for line in task.stderr):
			error = f"Command timeout: Command '{task.command}' timed out after {timeout} seconds."
		# e2fsck encodes "errors corrected" as bits 0 and 1 (exit 1, 2, or 3): a live clone's journal replay is exit 1,
		# a success, not a failed delayed UUID rewrite
		elif task.returncode != 0 and not (tool == 'e2fsck' and task.returncode in (1, 2, 3)):
			# What the tool said is the reason (a full disk, a permission): the last lines, as a tool may print a lot.
			said = ' '.join(line.strip() for line in task.stderr[-3:] if line.strip())
			error = f"Task return code error: Command '{task.command}' failed with return code {task.returncode}" + (f": {said}" if said else '.')
		else:
			continue
		if not quiet:
			eprint(error)
		else:
			ERRORS.append(error)
		if strict:
			raise RuntimeError(error)
	return [task.stdout for task in tasks]

def get_free_space_bytes(path):
	try:
		st = os.statvfs(path)
	except OSError:
		# missing, or below a directory the runner cannot search (EACCES) or a file (ENOTDIR): the nearest parent that
		# answers, as 9.86 asked; the copy then meets the real error and reports it as a failed copy
		st = os.statvfs(get_last_existing_parent(path))
	return st.f_bavail * st.f_frsize  # available blocks * fragment size

# How long `cp --version` (and the held-file probe, _held_route) may stay silent: it answers in milliseconds, and one that
# hangs this long under a host's load gets asked again at the next call (cp_supports_sparse).
_CP_ASK_TIMEOUT = 10

# multiCMD's codes for a command that did not end by itself: timed out (124), handed nothing to run (125), could not start
# (126, 127), killed (137, -1). cp gave no answer then.
_NO_ANSWER = frozenset((124, 125, 126, 127, 137, -1))

_SPARSE = [None, False]  # (pid, cp_supports_sparse's answer): kept once cp answered, per process, as _HELD

def cp_supports_sparse():
	"""Whether the cp the copy runs (CP_PATH, else `cp` from PATH) accepts --sparse (GNU coreutils does, BSD / macOS cp does
	not): True or False once `cp --version` ended by itself (exit 0: by its text; any other exit: not GNU), kept for the
	process; None when it gave no answer (it timed out or could not start), and the next call asks again. 9.86 kept a
	timeout's False for the process's life (an lru_cache, which a forked worker also inherited)."""
	if os.name != 'posix':
		return False
	pid = os.getpid()
	if _SPARSE[0] != pid:
		try:
			task = multiCMD.run_commands([[_bin('cp'), '--version']], timeout=_CP_ASK_TIMEOUT, quiet=True, return_object=True,
				extra_env=_C_LOCALE)[0]
		except Exception:
			return None
		if task.returncode in _NO_ANSWER:
			return None
		_SPARSE[:] = [pid, task.returncode == 0 and 'GNU' in '\n'.join(task.stdout)]
	return _SPARSE[1]

def _allocated_size(st):
	"""The bytes a file takes on its disk, from its stat: its apparent size where the platform gives no other."""
	# stat_result is a tuple: `'st_blocks' in st` would compare against field values, not names
	if hasattr(st, 'st_rsize'):
		return st.st_rsize
	if hasattr(st, 'st_blocks'):
		return st.st_blocks * 512
	return st.st_size

def get_file_size(path, dir_fd=None):
	"""
	Get the on-disk (allocated) size of a file, falling back to the apparent size.
	Use os.path.getsize() where the apparent size is needed, e.g. for content comparison.
	dir_fd: path is a name relative to that directory's descriptor.
	"""
	how = {} if dir_fd is None else {'dir_fd': dir_fd}
	try:
		realSize = _allocated_size(os.stat(path, follow_symlinks=False, **how))
	except Exception:
		try:
			realSize = os.stat(path, **how).st_size
		except Exception:
			realSize = 0
	return realSize

# The steps a filesystem keeps a modification time in, coarsest first: vfat 2 s, ext3 and HFS+ 1 s, exFAT 10 ms, 1 ms, UDF
# 1 us, NTFS and SMB 100 ns, ext4, xfs and btrfs 1 ns (each as a copy onto it reads back).
_MTIME_STEPS_NS = (2 * 10**9, 10**9, 10**7, 10**6, 10**3, 100, 1)

# vfat keeps no time before 1980-01-01 (FAT dates start there): a copy of an older file lands at or after it.
_FAT_TIME_FLOOR_NS = 315532800 * 10**9

def _mtime_step_ns(mtime_ns):
	"""The step a modification time (integer nanoseconds) was kept in, guessed as the coarsest of _MTIME_STEPS_NS that divides
	it. A finer time that divides by a coarser step by chance (1 NTFS time in 10,000 divides by 1 ms) only widens the
	tolerance. Integers, as a float near 1.7e9 s carries about 2.4e-7 s of noise and can show no step finer than 1 s."""
	for step in _MTIME_STEPS_NS:
		if mtime_ns % step == 0:
			return step
	return 1

def _mtimes_agree(first_ns, second_ns):
	"""Two modification times (integer nanoseconds, st_mtime_ns) are one as a copy keeps them: a filesystem truncates or
	rounds a time to its step, so a copy's time lies less than one step of the coarser time away (_mtime_step_ns). And a
	time set through float seconds (os.utime with st_mtime, as many Python tools do) lands within half a float step of
	where it was, so one float step at that time is the least difference: 238 ns from 2004 to 2038."""
	float_step_ns = 2.0 ** (math.frexp(max(abs(first_ns), abs(second_ns)) / 1e9)[1] - 53) * 1e9
	return abs(first_ns - second_ns) <= max(_mtime_step_ns(first_ns), _mtime_step_ns(second_ns), float_step_ns)

def _time_stored_as(stored_ns, wanted_ns):
	"""Whether a destination that reads back stored_ns holds wanted_ns as well as it can: within the step it keeps (its own
	time's, _mtime_step_ns) or one float step (_mtimes_agree). Not within the wanted time's step: a source time on an even
	second (vfat's) is no reason to leave a finer destination's time 1.5 s off it."""
	float_step_ns = 2.0 ** (math.frexp(max(abs(stored_ns), abs(wanted_ns)) / 1e9)[1] - 53) * 1e9
	return abs(stored_ns - wanted_ns) <= max(_mtime_step_ns(stored_ns), float_step_ns)

def get_last_existing_parent(path):
	"""
	Get the last existing parent directory of the given path.

	Args:
		path (str): The path to check.

	Returns:
		str: The last existing parent directory.
	"""
	current_path = os.path.realpath(path)
	while not os.path.exists(current_path):
		parent_path = os.path.dirname(current_path)
		if current_path == parent_path:
			return ''
		current_path = parent_path
	return current_path

def get_fs_type_from_path(path):
	"""
	Get the filesystem type of the given path.

	Args:
		path (str): The path to check.

	Returns:
		str: The filesystem type of the path.
	"""
	try:
		if not os.path.exists(path):
			path = get_last_existing_parent(path)
		if os.name == 'nt':
			# Windows
			drive, _ = os.path.splitdrive(os.path.abspath(path))
			fs_type = run_command_in_multicmd_with_path_check(["fsutil", "fsinfo", "volumeinfo", drive],quiet=True)
			for line in fs_type:
				if 'File System Name' in line:
					return line.partition(':')[-1].strip()
			return ''
		else:
			fs_type = run_command_in_multicmd_with_path_check(["df", "-PT", path],quiet=True,extra_env=_C_LOCALE)[-1].split()[1]
		return fs_type
	except Exception:
		return ''

def expand_paths(paths):
	"""
	Expand a list of paths with glob patterns.

	Args:
		paths (list): The list of paths to expand (a single str is one pattern, not a list of characters).

	Returns:
		list: The list of expanded paths.
	"""
	if isinstance(paths, str):
		paths = [paths]
	output_list = []
	for src_pattern in paths:
		src_pattern = os.path.expanduser(src_pattern)
		if os.name != 'nt':
			try:
				output_list.extend(glob.glob(src_pattern,include_hidden=True,recursive=True))
			except Exception:
				output_list.extend(glob.glob(src_pattern,recursive=True))
		else:
			output_list.append(src_pattern)
	return output_list

def _named_sources(src_path, may_name_destination):
	"""(sources as expand_paths gives them, the last word when it is held back). A pattern that matches nothing is kept as the
	name it is: verify_src_path then reports it as it does a list entry, and a file whose name holds glob characters is that
	file. may_name_destination: the last of several may be the destination (get_dest_from_path), which need not exist, so a
	last word that matches nothing is held back until that is known."""
	patterns = [src_path] if isinstance(src_path, str) else list(src_path)
	named, held = [], None
	for index, pattern in enumerate(patterns):
		found = expand_paths([pattern])
		if not found and may_name_destination and len(patterns) > 1 and index == len(patterns) - 1:
			held = pattern
		elif not found:
			found = [os.path.expanduser(pattern)]
		named.extend(found)
	return named, held

# What a person, a shell or a service manager stops a run with. Each turns into the cleanup path of Ctrl-C (_stop).
_STOP_NAMES = ('SIGINT', 'SIGTERM', 'SIGHUP')
# hpcp's own signal: a worker's parent-death watch sends it to the worker's main thread (neither hpcp nor multiCMD uses it).
_WATCH_SIGNAL = getattr(signal, 'SIGUSR2', None)
# The stops and the watch's signal: blocked while a pool's worker is started and until it has set itself up (_pool.submit,
# _init_worker), and while a directory is asked about a name (_ask_names).
_STOPS = [getattr(signal, name) for name in _STOP_NAMES + ('SIGUSR2',) if hasattr(signal, name)]
# The signal that stopped this process, once one did (_stop), and that process: a worker forked after it has both.
_STOPPED = None
_STOPPED_PID = None
# True in a worker while it runs a task (_run_task): a stop then ends the task by its cleanup path, else the worker at once.
_IN_TASK = False
# The process that runs main(): a stop there runs the cleanup path; any other process with _stop is a pool worker (forked
# with it, or set up with it, _init_worker), where a stop outside a task ends the worker.
_MAIN_PID = None
# True once this process is ending (_end_by): a stop then changes nothing.
_ENDING = False
# Twice multiCMD's longest stop of a command (0.01 s, up to 1 s, SIGKILL, up to 1 s more): the watch's grace before os._exit;
# after a stop, how long a pool's worker between tasks may take to end before it is killed (_pool); and how long a pool may
# stay silent after one of its workers died before the run ends (_pool.watch).
_WATCH_GRACE = 4
# Pool regions this process is in (_pool, _looked_at). Inside one the parent's _stop only records its stop, and the region's
# loop acts on it within 1 s (_check_stop): raised anywhere in there, its KeyboardInterrupt could leave a lock of Python's
# pool code held (concurrent.futures.wait's), and the pool's shutdown would wait on it for good.
_DEFER = 0
# The parent has acted on its stop (_check_stop raised for it, or hpcp() took its KeyboardInterrupt). Not set where _stop
# raises: Python drops a KeyboardInterrupt raised in a weakref callback or a __del__, and the next check acts on it then.
_STOP_ACTED = False
# The parent has said once that a later stop is ignored while it cleans up (_stop).
_STOP_NOTED = False
# A pool, or _looked_at's threads, was left with something of it still running or stuck: main() ends by os._exit, not by the
# interpreter's exit, which joins Python's pool threads.
_ABANDONED = False
# The pools left so, kept for the process's life: Python 3.6 wakes a collected pool's manager thread by a write into its
# result queue, under a lock a killed worker may hold, and the parent would wait on it for good.
_LEFT = []
# A future of the pool in use failed with BrokenProcessPool (_task_result): that pool ends the bounded way (_pool).
_POOL_BROKEN = False
# In a pool's worker: the pool's slots, (lock, a (pid, in a task) pair for each worker it can start), and the index of this
# worker's state there (_claim_slot); None without one. The parent reads the states after a stop (_pool).
_SLOTS = None
_MY_SLOT = None

def _end_by(signum):
	"""End this process by signum with its default action, as a process that never handled it would end. From its first
	line every stop is ignored (_ENDING): in a worker, one landing in the flush would raise SystemExit out of its ending
	(_stop), and the pool's worker loop would keep it (a Ctrl-C, its pass-on and the pool's own SIGTERM come close together)."""
	global _ENDING
	_ENDING = True
	try:
		sys.stdout.flush()
		sys.stderr.flush()
	except Exception:
		pass
	# Blocked here while the action changes: a stop landing inside signal.signal, between Python's check for pending ones and
	# the switch (a worker gets a Ctrl-C twice, from the terminal and passed on by the pool), would find the default action
	# already Python's, which prints 'Signal 2 ignored due to race condition' with a traceback (3.10+; 3.6 to 3.9 raise
	# TypeError out of here). Blocked, it waits, and the default action ends the process once it is let in.
	held = _block_stops()
	signal.signal(signum, signal.SIG_DFL)
	os.kill(os.getpid(), signum)
	if held is not None:
		signal.pthread_sigmask(signal.SIG_UNBLOCK, [signum])
	# A default action that does not end the process (none of ours): end anyway.
	os._exit(128 + signum)

def _stop(signum, frame):
	"""The handler for every stop a process can catch: the first raises KeyboardInterrupt, so multiCMD stops cp (in a process
	group of its own, which no group signal reaches) and every finally runs (a temporary file is discarded, mounts are
	cleaned up); later ones return, so a second Ctrl-C cannot cut the cleanup halfway (the parent says so once, naming
	Ctrl-\\). The stop is passed on to this process's pool workers, which a signal to this process alone (kill PID, timeout)
	never reaches. A pool worker between tasks has nothing to clean up and ends at once. In the parent inside a pool region
	(_DEFER) the first stop is only recorded: the region's loop raises it (_check_stop). A file hpcp makes is held by C before
	this can run (_create), so a stop that raises there leaves nothing the caller's cleanup cannot find."""
	global _STOPPED, _STOPPED_PID, _STOP_NOTED
	if _ENDING:
		return
	if not _IN_TASK and os.getpid() != _MAIN_PID:
		if _STOPPED_PID == os.getpid():
			# Stopped already, outside a task: on its way out (_run_task's _end_by, whose first lines a stop raising here would
			# cut, and the pool's loop would keep the worker, idle for good), or kept by the pool's loop. Either way it ends here.
			_end_by(_STOPPED)
			return
		# A pool worker outside a task (idle, starting, between tasks): nothing to clean up, and any stop ends it. Kept, it
		# would wait in the pool's queue for good, and the pool's shutdown with it. SystemExit unwinds through the queue's
		# locks, letting go of the one it holds while it waits (killed holding it, the other workers could never read the
		# pool's shutdown), and multiprocessing ends a worker on it without a traceback. At a task's start or end, _run_task
		# ends the worker on it (_STOPPED). Every such stop raises while the pool's worker loop runs: it can take one for a
		# task's error, or (3.7+) a result's, and go on. Outside that loop (starting before it, or ending after it: threading's
		# shutdown, multiprocessing's exit work, where Python prints a raise as 'Exception ignored' with its traceback) there
		# is nothing of the pool's to let go of, and the worker ends at once, quietly.
		if _STOPPED_PID != os.getpid():
			_STOPPED, _STOPPED_PID = signum, os.getpid()
		if not _in_pool_loop(frame):
			_end_by(_STOPPED)
		raise SystemExit(128 + signum)
	if _STOPPED_PID == os.getpid():
		if os.getpid() == _MAIN_PID and not _STOP_NOTED:
			# One write, never print: a handler printing while the main thread prints raises 'reentrant call'.
			_STOP_NOTED = True
			_say('Stopping: cleaning up first (cp stopped, temporary files removed, images unmounted); Ctrl-\\ ends hpcp now, '
				'which may leave them')
		return
	_STOPPED, _STOPPED_PID = signum, os.getpid()
	_pass_stop_on(signum)
	if _DEFER and os.getpid() == _MAIN_PID:
		return
	if _DROPPING and _in_dropping_call(frame):
		# Python would drop the raise, and 3.6 and 3.7 print it as 'Exception ignored' with its traceback. Recorded, it is acted
		# on all the same, as on 3.8+ (_check_stop, _run_task).
		return
	raise KeyboardInterrupt

def _dropping_codes():
	"""The code Python runs as an object of a run goes, where a raise is dropped: the weak containers' callbacks (a thread's,
	a pool worker's, an ABC cache's) and multiprocessing's finalizers; a __del__ is found by its name (_in_dropping_call).
	Only where Python prints a dropped raise: 3.6 and 3.7 have no sys.unraisablehook (_install_stops)."""
	if hasattr(sys, 'unraisablehook'):
		return frozenset()
	codes = set()
	for make in (weakref.WeakSet, weakref.WeakValueDictionary, weakref.WeakKeyDictionary):
		with contextlib.suppress(AttributeError):
			codes.add(make()._remove.__code__)
	codes.add(multiprocessing.util.Finalize.__call__.__code__)
	return frozenset(codes)

_DROPPING = _dropping_codes()

def _in_dropping_call(frame):
	"""Whether the code a stop interrupted (frame) runs inside a call whose raise Python drops (_dropping_codes)."""
	while frame is not None:
		if frame.f_code in _DROPPING or frame.f_code.co_name == '__del__':
			return True
		frame = frame.f_back
	return False

# The pool's worker loop (concurrent.futures.process._process_worker, Python 3.6 to 3.14): a worker's stop raises only while
# it runs (_stop). Unknown (a Python that renamed it), or no frame to look at: every stop raises, as in 9.86.
_POOL_LOOP = getattr(getattr(concurrent.futures.process, '_process_worker', None), '__code__', None)

def _in_pool_loop(frame):
	"""Whether the code a stop interrupted (frame) runs inside the pool's worker loop."""
	if _POOL_LOOP is None or frame is None:
		return True
	while frame is not None:
		if frame.f_code is _POOL_LOOP:
			return True
		frame = frame.f_back
	return False

def _say(text):
	"""text as a line of its own on stderr, in one write, after what stdout holds (a progress bar's line ends first)."""
	with contextlib.suppress(Exception):
		sys.stdout.flush()
	with contextlib.suppress(OSError):
		os.write(2, ('\n' + text + '\n').encode(errors='backslashreplace'))

def _check_stop():
	"""In the process that runs main(): raise the KeyboardInterrupt of a stop _stop recorded and has not been acted on, once.
	Inside a pool region _stop only records it; and Python drops one raised in a weakref callback or a __del__ (every Python,
	3.6 to 3.14), where the run would go on, a move to its removal. Called by every pool loop's wait (_wait), and between
	the steps of a run (hpcp()): a cheap test, so it can be called often."""
	global _STOP_ACTED
	if _STOPPED_PID is not None and not _STOP_ACTED and _STOPPED_PID == os.getpid() == _MAIN_PID:
		_STOP_ACTED = True
		raise KeyboardInterrupt

def _pass_stop_on(signum):
	"""Send a stop to this process's pool workers, which a signal to this process alone (kill PID, timeout) never reaches."""
	for child in multiprocessing.active_children():
		try:
			os.kill(child.pid, signum if signum != _WATCH_SIGNAL else signal.SIGTERM)
		except OSError:
			pass

def _install_stops(signals=_STOP_NAMES):
	"""_stop for each of signals that this process was not started ignoring: nohup starts a run with SIGHUP ignored, a
	background job of a non-interactive sh with SIGINT ignored, and both must stay so, as Python itself keeps an ignored
	SIGINT. Only from the main thread (signal.signal). And (Python 3.8+) this process's own stop that Python drops, raised
	in a weakref callback or a __del__, is not reported as 'Exception ignored' with its traceback: it is acted on all the
	same (_check_stop, _run_task)."""
	for name in signals:
		signum = getattr(signal, name, None)
		if signum is not None and signal.getsignal(signum) is not signal.SIG_IGN:
			signal.signal(signum, _stop)
	hook = getattr(sys, 'unraisablehook', None)
	if hook is not None and getattr(hook, '__name__', '') != '_unraisable':
		def _unraisable(unraisable, default=hook):
			if not (isinstance(unraisable.exc_value, (KeyboardInterrupt, SystemExit)) and _STOPPED_PID == os.getpid()):
				default(unraisable)
		sys.unraisablehook = _unraisable

def _on_worker_start():
	# The process that started the pool: under forkserver (Python 3.14's default) and spawn the worker's own parent is the
	# fork server, which outlives it, so the main process is watched by its sentinel (Python 3.8+); else by getppid.
	main = getattr(multiprocessing, 'parent_process', lambda: None)()
	# The process that forked this worker (its _MAIN_PID, under fork): getppid now could already be init's, its parent gone
	# before it ran.
	parent = _MAIN_PID or os.getppid()
	def parent_alive():
		return main.is_alive() if main is not None else os.getppid() == parent
	_install_stops()
	watch = _WATCH_SIGNAL is not None and hasattr(signal, 'pthread_kill')
	if watch:
		# hpcp's own: always handled, so the watch below reaches the cleanup whatever the stops' dispositions are.
		signal.signal(_WATCH_SIGNAL, _stop)
	main_thread = threading.main_thread().ident
	def watch_parent():
		# every 0.5s, check if parent still exists
		while True:
			if not parent_alive():
				if watch:
					# The main thread stops its cp and discards its temporary file (_stop), then ends; the grace is for that.
					signal.pthread_kill(main_thread, _WATCH_SIGNAL)
					time.sleep(_WATCH_GRACE)
				os._exit(1)
			time.sleep(0.5)
	t = threading.Thread(target=watch_parent, daemon=True)
	t.start()

_worker_initialized = False
def _init_worker():
	"""Set this worker up, once: the stops and the parent watch (_on_worker_start), stdout, its slot (_claim_slot); then let
	the stops in. A pool worker starts with them blocked (_pool.submit blocks them while it may start one, and a fork server
	keeps the mask it started with for every worker it forks), so a stop that came meanwhile reaches hpcp's handler only now
	and ends the worker without a traceback (Python's own SIGINT handler raises one there under forkserver and spawn)."""
	global _worker_initialized
	if not _worker_initialized:
		_on_worker_start()
		# A worker that did not fork from the parent (forkserver, the default from Python 3.14) has a new stdout.
		_escape_unencodable_stdout()
		_claim_slot()
		_worker_initialized = True
		_unblock_stops()

def _unblock_stops():
	"""Let this worker's stops in. One pending since its start runs _stop here, which ends the worker (outside a task it
	raises SystemExit): caught, so nothing raises out of a pool's initializer (concurrent.futures prints a traceback for one
	that raises) or out of multiprocessing's after-fork hook."""
	if hasattr(signal, 'pthread_sigmask'):
		try:
			signal.pthread_sigmask(signal.SIG_UNBLOCK, _STOPS)
		except (KeyboardInterrupt, SystemExit):
			_end_by(_STOPPED or signal.SIGTERM)

def _block_stops():
	"""Block the stops in this thread: the old mask, for _restore_stops; None where there is no pthread_sigmask (Windows)."""
	if hasattr(signal, 'pthread_sigmask'):
		return signal.pthread_sigmask(signal.SIG_BLOCK, _STOPS)
	return None

def _restore_stops(mask):
	if mask is not None:
		signal.pthread_sigmask(signal.SIG_SETMASK, mask)

def _claim_slot():
	"""Take this worker's (pid, in a task) pair in its pool's slots, the first free one. Under the slots' lock, taken within
	1 s (as _claim_space: far longer than a scan of them takes; a holder that died must not hang the worker). Without one
	(no slots, all taken, no lock) the worker runs on, and the parent counts it as always in a task: it is never killed."""
	global _MY_SLOT
	share = _SLOTS
	if share is None:
		return
	lock, slots = share
	if not lock.acquire(timeout=1):
		return
	try:
		for index in range(0, len(slots), 2):
			if slots[index] == 0:
				slots[index + 1] = 0
				slots[index] = os.getpid()
				_MY_SLOT = index + 1
				return
	finally:
		lock.release()

def _slot_state(in_task):
	"""Write this worker's state into its slot: one aligned 8-byte store, which the parent reads whole."""
	if _MY_SLOT is not None:
		_SLOTS[1][_MY_SLOT] = 1 if in_task else 0

def _pool_worker_start(slots, space_array, rate_share):
	"""A pool's initializer (Python 3.7+, every start method): the pool's slots and shares, then the worker's setup
	(_init_worker; under fork multiprocessing's after-fork hook has run it already, with the slots it inherited). A stop
	from here on ends the worker; nothing raises out of an initializer."""
	global _SLOTS
	try:
		if slots is not None:
			_SLOTS = (slots.get_lock(), slots.get_obj())
		if space_array is not None:
			_init_space_share(space_array)
		if rate_share is not None:
			_init_rate_share(*rate_share)
		_init_worker()
	except (KeyboardInterrupt, SystemExit):
		_end_by(_STOPPED or signal.SIGTERM)

def _register_worker_start():
	"""multiprocessing runs its after-fork hooks in each child it starts, before the child's work (a pool worker's loop):
	the worker sets itself up there (_init_worker), not at its first task, so one that never runs a task, or that its pool
	never counted (a stop landing as it was forked), still has its watch and ends when its parent does. main() registers it
	(not an import: a program that imports hpcp must not get hpcp's stops and parent watch in its own multiprocessing
	children). Under fork the registry is inherited, so every pool worker of the run sets up at its start (Python 3.6 has
	no other way); forkserver and spawn workers, which start from a new interpreter, set up in the pool's initializer
	(_pool_worker_start, Python 3.7+). The registry holds its key weakly: a module-level function lives as long as the module."""
	multiprocessing.util.register_after_fork(_init_worker, lambda init: init())

# The run settings hpcp() keeps in module globals. A pool worker that did not fork from the parent
# (spawn, forkserver) would otherwise run with the import-time defaults.
# The destinations the operator named (-d), absolute: what is followed and written through, as cp follows its destination
# argument; below them nothing is (_landing_of, _dest_directory). RUN_ID: one hpcp() run, so the descriptors a process's door
# holds (_dest_directory) never outlive it.
NAMED_DESTS = frozenset()
RUN_ID = None
# The named destinations that are directories when the run's copy starts, as named (a link followed): {path (absolute):
# (st_dev, st_ino)}, the run's pin. Each process checks a named destination against it as its door opens it
# (_dest_directory): a worker that opens it later (a fresh one: forkserver, Python 3.14's default on Linux; spawn, macOS's)
# never follows a swap of it for a link to another directory. One the run makes later is pinned by each process's open.
NAMED_IDS = {}
_TASK_SETTINGS = ('NAMED_DESTS', 'RUN_ID', 'NAMED_IDS', 'CONTENT_ONLY', 'WRITE_BESIDE', 'READ_FIRST', 'NO_CREATE_DIR', 'SYNC_COPIES', 'RANDOM_DESTINATION_SELECTION', 'HASH_SIZE', 'COMMAND_TIMEOUT',
	'BYTES_RATE_LIMIT', 'FILES_RATE_LIMIT', 'MIRROR_FS_PARAMS', 'REMOVE_FILES_WHILE_LISTING', '_BAR')

def _submit_task(executor, func, *args, **kwargs):
	"""Submit func to a process pool (a _pool) with this process's run settings; collect it with _task_result."""
	settings = {name: globals()[name] for name in _TASK_SETTINGS}
	# What _copy_pool asked once for the run's workers (_held_route): the same cp and the same /proc in every process of it.
	settings['_HELD_GIVEN'] = (_HELD[1],) if _HELD[0] == os.getpid() else None
	return executor.submit(_run_task, settings, func, *args, **kwargs)

def _run_task(settings, func, *args, **kwargs):
	"""Worker side of _submit_task: (result, the errors func recorded, None), or (None, errors, crash text) when it raised.

	ERRORS is process-local, so the errors go back through the future; a reused worker never carries them into its next task.
	"""
	global _IN_TASK
	try:
		if _STOPPED_PID == os.getpid():
			# A stop the pool's worker loop took for an earlier task's error, or (Python 3.7+) its result's: kept, the worker
			# would run this task whole, every later stop ignored as already handled.
			_end_by(_STOPPED)
		_init_worker()
		globals().update(settings)
		error_start = len(ERRORS)
		# In a task from here (the parent never kills a worker in one, _pool) until its cleanup has run.
		_slot_state(True)
		_IN_TASK = True
		try:
			return func(*args, **kwargs), ERRORS[error_start:], None
		except Exception as e:
			return None, ERRORS[error_start:], f'{type(e).__name__}: {e}'
		finally:
			_IN_TASK = False
			del ERRORS[error_start:]
			if _STOPPED_PID == os.getpid():
				# A stop came during the task: its cleanup has run, or the KeyboardInterrupt never got here (Python reports and
				# drops one raised in a weakref callback or a __del__, and the task went on). Either way the worker ends.
				_end_by(_STOPPED)
			# Between tasks: nothing of this one is left (no temporary or torn file, no fsync in flight).
			_slot_state(False)
	except (KeyboardInterrupt, SystemExit):
		if _STOPPED_PID != os.getpid():
			raise
		# A stop (_stop) at the task's start or end: the worker ends as the signal asked. Let through, the pool's worker loop
		# would take it for the task's error and keep the worker (except BaseException, Python 3.6 to 3.14).
		_end_by(_STOPPED)

def _settled(future):
	"""(result, errors, crash) of a done _submit_task future. What it failed with is the worker's, a stop (SystemExit,
	KeyboardInterrupt: the pool's worker loop sends back one raised as it sent a result) included, and becomes a crash; a
	stop of this process's own is raised on. A BrokenProcessPool marks the pool (_POOL_BROKEN)."""
	global _POOL_BROKEN
	try:
		return future.result()
	except Exception as e:
		error = e
	except BaseException as e:
		try:
			theirs = future.exception(timeout=0)
		except Exception:
			theirs = None
		if e is not theirs:
			raise
		error = e
	if isinstance(error, BrokenProcessPool):
		_POOL_BROKEN = True
	return None, [], f'{type(error).__name__}: {error}'

def _task_result(future, default, crash_prefix):
	"""A _submit_task future's result in the parent. The errors the task recorded join ERRORS (the worker printed them);
	a task that raised, or whose worker died (BrokenProcessPool), is recorded under crash_prefix and gives default.
	Either way the other tasks go on (a broken pool then fails the next submit)."""
	result, errors, crash = _settled(future)
	ERRORS.extend(errors)
	if crash is not None:
		eprint(f'{crash_prefix}: {crash}')
		return default
	return result

def _raise_task_failures(future):
	"""A _submit_task future's result, raising its recorded failures (and a crash) as one RuntimeError, as
	_call_raising_recorded_failures does in-process."""
	result, errors, crash = _settled(future)
	failures = [error for error in errors if _is_failure(error)]
	if crash is not None:
		failures.append(crash)
	if failures:
		raise RuntimeError('\n'.join(failures))
	return result

def _call_raising_recorded_failures(func, *args, **kwargs):
	"""Run func; the failures it records in ERRORS are taken out of ERRORS and raised as one RuntimeError."""
	error_start = len(ERRORS)
	try:
		result = func(*args, **kwargs)
		failures = [error for error in ERRORS[error_start:] if _is_failure(error)]
		if failures:
			raise RuntimeError('\n'.join(failures))
		return result
	finally:
		# A reused worker must not carry this task's errors into its next job;
		# an in-process caller reports the raised failures itself.
		del ERRORS[error_start:]

def _new_slots(count):
	"""Slots for a pool of count workers: (the parent's (lock, slots), what the initializer hands a worker), or (None, None)
	where none can be made (its workers then count as always in a task). Forked workers inherit an anonymous shared mapping
	and a lock, which leave no file (nothing to hand: a pool given another start method than the default gets no slots);
	forkserver and spawn ones get an Array through the initializer (as _space_share)."""
	with contextlib.suppress(OSError, ImportError, ValueError, TypeError):
		if multiprocessing.get_start_method() == 'fork':
			import mmap
			return (multiprocessing.Lock(), memoryview(mmap.mmap(-1, 16 * count)).cast('q')), None
		if sys.version_info >= (3, 7):
			array = multiprocessing.Array('q', 2 * count)
			return (array.get_lock(), array.get_obj()), array
	return None, None

def _submit_36(executor, mask, fn, args, kwargs):
	"""ProcessPoolExecutor.submit of Python 3.6 (3.6.15), whose wakeup of the manager thread (_result_queue.put(None)) waits
	for good on the result queue's write lock when a worker ended holding it (stopped inside SemLock.__enter__): here the
	lock is taken in 0.1 s steps (_Done's and _wait's step, ten within the 1 s in which a stop is acted on), and in the process
	that runs main() (mask: the stops are blocked in this thread) each step looks for a stop _stop handles, recorded (_stop
	ran for one another thread took) or pending (every thread blocks it), and raises it before anything is registered. A
	signal ignored (nohup's SIGHUP) stays pending while blocked, and is no stop. A pool that broke meanwhile raises
	BrokenProcessPool, as submit does. The rest is 3.6's own: the work item registered, then the wakeup sent under the lock."""
	global _STOP_ACTED
	results = executor._result_queue
	if results._wlock is None:
		# Windows: 3.6's result queue has no write lock there (a pipe message is written whole), so its submit cannot hang on one
		return executor.submit(fn, *args, **kwargs)
	from concurrent.futures import _base
	from concurrent.futures.process import _WorkItem
	with executor._shutdown_lock:
		if executor._broken:
			raise BrokenProcessPool('A child process terminated abruptly, the process pool is not usable anymore')
		if executor._shutdown_thread:
			raise RuntimeError('cannot schedule new futures after shutdown')
		while not results._wlock.acquire(True, 0.1):
			if mask is not None and (_STOPPED_PID == os.getpid()
					or any(signal.getsignal(s) is _stop for s in signal.sigpending() if s in _STOPS)):
				_STOP_ACTED = True
				raise KeyboardInterrupt
			if executor._broken:
				raise BrokenProcessPool('A child process terminated abruptly, the process pool is not usable anymore')
		try:
			future = _base.Future()
			executor._pending_work_items[executor._queue_count] = _WorkItem(future, fn, args, kwargs)
			executor._work_ids.put(executor._queue_count)
			executor._queue_count += 1
			results._writer.send_bytes(multiprocessing.reduction.ForkingPickler.dumps(None))
		finally:
			results._wlock.release()
		executor._start_queue_management_thread()
		return future

class _pool:
	"""The one door to a process pool: `with _pool(max_workers) as executor:`, submitting with _submit_task(executor, ...) and
	waiting with _wait(futures, executor). Inside, the parent only loops (submits, waits, results, the progress bar): no cp,
	no prompt. A stop is acted on there within 1 s (_DEFER, _check_stop in _wait). Each worker can start (ProcessPoolExecutor
	never starts more than max_workers, and Python 3.6 to 3.14 replace none that died) gets a (pid, in a task) slot.

	Its exit: as concurrent.futures' own when the loop ended normally. After a stop or a broken pool, the stop (SIGTERM for a
	broken pool) is passed on, and the parent waits for the pool's workers, each second: from _WATCH_GRACE after the wait
	began, one between tasks is SIGKILLed (it is stuck on the pool's queue, or on a result half sent, and holds nothing of a
	file), and one in a task, or without a slot, is waited for with no bound, its cleanup never cut, with one line naming
	Ctrl-\\. After any other exception the futures not started are cancelled and the running ones end as they would, then
	the same wait. Either way the pool is then left without a join (_ABANDONED): its manager thread may be stuck for good.
	space: the workers share the run's space-warning slots (_SPACE_WARNED); rate: the -pfl rate share under -F."""
	def __init__(self, max_workers, space=False, rate=None):
		self.max_workers = max_workers
		self.space = space
		self.rate = rate
		self.broken = False

	def __enter__(self):
		global _DEFER, _POOL_BROKEN, _SLOTS, _RATE_SHARE
		self.outer = (_SLOTS, _RATE_SHARE)
		_DEFER += 1
		try:
			# Processes of this run alive before this pool: an outer pool's, never this one's to wait for or kill.
			self.before = {child.pid for child in multiprocessing.active_children()}
			self.seen = set()
			self.gone_at = None
			# The futures not done yet: each leaves by its done callback (run by the pool's manager thread, or by a cancel). No
			# weak references: their callbacks run in the main thread, where a stop's KeyboardInterrupt is dropped.
			self.pending = set()
			_POOL_BROKEN = False
			if sys.version_info < (3, 7):
				_quiet_36_pool_shutdown()
			# The slots' lock and the executor's queues are SemLocks: made with the stops blocked in the process that runs main(),
			# so a semaphore tracker Python 3.6 and 3.7 start for them (forkserver, spawn) starts with them blocked (hpcp()).
			mask = _block_stops() if os.getpid() == _MAIN_PID else None
			try:
				self.slots, handed = _new_slots(self.max_workers)
				# Forked workers (Python 3.6, which has no initializer, and the fork start method) inherit these.
				_SLOTS = self.slots
				if self.rate is not None:
					_RATE_SHARE = self.rate
				kwargs = {}
				if sys.version_info >= (3, 7):
					space = _SPACE_WARNED[2] if self.space and _SPACE_WARNED is not None else None
					kwargs = {'initializer': _pool_worker_start, 'initargs': (handed, space, self.rate)}
				self.executor = ProcessPoolExecutor(max_workers=self.max_workers, **kwargs)
			finally:
				_restore_stops(mask)
		except BaseException:
			_DEFER -= 1
			_SLOTS, _RATE_SHARE = self.outer
			raise
		return self

	def submit(self, func, *args, **kwargs):
		"""executor.submit, with the stops blocked in the process that runs main() (a library caller's are left alone): a
		worker it starts (and a fork server, which every later worker forks from) starts with them blocked, and lets them in
		once set up (_init_worker). A stop meanwhile is recorded at the restore (_stop, inside the region). On Python 3.6 hpcp's
		own copy of its submit (_submit_36), which acts on a stop while a dead worker holds the result queue's lock."""
		# A stop recorded inside the region starts no more work: the loop acts on it here, as at its wait.
		_check_stop()
		mask = _block_stops() if os.getpid() == _MAIN_PID else None
		try:
			if sys.version_info >= (3, 7):
				future = self.executor.submit(func, *args, **kwargs)
			else:
				future = _submit_36(self.executor, mask, func, args, kwargs)
		finally:
			_restore_stops(mask)
		self.pending.add(future)
		future.add_done_callback(self.pending.discard)
		return future

	def _states(self):
		"""{pid: in a task} for the workers that took a slot."""
		states = {}
		if self.slots is not None:
			slots = self.slots[1]
			for index in range(0, len(slots), 2):
				pid = slots[index]
				if pid:
					states[pid] = bool(slots[index + 1])
		return states

	def _alive(self):
		return [child for child in multiprocessing.active_children() if child.pid not in self.before]

	def watch(self, done):
		"""One pass of a pool loop's wait (_wait): a worker that died while nothing completes for _WATCH_GRACE means the pool's
		manager thread is stuck (a pool that sees a death fails every future at once), on a result the dead worker half wrote
		(the parent holds the result pipe's write end too, so the read never ends): the run ends with 191, as a broken pool's.
		Workers never end on their own while a pool runs (no max_tasks_per_child), except on a stop, acted on before this."""
		if done:
			self.gone_at = None
			return
		alive = {child.pid for child in self._alive()}
		self.seen |= alive | set(self._states())
		if not self.seen - alive:
			self.gone_at = None
		elif self.gone_at is None:
			self.gone_at = time.monotonic()
		elif time.monotonic() - self.gone_at >= _WATCH_GRACE:
			self.broken = True
			raise RuntimeError('Copy process crashed: a pool worker died and its pool stopped answering')

	def __exit__(self, kind, error, tb):
		global _DEFER, _ABANDONED, _SLOTS, _RATE_SHARE
		try:
			stopped = kind is not None and issubclass(kind, (KeyboardInterrupt, SystemExit))
			broken = self.broken or _POOL_BROKEN or (kind is not None and issubclass(kind, BrokenProcessPool))
			if kind is None and not broken:
				self.executor.shutdown(wait=True)
				# Let go of it here, inside the region: its parts' weakref callbacks and finalizers run as it goes, where a stop
				# is only recorded (_stop); outside, Python would drop one raised in them.
				self.executor = self.slots = None
				return False
			if not (stopped or broken):
				# A failure in the loop, with no stop: what is not started is cancelled, and what runs (with the calls already
				# queued, which no cancel reaches) ends as it would. A stop meanwhile, or the pool stuck on a dead worker, ends
				# this wait (_wait).
				for future in list(self.pending):
					future.cancel()
				with contextlib.suppress(KeyboardInterrupt, RuntimeError):
					while self.pending:
						_wait(set(self.pending), self)
			# A stop ends each worker: idle ones at once, busy ones (after a stop or a broken pool) by their cleanup. SIGTERM for
			# a broken pool or a failure.
			_pass_stop_on(_STOPPED or signal.SIGTERM)
			self._wait_for_workers()
			# From 3.9 the manager thread is told to end, before main()'s finalizers close the queues it may still close itself.
			# Not before: 3.6's shutdown puts a wakeup into the result queue under a lock a killed worker may hold, and 3.7's and
			# 3.8's close the manager thread's pipes and call queue under it (a traceback from that thread).
			if sys.version_info >= (3, 9):
				self.executor.shutdown(wait=False, cancel_futures=True)
			_ABANDONED = True
			_LEFT.append(self.executor)
			return False
		finally:
			_DEFER -= 1
			_SLOTS, _RATE_SHARE = self.outer

	def _wait_for_workers(self):
		"""Wait for this pool's processes to end, as the class says."""
		started, said = time.monotonic(), False
		while True:
			alive = self._alive()
			if not alive:
				return
			if time.monotonic() - started >= _WATCH_GRACE:
				states = self._states()
				idle = [child for child in alive if states.get(child.pid) is False]
				for child in idle:
					with contextlib.suppress(OSError):
						os.kill(child.pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
				# One deadline for them all: a SIGKILLed process not in an uninterruptible wait exits in milliseconds, 1 s covers a
				# loaded host, and -j 64 must not wait 64 s.
				deadline = time.monotonic() + 1
				for child in idle:
					child.join(max(0, deadline - time.monotonic()))
				alive = [child for child in alive if states.get(child.pid) is not False]
				if alive and not said:
					said = True
					_say(f'Waiting for {len(alive)} worker(s) still finishing a file (pids {", ".join(str(child.pid) for child in alive)}): '
						'Ctrl-\\ ends hpcp now, which may leave a temporary file or a torn file there')
			if alive:
				multiprocessing.connection.wait([child.sentinel for child in alive], timeout=1)

_QUIETED_36 = []

def _quiet_36_pool_shutdown():
	"""Python 3.6's pool manager thread raises queue.Full out of its shutdown_worker when a worker dies while the call queue
	is full (a stop ends the workers), and threading prints its traceback; 3.7's shutdown_worker catches it. The pool is
	broken by then, its futures failed and its workers terminated: the thread just ends, quietly. Once per process."""
	if _QUIETED_36:
		return
	real = concurrent.futures.process._queue_management_worker
	def _queue_management_worker(*args, **kwargs):
		try:
			return real(*args, **kwargs)
		except queue.Full:
			return None
	concurrent.futures.process._queue_management_worker = _queue_management_worker
	_QUIETED_36.append(real)

def _wait(futures, pool):
	"""The one wait of every pool loop: the futures done within 1 s (most loops refresh the progress bar each second), then
	the parent's stop acted on (_check_stop), then the pool watched (_pool.watch). So a stop is acted on within 1 s in every
	loop, and a pool stuck on a dead worker ends the run."""
	done, _ = concurrent.futures.wait(futures, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
	_check_stop()
	pool.watch(done)
	return done

#%% -- Exclude --
def is_excluded(path, exclude=None):
	"""
	Check if a given path is excluded based on a list of patterns.

	Args:
		path (str): The path to check.
		exclude (list[str], optional): List of patterns to exclude. Defaults to None.

	Returns:
		bool: True if the path is excluded, False otherwise.

	Examples:
		>>> is_excluded('/data/tmp/cache', ['*/cache'])
		True
		>>> is_excluded('/data/tmp/cache', ['*/logs'])
		False
		>>> is_excluded('/data/tmp/cache', None)
		False
	"""
	if exclude is None:
		return False
	for pattern in exclude:
		if fnmatch.fnmatch(path, pattern):
			return True
	return False

def _path_and_parents(path, stop=None):
	"""path, then each directory above it, ending below stop (exclusive) or at the filesystem root."""
	while path != stop:
		yield path
		parent = os.path.dirname(path)
		if parent == path:
			return
		path = parent

def _excluded_or_below(path, exclude, stop=None):
	"""True when path, or a directory above it below stop, matches exclude: the listing prunes an excluded directory
	with everything below it, so a path reached another way (a symlink's target) is checked against its parents too,
	but (with stop) not above where the listing could have reached it."""
	return bool(exclude) and any(is_excluded(part, exclude) for part in _path_and_parents(os.path.normpath(os.path.abspath(path)), stop))

def format_exclude(exclude = None,exclude_file = None) -> frozenset:
	"""
	Format and normalize exclusion patterns for path matching.

	This function processes exclusion patterns from both a list and an optional file.
	It normalizes paths by replacing consecutive slashes with a single slash and
	ensures patterns have appropriate prefixes for glob-style matching.

	Parameters:
		exclude (iterable, optional): Collection of path patterns to exclude. 
			Defaults to None (empty set).
		exclude_file (str, optional): Path to a file containing exclusion patterns, 
			one per line. Defaults to None.

	Returns:
		frozenset: A frozen set of normalized exclusion patterns.

	Note:
		- Patterns not starting with '/' will have '*/' prepended unless they already start with '*/'.
		- The function handles errors gracefully if the exclude_file doesn't exist or can't be read.

	Examples:
		>>> sorted(format_exclude(['cache', '*/logs']))
		['*/cache', '*/logs']
		>>> format_exclude(None)
		frozenset()
	"""
	if not exclude:
		exclude = set()
	else:
		exclude = set(exclude)
	if exclude_file:
		if os.path.exists(exclude_file):
			try:
				with open(exclude_file,'r') as f:
					exclude.update(f.read().splitlines())
			except Exception:
				eprint(f"Exclude file read error: Error encountered while reading exclude file {exclude_file}, skipping")
		else:
			eprint(f"Exclude file not exist: Exclude file {exclude_file} does not exist, skipping")

	# A blank value (an empty -e, a blank line in an -x file) would become '*/', which matches every root: skip it.
	exclude = set([re.sub(r'/+','/','*/'+ path if not path.startswith('*/') else path) if not path.startswith('/') else path for path in exclude if path.strip()])
	# freeze frozenset
	return frozenset(exclude)

#%% -- DD --
def _dev_path(kernel_name):
	"""/dev path of a block device from its kernel (lsblk, sysfs) name: a '!' spells a '/' (cciss!c0d0p1 is
	/dev/cciss/c0d0p1). Never relative to the path the disk was given by (a /dev/disk/by-id link)."""
	return '/dev/' + kernel_name.replace('!', '/')

_LSBLK_PAIR_RE = re.compile(r'([A-Z][A-Z0-9:_-]*)="((?:[^"\\]|\\.)*)"')

def _lsblk_pairs(line):
	"""{column: value} of one `lsblk -P` line (KEY="value" pairs; lsblk writes special characters as \\xNN)."""
	return {key: _udev_decode(value) for key, value in _LSBLK_PAIR_RE.findall(line)}

def _lsblk_disk_rows(disk, columns):
	"""(disk row, partition rows) of `lsblk -p -nb -P -o <columns>,KNAME,PKNAME,TYPE disk`, each row the list of the
	asked columns' values.

	lsblk is asked for full paths (-p): it names each node itself (cciss, by-id and kpartx nodes alike), so no path
	is built here from the kernel names. The first row is the disk itself. lsblk also lists every device stacked on
	the disk's partitions (LVM, LUKS and dm-linear devices, md arrays) and those devices' own partitions (md127p1 of a
	partitioned md array built on the disk): a row is a partition of the disk only when its parent (PKNAME) is the
	disk itself (its KNAME) and it is a partition: the kernel says so in sysfs (_is_kernel_partition), or TYPE is part
	(a kpartx partition is a dm device, which the kernel knows nothing about; lsblk 2.23.2 and 2.32.1 call a loop
	device's partitions TYPE loop, so TYPE alone misses those). -P prints KEY="value" pairs, so the disk's empty
	PKNAME cannot shift the columns.
	"""
	names = columns.split(',')
	# A partition table just read or written by sgdisk (a read-write open) makes udev re-read it and remove and re-add
	# the partition nodes a few ms later: unsettled, lsblk can list a partition as missing, or without its type.
	_udev_settle()
	lines = run_command_in_multicmd_with_path_check(['lsblk', '-p', '-nb', '-P', '-o', columns + ',KNAME,PKNAME,TYPE', disk], strict=True, extra_env=_C_LOCALE)
	rows = [_lsblk_pairs(line) for line in lines if line.strip()]
	if not rows:
		raise RuntimeError(f'Disk error: lsblk printed nothing for {disk}')
	disk_row = rows[0]
	partitions = [row for row in rows[1:] if row.get('PKNAME') == disk_row.get('KNAME')
		and (row.get('TYPE') == 'part' or _is_kernel_partition(row.get('KNAME', '')))]
	return [disk_row.get(name, '') for name in names], [[row.get(name, '') for name in names] for row in partitions]

def get_largest_partition(disk):
	"""
	Get the largest partition on the disk (see _lsblk_disk_rows: devices stacked on its partitions, and their own partitions, are not partitions).

	Args:
		disk (str): The disk name or path.

	Returns:
		str: The path of the largest partition on the disk, or the disk itself when it has no partitions.
	"""
	disk_row, partitions = _lsblk_disk_rows(disk, 'NAME,SIZE')
	if not partitions:
		return disk_row[0]
	# sort by size
	return sorted(partitions, key=lambda row: int(row[1]))[-1][0]

def get_partitions(disk):
	"""Get all the partitions on the disk, as lsblk names their nodes (see _lsblk_disk_rows: devices stacked on its
	partitions, and their own partitions, are not partitions).

	Args:
		disk (str): The disk name or path.

	Returns:
		list: A list of partition paths.

	"""
	_, partitions = _lsblk_disk_rows(disk, 'NAME,SIZE')
	return [row[0] for row in partitions]

def get_fs_type(path):
	"""
	Get the filesystem type of the path.

	Args:
		path (str): The path for which to determine the filesystem type.

	Returns:
		str: The filesystem type of the path.
	"""
	fs_type = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'FSTYPE', path],strict=True,extra_env=_C_LOCALE)[0].strip()
	return fs_type

_FS_FIX_COMMANDS = {
	'ext4':     ['e2fsck', '-f', '-y'],
	'ext3':     ['e2fsck', '-f', '-y'],
	'ext2':     ['e2fsck', '-f', '-y'],
	'btrfs':    ['btrfs', 'check', '--repair'],
	'xfs':      ['xfs_repair', '-L'],
	'f2fs':     ['fsck.f2fs', '-a', '-f'],
	'ntfs':     ['ntfsfix'],
	'fat32':    ['fsck.fat', '-w', '-r', '-l', '-a', '-v'],
	'fat16':    ['fsck.fat', '-w', '-r', '-l', '-a', '-v'],
	'fat12':    ['fsck.fat', '-w', '-r', '-l', '-a', '-v'],
	'fat':      ['fsck.fat', '-w', '-r', '-l', '-a', '-v'],
	'vfat':     ['fsck.fat', '-w', '-r', '-l', '-a', '-v'],
	'exfat':    ['fsck.exfat', '-p', '-y'],
	'hfsplus':  ['fsck.hfsplus', '-y'],
	'hfs':      ['fsck.hfs', '-y'],
	'jfs':      ['fsck.jfs', '-y'],
	'reiserfs': ['fsck.reiserfs', '-y'],
	'ufs':      ['fsck.ufs', '-y'],
	'minix':    ['fsck.minix', '-a'],
}
_FS_FIX_UNSUPPORTED = {'udf', 'bfs'}

#%% -- Filesystem Parameter Mirroring --
# Probe a source filesystem for the parameters mkfs would otherwise pick by
# default (geometry + features), so a dd-mode clone is not silently rebuilt
# with the building host's defaults. Registries are filled in below.
_FS_PARAM_PROBES = {}
_FS_MKFS_BUILDERS = {}

def probe_fs_params(target_partition, fs_type):
	"""
	Read the creation parameters of an existing filesystem.

	Args:
		target_partition (str): Path to the partition holding the filesystem.
		fs_type (str): Filesystem type as reported by blkid (e.g. 'ext4').

	Returns:
		dict: Opaque, filesystem-specific parameters for the matching builder in
			_FS_MKFS_BUILDERS. Empty dict when the type is unknown, the probe tool
			is missing, or parsing fails. This function never raises: a failed
			probe must degrade to default mkfs behaviour, not abort a copy.
	"""
	probe = _FS_PARAM_PROBES.get(fs_type)
	if not probe:
		return {}
	# A probe tool can fail (e.g. a dirty FAT bit makes fsck.fat exit non-zero) and
	# still be a correct, quiet degrade to {} - but run_command_in_multicmd_with_path_check
	# appends "Task return code error: ..." to the global ERRORS even with quiet=True,
	# and that prefix IS in ERROR_TO_RETURNCODE_TABLE, so it would otherwise poison
	# hpcp's exit code after a fully successful copy. Snapshot and truncate ERRORS
	# back to this mark on the way out so nothing appended during this probe call
	# survives it. Probing is strictly serial (get_partition_infos iterates
	# partitions one at a time), so this can only ever discard messages this same
	# probe call appended - never another operation's errors.
	mark = len(ERRORS)
	try:
		return probe(target_partition) or {}
	except Exception as e:
		# Use a plain print here, not eprint: eprint's own ERRORS.append() would
		# just be wiped by the finally below anyway, and this keeps the intent
		# (stderr diagnostic, not a copy-breaking error) obvious at the call site.
		print(f"FS param probe warning: Could not read {fs_type} parameters from {target_partition}: {e}", file=sys.stderr)
		return {}
	finally:
		del ERRORS[mark:]

def build_mkfs_params(fs_type, fs_params):
	"""
	Translate probed filesystem parameters into mkfs arguments.

	Args:
		fs_type (str): Filesystem type as reported by blkid.
		fs_params (dict): The dict returned by probe_fs_params for the same type.

	Returns:
		list: Extra argv fragments to splice into the mkfs command, or an empty
			list when the type is unknown, params are empty, or building fails.
			This function never raises.
	"""
	if not fs_params:
		return []
	builder = _FS_MKFS_BUILDERS.get(fs_type)
	if not builder:
		return []
	# Same rationale as probe_fs_params above: a builder can legitimately warn on
	# the success path (e.g. _build_f2fs's "dropped, not recognised" notice) as
	# well as on failure, and neither should inflate hpcp's exit code. Building is
	# also serial per partition, so truncating back to this mark is safe.
	mark = len(ERRORS)
	try:
		return builder(fs_params) or []
	except Exception as e:
		print(f"FS param build warning: Could not build {fs_type} mkfs parameters: {e}", file=sys.stderr)
		return []
	finally:
		del ERRORS[mark:]

def _strip_dash_o(args):
	"""
	Return `args` with a `-O <value>` pair removed, or None if `-O` is not present.

	-O is the only version-fragile flag shared across ext/btrfs/f2fs: an ext or
	btrfs -O value includes this branch's curated feature negations (see
	_EXT_CURATED_FEATURES / _BTRFS_CURATED_FEATURES below), and a single feature
	name the local mkfs does not recognise makes it reject the ENTIRE argument
	set - block size, inode size, -m, -i, FAT width and all - not just -O.
	_build_ext now drops names local mke2fs does not parse, so this fallback is
	the last resort for invalid *combinations* rather than unknown names
	(common leftover: e2fsprogs < 1.46/1.47, btrfs-progs < 6.1/6.7).
	"""
	stripped = []
	found = False
	i = 0
	while i < len(args):
		if args[i] == '-O' and i + 1 < len(args):
			found = True
			i += 2
			continue
		stripped.append(args[i])
		i += 1
	return stripped if found else None

def _run_mkfs_with_fallback(base_command, param_args, target_partition, fs_type):
	"""
	Run mkfs with mirrored source parameters, falling back to defaults if rejected.

	Mirrored parameters can be legitimately invalid on a resized destination (a
	1 KiB block size on a filesystem grown past 16 TiB, FAT32 on a partition too
	small for it). A rejected parameter set must degrade to a default filesystem,
	not abort the copy.

	Three tiers are attempted, in order, stopping at the first success:
	  1. The full mirrored parameter set, including -O if present.
	  2. If (1) failed and -O was present: the same parameters with the -O pair
	     removed. This is what recovers block size / inode size / -m / -i / FAT
	     width - the parameters that actually affect bootability and layout -
	     when the only problem is an -O feature name the local mkfs is too old
	     to know (see _strip_dash_o).
	  3. Bare mkfs defaults, no mirrored parameters at all.

	Args:
		base_command (list): The mkfs command without the target partition.
		param_args (list): Mirrored parameter arguments, possibly empty.
		target_partition (str): Partition to create the filesystem on.
		fs_type (str): Filesystem type, for messages.

	Returns:
		bool: True if the filesystem was created by any attempted tier.
	"""
	def _attempt(command):
		resolved = [_bin(command[0])] + list(command[1:])
		task = multiCMD.run_commands([resolved], timeout=COMMAND_TIMEOUT, max_threads=1, return_object=True)[0]
		return task.returncode, (task.stderr[-1].strip() if task.stderr else '')

	if param_args:
		rc, err = _attempt(list(base_command) + list(param_args) + [target_partition])
		if rc == 0:
			return True
		eprint(f"FS param warning: mkfs rejected mirrored {fs_type} parameters {' '.join(param_args)} on {target_partition}: {err}")
		stripped = _strip_dash_o(param_args)
		if stripped is not None:
			eprint(f"FS param warning: Retrying {fs_type} on {target_partition} with -O dropped, keeping the remaining mirrored parameters.")
			rc, err = _attempt(list(base_command) + list(stripped) + [target_partition])
			if rc == 0:
				return True
			eprint(f"FS param warning: mkfs still rejected {fs_type} parameters {' '.join(stripped)} on {target_partition} after dropping -O: {err}")
		eprint(f"FS param warning: Retrying with {fs_type} defaults. The destination filesystem will not match the source exactly.")
	rc, err = _attempt(list(base_command) + [target_partition])
	if rc != 0:
		eprint(f"Create fs error: Failed to create {fs_type} on {target_partition}: {err}")
		return False
	return True

# Features mke2fs may enable from mke2fs.conf regardless of the source. A plain
# -O list is *merged* with those defaults, so every curated feature the source
# lacks has to be negated explicitly or it silently comes back.
_EXT_CURATED_FEATURES = ('has_journal', 'ext_attr', 'resize_inode', 'dir_index', 'filetype', 'extent',
						 '64bit', 'flex_bg', 'metadata_csum', 'metadata_csum_seed', 'sparse_super',
						 'large_file', 'huge_file', 'dir_nlink', 'extra_isize', 'orphan_file',
						 'fast_commit', 'casefold', 'project', 'quota', 'verity', 'encrypt',
						 'bigalloc', 'inline_data', 'ea_inode', 'mmp', 'stable_inodes', 'uninit_bg',
						 'sparse_super2', 'meta_bg')
# Runtime state, not creation parameters. Never hand these to mke2fs.
_EXT_RUNTIME_FEATURES = {'needs_recovery', 'orphan_present', 'has_snapshot', 'journal_dev', 'shared_blocks'}

# Local mke2fs -O name -> bool. Unknown names (including ^negations of features
# this e2fsprogs has never heard of) make mke2fs reject the entire option set.
_EXT_MKFS_FEATURE_KNOWN = {}

def _ext_mkfs_accepts(option_sets):
	"""Run `mke2fs -n` on /dev/null once per -O argument (8 at a time): True unless mke2fs rejects exactly that argument
	('Invalid filesystem option set: <it>', in the C locale; mke2fs.conf has the same words for a bad feature of its
	own, and any other outcome says nothing against the names). MKE2FS_CONFIG=/dev/null, so no stanza of the host's
	mke2fs.conf (a 256-block device is a `floppy`) can put its own features in the way. Never a path to make:
	e2fsprogs 1.45.x-1.46.2 create the regular file they are given before they parse -O."""
	commands = [[_bin('mkfs'), '-t', 'ext4', '-n', '-F', '-O', option_set, os.devnull, '256'] for option_set in option_sets]
	tasks = multiCMD.run_commands(commands, timeout=COMMAND_TIMEOUT, max_threads=8, return_object=True, quiet=True,
		extra_env=dict(_C_LOCALE, MKE2FS_CONFIG=os.devnull))
	return [f"Invalid filesystem option set: {option_set}" not in [line.strip() for line in task.stdout + task.stderr]
		for task, option_set in zip(tasks, option_sets)]

def _ext_mkfs_probe_features(names):
	"""Ask local mke2fs whether each -O feature name parses, and cache the answers. One unknown name (a negation of a
	feature this e2fsprogs has never heard of too: ^orphan_file on 1.45) makes mke2fs reject the whole list, and it
	never says which, so all names go in one run, and only when that is rejected, one run per name. A probe that
	cannot run fails open (known), so a missing mkfs does not strip a usable -O list."""
	todo = [n for n in dict.fromkeys(names) if n and n not in _EXT_MKFS_FEATURE_KNOWN]
	if not todo:
		return
	try:
		verdicts = [True] * len(todo) if _ext_mkfs_accepts([','.join(todo)])[0] else _ext_mkfs_accepts(todo)
	except Exception:
		verdicts = [True] * len(todo)
	_EXT_MKFS_FEATURE_KNOWN.update(zip(todo, verdicts))

def _probe_ext(device):
	"""Read ext2/3/4 geometry and features from dumpe2fs."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['dumpe2fs', '-h', device], quiet=True, extra_env=_C_LOCALE):
		key, sep, value = line.partition(':')
		if not sep:
			continue
		key = key.strip().lower()
		value = value.strip()
		if key == 'filesystem features':
			params['features'] = value.split()
		elif key == 'block size':
			params['block_size'] = int(value)
		elif key == 'inode size':
			params['inode_size'] = int(value)
		elif key == 'inode count':
			params['inode_count'] = int(value)
		elif key == 'block count':
			params['block_count'] = int(value)
		elif key == 'reserved block count':
			params['reserved_block_count'] = int(value)
	return params

def _build_ext(params):
	"""Translate probed ext parameters into mke2fs arguments."""
	args = []
	if params.get('block_size'):
		args.extend(['-b', str(params['block_size'])])
	if params.get('inode_size'):
		args.extend(['-I', str(params['inode_size'])])
	src_features = [f for f in params.get('features', []) if f not in _EXT_RUNTIME_FEATURES]
	if src_features:
		feature_opts = list(src_features) + ['^' + f for f in _EXT_CURATED_FEATURES if f not in src_features]
		_ext_mkfs_probe_features(
			opt[1:] if opt.startswith('^') else opt for opt in feature_opts)
		kept, dropped = [], []
		for opt in feature_opts:
			name = opt[1:] if opt.startswith('^') else opt
			if _EXT_MKFS_FEATURE_KNOWN.get(name, True):
				kept.append(opt)
			else:
				dropped.append(opt)
		if dropped:
			eprint(f"FS param warning: mke2fs does not recognise -O feature(s), dropping: {', '.join(dropped)}")
		if kept:
			args.extend(['-O', ','.join(kept)])
	block_count = params.get('block_count')
	block_size = params.get('block_size')
	inode_count = params.get('inode_count')
	if block_count and block_size and inode_count:
		# Mirror the inode ratio, never the absolute count: -ddr can resize the partition.
		args.extend(['-i', str(max(1024, (block_count * block_size) // inode_count))])
	reserved = params.get('reserved_block_count')
	if block_count and reserved is not None:
		args.extend(['-m', f"{reserved * 100.0 / block_count:.2f}"])
	return args

_FS_PARAM_PROBES.update({'ext2': _probe_ext, 'ext3': _probe_ext, 'ext4': _probe_ext})
_FS_MKFS_BUILDERS.update({'ext2': _build_ext, 'ext3': _build_ext, 'ext4': _build_ext})

_XFS_RO_MOUNT_OPTIONS = 'ro,nouuid,norecovery'

# $1 mount, $2 xfs_info, $3 device, $4 mount point (see _probe_xfs). rprivate first: the namespace starts out sharing
# the host's mounts, and this one must not reach it.
_XFS_NAMESPACE_PROBE = '"$1" --make-rprivate / && "$1" -o ' + _XFS_RO_MOUNT_OPTIONS + ' "$3" "$4" && exec "$2" "$4"'

def _probe_xfs(device):
	"""
	Read xfs geometry and features from xfs_info.

	xfs_info groups values into sections (meta-data / data / naming / log /
	realtime) and reuses key names across them - `bsize` is the block size under
	`data` but the directory block size under `naming` - so values are bucketed
	by section rather than flattened.
	xfs_info of xfsprogs 4.5 (EL7) reads a mounted filesystem only: one it prints nothing for is mounted read-only
	without log recovery (nothing is written), for the moment it takes, in a mount namespace of its own. The mount
	is then the kernel's to clean up (whatever ends the probe, Ctrl-C and SIGKILL included), nothing else on the
	system sees it, and nothing holds the superblock after the probe: a host mount that a process kept busy and that
	was taken off with umount -l left the superblock alive, and the next mount of the device shared it, with no log
	recovery.
	"""
	params = {}
	section = ''
	lines = run_command_in_multicmd_with_path_check(['xfs_info', device], quiet=True, extra_env=_C_LOCALE)
	if not lines and check_path('xfs_info'):
		mount_point = tempfile.mkdtemp()
		try:
			# device and mount_point are arguments of the script, never part of it; mount and xfs_info are the ones hpcp finds
			lines = run_command_in_multicmd_with_path_check(['unshare', '-m', 'sh', '-c', _XFS_NAMESPACE_PROBE, 'sh',
				_bin('mount'), _bin('xfs_info'), device, mount_point], quiet=True, extra_env=_C_LOCALE)
		finally:
			try:
				os.rmdir(mount_point)
			except OSError:
				pass
	for line in lines:
		if not line.strip():
			continue
		if not line[0].isspace():
			section = line.split('=')[0].strip()
		if not section:
			continue
		bucket = params.setdefault(section, {})
		for token in line.replace(',', ' ').split():
			key, sep, value = token.partition('=')
			if key and sep and value:
				bucket[key] = value
	return params

def _build_xfs(params):
	"""Translate probed xfs parameters into mkfs.xfs arguments."""
	meta = params.get('meta-data', {})
	data = params.get('data', {})
	naming = params.get('naming', {})
	args = []
	if data.get('bsize'):
		args.extend(['-b', 'size=' + data['bsize']])
	if meta.get('sectsz'):
		args.extend(['-s', 'size=' + meta['sectsz']])
	inode_opts = []
	if meta.get('isize'):
		inode_opts.append('size=' + meta['isize'])
	for key in ('sparse', 'projid32bit', 'nrext64'):
		if key in meta:
			inode_opts.append(f'{key}={meta[key]}')
	if data.get('imaxpct'):
		inode_opts.append('maxpct=' + data['imaxpct'])
	if inode_opts:
		args.extend(['-i', ','.join(inode_opts)])
	naming_opts = []
	if naming.get('bsize'):
		naming_opts.append('size=' + naming['bsize'])
	if 'ftype' in naming:
		naming_opts.append('ftype=' + naming['ftype'])
	if naming_opts:
		args.extend(['-n', ','.join(naming_opts)])
	meta_opts = [f'{key}={meta[key]}' for key in ('crc', 'finobt', 'rmapbt', 'reflink', 'bigtime', 'inobtcount') if key in meta]
	if meta_opts:
		args.extend(['-m', ','.join(meta_opts)])
	# Log size and agcount are deliberately not mirrored: both scale with fs size.
	return args

_FS_PARAM_PROBES['xfs'] = _probe_xfs
_FS_MKFS_BUILDERS['xfs'] = _build_xfs

# btrfs superblock flag names -> the feature names mkfs.btrfs -O accepts.
# Flags with no mkfs equivalent (MIXED_BACKREF, BIG_METADATA, DEFAULT_SUBVOL,
# FREE_SPACE_TREE_VALID, COMPRESS_*) are intentionally absent and get dropped.
_BTRFS_FLAG_TO_MKFS = {
	'MIXED_GROUPS':     'mixed-bg',
	'EXTENDED_IREF':    'extref',
	'SKINNY_METADATA':  'skinny-metadata',
	'NO_HOLES':         'no-holes',
	'RAID56':           'raid56',
	'RAID1C34':         'raid1c34',
	'ZONED':            'zoned',
	'SIMPLE_QUOTA':     'squota',
	'FREE_SPACE_TREE':  'free-space-tree',
	'BLOCK_GROUP_TREE': 'block-group-tree',
}
_BTRFS_CURATED_FEATURES = tuple(sorted(set(_BTRFS_FLAG_TO_MKFS.values())))

def _probe_btrfs(device):
	"""Read btrfs geometry, checksum type and feature flags from the superblock."""
	params = {'incompat_flags': [], 'compat_ro_flags': []}
	current = None
	for line in run_command_in_multicmd_with_path_check(['btrfs', 'inspect-internal', 'dump-super', device], quiet=True, extra_env=_C_LOCALE):
		stripped = line.strip()
		if not stripped:
			continue
		fields = stripped.split()
		name = fields[0]
		if name in ('nodesize', 'sectorsize') and len(fields) > 1:
			params[name] = int(fields[1])
			current = None
		elif name == 'csum_type' and len(fields) > 2:
			params['csum_type'] = fields[2].strip('()')
			current = None
		elif name in ('incompat_flags', 'compat_ro_flags'):
			current = name
		elif current:
			# Flag blocks look like "( MIXED_BACKREF |" ... "  NO_HOLES )".
			token = stripped.strip('()| \t')
			if token and token.replace('_', '').isalnum() and token.upper() == token:
				params[current].append(token)
			if stripped.endswith(')'):
				current = None
		else:
			current = None
	return params

def _build_btrfs(params):
	"""Translate probed btrfs parameters into mkfs.btrfs arguments."""
	args = []
	if params.get('nodesize'):
		args.extend(['-n', str(params['nodesize'])])
	if params.get('sectorsize'):
		args.extend(['-s', str(params['sectorsize'])])
	if params.get('csum_type'):
		args.extend(['--csum', params['csum_type']])
	flags = list(params.get('incompat_flags', [])) + list(params.get('compat_ro_flags', []))
	if flags:
		src_features = {_BTRFS_FLAG_TO_MKFS[f] for f in flags if f in _BTRFS_FLAG_TO_MKFS}
		feature_opts = sorted(src_features) + ['^' + f for f in _BTRFS_CURATED_FEATURES if f not in src_features]
		args.extend(['-O', ','.join(feature_opts)])
	return args

_FS_PARAM_PROBES['btrfs'] = _probe_btrfs
_FS_MKFS_BUILDERS['btrfs'] = _build_btrfs

def _probe_vfat(device):
	"""
	Read FAT width and geometry.

	blkid -o export reports TYPE=vfat for FAT12, FAT16 and FAT32 alike, so the
	width has to come from blkid -p's VERSION field. Without it mkfs.vfat picks
	the width from the partition size and can turn a FAT32 EFI System Partition
	into FAT16 that firmware refuses to boot.
	"""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'export', device], quiet=True, extra_env=_C_LOCALE):
		key, sep, value = line.partition('=')
		if sep and key.strip() == 'VERSION':
			version = value.strip().upper()
			if version.startswith('FAT') and version[3:].isdigit():
				params['fat_bits'] = int(version[3:])
	for line in run_command_in_multicmd_with_path_check(['fsck.fat', '-nv', device], quiet=True, extra_env=_C_LOCALE):
		fields = line.split()
		if not fields or not fields[0].isdigit():
			continue
		if 'bytes per logical sector' in line:
			params['sector_size'] = int(fields[0])
		elif 'bytes per cluster' in line:
			params['cluster_size'] = int(fields[0])
		elif 'reserved sectors' in line:
			params['reserved_sectors'] = int(fields[0])
		elif 'FATs,' in line and 'bit entries' in line:
			params['fat_count'] = int(fields[0])
			params.setdefault('fat_bits', int(fields[2]))
	return params

def _build_vfat(params):
	"""Translate probed FAT parameters into mkfs.vfat arguments."""
	args = []
	if params.get('fat_bits') in (12, 16, 32):
		args.extend(['-F', str(params['fat_bits'])])
	sector_size = params.get('sector_size')
	if sector_size:
		args.extend(['-S', str(sector_size)])
		cluster_size = params.get('cluster_size')
		if cluster_size and cluster_size >= sector_size:
			args.extend(['-s', str(cluster_size // sector_size)])
	if params.get('fat_count'):
		args.extend(['-f', str(params['fat_count'])])
	if params.get('reserved_sectors'):
		args.extend(['-R', str(params['reserved_sectors'])])
	return args

for _fat_alias in ('vfat', 'fat', 'fat12', 'fat16', 'fat32', 'msdos'):
	_FS_PARAM_PROBES[_fat_alias] = _probe_vfat
	_FS_MKFS_BUILDERS[_fat_alias] = _build_vfat
del _fat_alias

def _probe_ntfs(device):
	"""Read ntfs cluster and sector size from ntfsinfo."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['ntfsinfo', '-m', device], quiet=True, extra_env=_C_LOCALE):
		key, sep, value = line.partition(':')
		if not sep:
			continue
		key = key.strip().lower()
		value = value.strip()
		if key == 'cluster size' and value.isdigit():
			params['cluster_size'] = int(value)
		elif key == 'sector size' and value.isdigit():
			params['sector_size'] = int(value)
	return params

def _build_ntfs(params):
	"""Translate probed ntfs parameters into mkfs.ntfs arguments."""
	args = []
	if params.get('cluster_size'):
		args.extend(['-c', str(params['cluster_size'])])
	if params.get('sector_size'):
		args.extend(['-s', str(params['sector_size'])])
	return args

def _probe_exfat(device):
	"""Read exfat sector and cluster size from dump.exfat."""
	params = {}
	# Geometry only: under LC_ALL=C it prints a non-ASCII label as '?', so labels come from blkid.
	for line in run_command_in_multicmd_with_path_check(['dump.exfat', device], quiet=True, extra_env=_C_LOCALE):
		key, sep, value = line.partition(':')
		if not sep:
			continue
		key = key.strip().lower()
		value = value.strip()
		if key == 'bytes per sector' and value.isdigit():
			params['sector_size'] = int(value)
		elif key == 'cluster size' and value.isdigit():
			params['cluster_size'] = int(value)
	return params

def _build_exfat(params):
	"""Translate probed exfat parameters into mkfs.exfat arguments."""
	args = []
	if params.get('sector_size'):
		args.extend(['-s', str(params['sector_size'])])
	if params.get('cluster_size'):
		args.extend(['-c', str(params['cluster_size'])])
	return args

# f2fs superblock feature names dump.f2fs reports that differ from the name
# mkfs.f2fs -O expects for the same feature. Confirmed via a live round-trip:
# `mkfs.f2fs -O quota` produces a superblock dump.f2fs reports as `quota_ino`;
# feeding `quota_ino` straight back to `mkfs.f2fs -O` is rejected with
# "Error: Wrong features quota_ino".
_F2FS_FLAG_TO_MKFS = {'quota_ino': 'quota'}

# Feature names mkfs.f2fs -O actually accepts, per `man mkfs.f2fs` / mkfs.f2fs's
# own usage text (f2fs-tools 1.16.0) and confirmed live: `mkfs.f2fs -O <all of
# these, comma separated>` exits 0. Anything dump.f2fs reports that isn't in
# here (e.g. `blkzoned`, a zoned-device property rather than an -O feature) is
# dropped rather than passed through, since an unrecognised -O name makes
# mkfs.f2fs reject the whole argument set - geometry included.
_F2FS_CURATED_FEATURES = (
	'encrypt', 'extra_attr', 'project_quota', 'inode_checksum', 'flexible_inline_xattr',
	'quota', 'inode_crtime', 'lost_found', 'verity', 'sb_checksum', 'casefold', 'compression',
)

# mkfs.f2fs enables no features by default and has no negation syntax, so the
# source's feature list is applied additively rather than as a delta.
def _probe_f2fs(device):
	"""Read f2fs geometry and features from dump.f2fs."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['dump.f2fs', '-d', '1', device], quiet=True, extra_env=_C_LOCALE):
		if 'superblock features' in line:
			names = line.rpartition(':')[2]
			params['features'] = [n.strip() for n in names.replace(',', ' ').split() if n.strip()]
			continue
		fields = line.split()
		if not fields or '[' not in line or ':' not in line:
			continue
		if fields[0] not in ('log_sectorsize', 'segs_per_sec', 'secs_per_zone'):
			continue
		value = line.rpartition(':')[2].strip().rstrip(']').strip()
		if value.isdigit():
			params[fields[0]] = int(value)
	return params

def _build_f2fs(params):
	"""Translate probed f2fs parameters into mkfs.f2fs arguments."""
	args = []
	if params.get('log_sectorsize'):
		args.extend(['-w', str(1 << params['log_sectorsize'])])
	if params.get('segs_per_sec'):
		args.extend(['-s', str(params['segs_per_sec'])])
	if params.get('secs_per_zone'):
		args.extend(['-z', str(params['secs_per_zone'])])
	kept, dropped = [], []
	for name in params.get('features') or []:
		mapped = _F2FS_FLAG_TO_MKFS.get(name, name)
		if mapped in _F2FS_CURATED_FEATURES:
			kept.append(mapped)
		else:
			dropped.append(name)
	if dropped:
		eprint(f"FS param warning: f2fs feature(s) not recognised by mkfs.f2fs -O, dropping: {', '.join(dropped)}")
	if kept:
		args.extend(['-O', ','.join(kept)])
	return args

_FS_PARAM_PROBES.update({'ntfs': _probe_ntfs, 'exfat': _probe_exfat, 'f2fs': _probe_f2fs})
_FS_MKFS_BUILDERS.update({'ntfs': _build_ntfs, 'exfat': _build_exfat, 'f2fs': _build_f2fs})

def _probe_udf(device):
	"""Read udf block size and revision from udfinfo."""
	params = {}
	# Geometry only: under LC_ALL=C it prints a non-ASCII label as '?', so labels come from blkid.
	for line in run_command_in_multicmd_with_path_check(['udfinfo', device], quiet=True, extra_env=_C_LOCALE):
		key, sep, value = line.partition('=')
		if not sep:
			continue
		key = key.strip()
		value = value.strip()
		if key == 'blocksize' and value.isdigit():
			params['block_size'] = int(value)
		elif key == 'udfrev' and value:
			params['udfrev'] = value
	return params

def _build_udf(params):
	"""Translate probed udf parameters into mkudffs arguments."""
	args = []
	if params.get('block_size'):
		args.append('--blocksize=' + str(params['block_size']))
	if params.get('udfrev'):
		args.append('--udfrev=' + params['udfrev'])
	return args

def _probe_reiserfs(device):
	"""Read reiserfs block size, on-disk format and hash from debugreiserfs."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['debugreiserfs', device], quiet=True, extra_env=_C_LOCALE):
		stripped = line.strip()
		if stripped.lower().startswith('blocksize:'):
			value = stripped.partition(':')[2].strip()
			if value.isdigit():
				params['block_size'] = int(value)
		elif 'of format' in stripped:
			fields = stripped.split('of format')[1].split()
			if fields:
				params['fs_format'] = fields[0]
		elif 'hash function' in stripped.lower() and '"' in stripped:
			params['hash_function'] = stripped.split('"')[1]
	return params

def _build_reiserfs(params):
	"""Translate probed reiserfs parameters into mkfs.reiserfs arguments."""
	args = []
	if params.get('block_size'):
		args.extend(['-b', str(params['block_size'])])
	if params.get('fs_format'):
		args.extend(['--format', params['fs_format']])
	if params.get('hash_function'):
		args.extend(['-h', params['hash_function']])
	return args

_FS_PARAM_PROBES.update({'udf': _probe_udf, 'reiserfs': _probe_reiserfs})
_FS_MKFS_BUILDERS.update({'udf': _build_udf, 'reiserfs': _build_reiserfs})

# minix superblock magic -> (version, max filename length). v1/v2 keep the magic
# at byte 16 of the superblock, v3 at byte 24.
_MINIX_MAGICS = {0x137F: (1, 14), 0x138F: (1, 30), 0x2468: (2, 14), 0x2478: (2, 30)}
_MINIX3_MAGIC = 0x4D5A

def _probe_hfsplus(device):
	"""
	Read the hfs+ allocation block size straight from the volume header.

	No fsck.hfsplus output reports it, but the volume header is at a fixed
	offset: signature at byte 1024 ('H+' / 'HX'), blockSize as a big-endian
	uint32 40 bytes into the header.
	"""
	with open(device, 'rb') as f:
		f.seek(1024)
		header = f.read(64)
	if len(header) < 44:
		return {}
	signature = struct.unpack_from('>H', header, 0)[0]
	if signature not in (0x482B, 0x4858):
		return {}
	block_size = struct.unpack_from('>I', header, 40)[0]
	return {'block_size': block_size} if block_size else {}

def _build_hfsplus(params):
	"""Translate probed hfs+ parameters into mkfs.hfsplus arguments."""
	if params.get('block_size'):
		return ['-b', str(params['block_size'])]
	return []

def _probe_minix(device):
	"""Read the minix filesystem version and filename length from the superblock."""
	with open(device, 'rb') as f:
		f.seek(1024)
		superblock = f.read(64)
	if len(superblock) < 32:
		return {}
	magic = struct.unpack_from('<H', superblock, 16)[0]
	if magic in _MINIX_MAGICS:
		version, name_length = _MINIX_MAGICS[magic]
		return {'fs_version': version, 'name_length': name_length}
	if struct.unpack_from('<H', superblock, 24)[0] == _MINIX3_MAGIC:
		return {'fs_version': 3, 'name_length': 60}
	return {}

def _build_minix(params):
	"""
	Translate probed minix parameters into mkfs.minix arguments.

	mkfs.minix exposes no block size option, so a v3 block size cannot be
	mirrored; version and filename length are what it accepts.
	"""
	args = []
	version = params.get('fs_version')
	if version in (1, 2, 3):
		args.append(f'-{version}')
	if version in (1, 2) and params.get('name_length') in (14, 30):
		args.extend(['-n', str(params['name_length'])])
	return args

_FS_PARAM_PROBES.update({'hfsplus': _probe_hfsplus, 'hfs': _probe_hfsplus, 'minix': _probe_minix})
_FS_MKFS_BUILDERS.update({'hfsplus': _build_hfsplus, 'hfs': _build_hfsplus, 'minix': _build_minix})

# Deliberately not registered, with reasons:
#   jfs   - mkfs.jfs exposes no geometry options; block size is fixed at 4096.
#   bfs, ufs, swap - mkfs/newfs/mkswap expose nothing worth mirroring, and the
#                    on-disk headers carry no creation parameters we can act on.
#   zfs, cramfs, iso9660 - hpcp already declines to create these.
# Unregistered types fall through probe_fs_params/build_mkfs_params to {} / [],
# which is exactly today's default-mkfs behaviour.

def fix_fs(target_partition, fs_type=None):
	"""
	Fix the file system errors on the specified target partition.

	Args:
		target_partition (str): The target partition to fix.
		fs_type (str, optional): The file system type of the target partition. If not provided, it will be determined automatically.

	Returns:
		bool: True if the file system errors are fixed successfully, False otherwise.
	"""
	try:
		if not fs_type:
			fs_type = get_fs_type(target_partition)
		if fs_type in _FS_FIX_COMMANDS:
			run_command_in_multicmd_with_path_check(_FS_FIX_COMMANDS[fs_type] + [target_partition], strict=False)
		elif fs_type in _FS_FIX_UNSUPPORTED:
			eprint(f"Cannot fix {fs_type} file system. Skipping.")
		else:
			eprint(f"Not supported fs for fixing: File system {fs_type} not supported.")
			return False
	except Exception as e:
		eprint(f"Fix fs Error: Exception caught when trying to fix {target_partition} with {fs_type}: {e}")
		return False
	return True

def create_loop_device(image_path,read_only=False):
	"""
	Create a loop device for the image file.

	Args:
		image_path (str): The path to the image file.
		read_only (bool, optional): Whether to create the loop device in read-only mode. Defaults to False.

	Returns:
		str: The path to the created loop device.
	"""
	if read_only:
		loop_device_dest = run_command_in_multicmd_with_path_check(["losetup", '--partscan', '--find', '--show', '--read-only', image_path],strict=True)[0].strip()
	else:
		loop_device_dest = run_command_in_multicmd_with_path_check(["losetup", '--partscan', '--find', '--show', image_path],strict=True)[0].strip()
	#run_command_in_multicmd_with_path_check(f'partprobe {loop_device_dest}')
	print(f"Loop device {loop_device_dest} created.")
	return loop_device_dest

def get_target_partition(image, partition_name):
	"""
	Gets the device path for a specific partition within an image file or block device.
	
	This function handles both regular image files and block devices. If the input is an
	image file, it creates a loop device first. It then identifies all partitions and
	returns the path to the partition that matches the given partition name.
	The partitions are the rows of lsblk whose parent is the disk, with full paths (see _lsblk_disk_rows): a
	device stacked on a partition, or that device's own partitions, such as a dm-linear device on partition 1 named
	hpcp-holder-2, is never matched as partition 2.
	
	Args:
		image (str): Path to the disk image file or block device
		partition_name (str): Name or suffix of the target partition to find
		
	Returns:
		tuple: A tuple containing:
			- str: Path to the target partition (e.g., '/dev/loop0p1')
			- str or None: Path to the loop device if one was created, otherwise None
			
	Note:
		If the image is not a block device, a loop device will be created using
		the create_loop_device function and will need to be cleaned up by the caller.
	"""
	loop_device = None
	if not pathlib.Path(image).resolve().is_block_device():
		loop_device = create_loop_device(image)
		image = loop_device
	# Need to get a partition path for mkfs
	_, partitions = _lsblk_disk_rows(image, 'NAME')
	target_partition = ''
	for row in partitions:
		if row[0].endswith(partition_name):
			target_partition = row[0]
			break
	return target_partition, loop_device

@functools.lru_cache(maxsize=None)
def get_partition_details(device, partition,sector_size=512, identity=True):
	"""
	Retrieves detailed information about a specific partition on a device.

	This function gathers partition information including GUID codes, partition names,
	attributes, filesystem type, UUID, label, and size using tools like sgdisk and blkid.

	Args:
		device (str): Path to the device containing the partition (e.g., '/dev/sda').
		partition (str): The partition number or identifier (e.g., '1' for first partition).
		sector_size (int, optional): Sector size in bytes. Defaults to 512.
		identity (bool, optional): False reads the GPT entry only (one sgdisk call): no partition lookup, blkid or
			fs-params probe; read_disk_layout probes the settled device itself.

	Returns:
		dict: A dictionary containing partition details with the following keys:
			- 'partition_guid_code': The GUID code identifying the partition type.
			- 'unique_partition_guid': The unique GUID identifying this specific partition.
			- 'partition_name': The name of the partition if available.
			- 'partition_attrs': Attribute flags for the partition.
			- 'fs_type': The filesystem type (e.g., 'ext4', 'ntfs').
			- 'fs_uuid': The UUID of the filesystem.
			- 'fs_label': The filesystem label if available.
			- 'size': Size of the partition in bytes.
			- 'fs_params': Probed source filesystem creation parameters used for
				mirroring in dd mode (see probe_fs_params). Empty dict when
				fs_type is unknown, MIRROR_FS_PARAMS is False, or probing failed.

	Notes:
		- This function uses external tools (sgdisk, blkid) and requires appropriate permissions.
		- If the device is an image file, a loop device is temporarily created and then detached.
	"""
	# Get the partition info from the source
	result = run_command_in_multicmd_with_path_check(["sgdisk", '--info='+partition, device],strict=True,extra_env=_C_LOCALE)
	rtnDic = {'partition_guid_code': '', 'unique_partition_guid': '', 'partition_name': '', 'partition_attrs': '', 'fs_type': '', 'fs_uuid': '', 'fs_label': '', 'size': 0, 'fs_params': {}}
	for line in result:
		# Each field is matched at the start of its line: a GPT name may hold another field's label ('guid code: x').
		field = line.strip().lower()
		if field.startswith("partition guid code:"):
			rtnDic['partition_guid_code'] = line.split(":")[1].split()[0].strip()
		elif field.startswith("partition unique guid:"):
			rtnDic['unique_partition_guid'] = line.split(":")[1].strip()
		elif field.startswith("partition name:"):
			# sgdisk prints the name raw between quotes, a quote inside it included: everything between the first
			# and the last quote, spaces kept.
			first, last = line.find("'"), line.rfind("'")
			rtnDic['partition_name'] = line[first + 1:last] if last > first else ''
		elif field.startswith("attribute flags:"):
			rtnDic['partition_attrs'] = line.split(":")[1].strip()
		elif field.startswith('partition size:'):
			rtnDic['size'] = int(line.split(':')[1].split()[0].strip()) * int(sector_size)
	# Also get the fs type, use parted
	# result = run_command_in_multicmd_with_path_check(f"parted --machine --script {device} print")
	# for line in result:
	# 	if line.startswith(partition):
	# 		rtnDic['fs_type'] = line.split(':')[4].strip()
	# 		return rtnDic
	if not identity:
		return rtnDic
	# Get the fs type and uuid, use blkid
	target_partition, loop_device = get_target_partition(device, partition)
	# The same exact, uncached probe disk patches use (_probe_identity): -o export mangled non-ASCII labels, and
	# 'LABEL' in line also matched LABEL_FATBOOT and cut a label at its first '='.
	if target_partition:
		rtnDic['fs_type'], rtnDic['fs_uuid'], rtnDic['fs_label'] = _probe_identity(target_partition)
	if rtnDic['fs_type'] and MIRROR_FS_PARAMS:
		rtnDic['fs_params'] = probe_fs_params(target_partition, rtnDic['fs_type'])
	if loop_device:
		run_command_in_multicmd_with_path_check(["losetup", '--detach', loop_device])
	return rtnDic

@functools.lru_cache(maxsize=None)
def get_partition_infos(device, identity=True):
	"""
	Extracts detailed partition information from a specified block device.
	
	This function parses the output of the 'sgdisk --print' command to gather 
	information about partitions on the device, including disk size, identifier, 
	sector size, and individual partition details.
	
	Args:
		device (str): Path to the block device (e.g., '/dev/sda').
		identity (bool, optional): Passed on to get_partition_details. True (the -dd copy) also probes each
			partition's filesystem type, UUID, label and mkfs parameters; False reads the GPT entries alone, one
			sgdisk call per partition (read_disk_layout probes the settled partitions itself).
		
	Returns:
		dict: A dictionary containing disk and partition information with the following structure:
			{
				'disk_name': {
					'size': int,             # Total disk size in bytes
					'disk_identifier': str,   # Disk UUID/identifier
					'sector_size': int        # Sector size in bytes
				},
				'partition_number': {         # Partition details from get_partition_details()
					...
				},
				...
			}
	
	Note:
		This function depends on 'run_command_in_multicmd_with_path_check' and 'get_partition_details'.
	"""
	# partitions = run_command_in_multicmd_with_path_check(f"lsblk -nbl -o NAME,SIZE {device}")
	# partition_info = {part.split()[0]: int(part.split()[1]) for part in partitions}
	partitions = run_command_in_multicmd_with_path_check(["sgdisk", '--print', device],strict=True,extra_env=_C_LOCALE)
	line = partitions.pop(0)
	partition_info = {}
	disk_size_sector = 0
	disk_identifier = ''
	sector_size = 512
	disk_name = ''
	while not line.lower().startswith('number'):
		if line.lower().startswith('disk') and not line.lower().startswith('disk identifier') and 'sectors' in line:
			disk_size_sector = int(line.rpartition(':')[2].split()[0])
			disk_name = line.split()[1].strip(':')
		elif 'sector size' in line.lower():
			sector_size = int(line.rpartition(':')[2].split()[0].partition('/')[0])
		elif 'disk identifier' in line.lower():
			disk_identifier = line.rpartition(':')[2].strip()
		line = partitions.pop(0)
	partition_info[disk_name] = {'size': disk_size_sector*sector_size, 'disk_identifier': disk_identifier, 'sector_size': sector_size}
	for part in partitions:
		if not part or not part.strip() or len(part.split()) < 4 or part.lower().startswith('error'):
			continue
		part = part.split()
		partition_info[part[0]] = get_partition_details(device, part[0], sector_size, identity)
	return partition_info

_BTRFS_TEMP_FSID_FEATURE = '/sys/fs/btrfs/features/temp_fsid'

def _btrfs_temp_fsid_supported():
	"""True when the kernel's btrfs has temp-fsid: it can mount a second single-device btrfs whose fsid is already
	mounted (the static feature file exists; like rmdir_subvol it reads 0)."""
	return os.path.exists(_BTRFS_TEMP_FSID_FEATURE)

def write_partition_info(image, partition_infos, partition_name):
	"""
	Writes partition information to a specified partition within a disk image.
	
	This function handles the configuration of partition attributes, GUID codes,
	unique GUIDs, and file system creation. It supports various file systems including
	ext2/3/4, btrfs, xfs, f2fs, ntfs, fat/vfat variants, exfat, hfs/hfsplus, udf, jfs,
	reiserfs, and others.
	
	Args:
		image (str): Path to the disk image file to be modified.
		partition_infos (dict): Dictionary containing partition configuration details.
			Expected keys for the specified partition_name:
			- 'partition_guid_code': GUID type code for the partition.
			- 'unique_partition_guid': Unique GUID for the partition.
			- 'partition_attrs': Hexadecimal string of partition attributes.
			- 'fs_type': File system type to create.
			- 'fs_label': Label for the file system.
			- 'fs_uuid': UUID for the file system.
			- 'partition_name': Name for the partition.
		partition_name (str): The partition identifier to be configured.
	
	Raises:
		Exception: If any error occurs during the partition information writing process.
	
	Note:
		Some file systems have limitations on setting labels or UUIDs, and appropriate
		warnings will be printed in these cases. Special handling is applied for read-only
		file systems like cramfs and iso9660.
	"""
	delayed_commands = []
	try:
		# Apply the GUID code, unique GUID, and attributes to the target
		if partition_infos[partition_name]['partition_guid_code']:
			run_command_in_multicmd_with_path_check(["sgdisk", '--typecode='+partition_name+':'+partition_infos[partition_name]['partition_guid_code'], image],strict=False)
		if partition_infos[partition_name]['unique_partition_guid']:
			run_command_in_multicmd_with_path_check(["sgdisk", '--partition-guid='+partition_name+':'+partition_infos[partition_name]['unique_partition_guid'], image],strict=False)
		if partition_infos[partition_name]['partition_attrs']:
			binary_attributes = bin(int(partition_infos[partition_name]['partition_attrs'], 16))[2:].zfill(64)[::-1]
			for i, bit in enumerate(binary_attributes):
				if bit == '1':
					run_command_in_multicmd_with_path_check(["sgdisk", f"--attributes={partition_name}:set:{i}", image],strict=False)
		if partition_infos[partition_name]['fs_type']:
			target_partition, loop_device = get_target_partition(image, partition_name)
			if not target_partition:
				eprint(f"Partition not found error: Cannot find partition {partition_name} in {image}.")
				# Callers do delayed_commands.extend(write_partition_info(...)), so this
				# must stay a list; a bare return turned a handled error into TypeError.
				return delayed_commands
			fs_type = partition_infos[partition_name]['fs_type']
			fs_label = partition_infos[partition_name]['fs_label']
			fs_uuid = partition_infos[partition_name]['fs_uuid']
			fs_params = partition_infos[partition_name].get('fs_params') or {}
			param_args = build_mkfs_params(fs_type, fs_params) if MIRROR_FS_PARAMS else []
			if param_args:
				print(f"Mirroring source {fs_type} parameters onto {target_partition}: {' '.join(param_args)}")
			if fs_type in ('ext4', 'ext3', 'ext2'):
				command = ['mkfs', '-t', fs_type]
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					# UUID is delayed until after the copy so the live source and
					# the new dest do not share a UUID while both are mounted.
					# tune2fs -U on a metadata_csum filesystem (default ext4)
					# refuses unless the fs has been freshly checked, and the
					# copy dirties the superblock, so e2fsck -f must run first.
					# These two delayed commands are an ordered pair; clean_up
					# runs remaining delayed commands serially.
					delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			elif fs_type == 'btrfs':
				command = ['mkfs.btrfs']
				if fs_label:
					command.extend(['-L', fs_label])
				# Not delayed: btrfstune -U refuses a UUID the attached source carries; -U at mkfs needs temp_fsid to mount next to it.
				if fs_uuid and _btrfs_temp_fsid_supported():
					command.extend(['-U', fs_uuid])
				elif fs_uuid:
					print(f"Warning: this kernel cannot mount two btrfs filesystems with one UUID (no temp_fsid support); "
						f"{target_partition} gets a new btrfs UUID; set {fs_uuid} with btrfstune -U once no attached device carries {fs_uuid}", file=sys.stderr)
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			elif fs_type == 'xfs':
				command = ['mkfs.xfs']
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					#command.extend(['-m', f'uuid={fs_uuid}'])
					delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			elif fs_type == 'f2fs':
				command = ['mkfs.f2fs', '-f']
				if fs_label:
					command.extend(['-l', fs_label])
				if fs_uuid:
					command.extend(['-U', fs_uuid])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			elif fs_type == 'ntfs':
				command = ['mkfs.ntfs']
				if fs_label:
					command.extend(['-L', fs_label])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
				if fs_uuid:
					eprint("Cannot set fs uuid for ntfs: Skipping.")
			elif fs_type in ('fat32', 'fat16', 'fat12', 'fat', 'vfat', 'msdos'):
				command = ['mkfs.vfat']
				if fs_label:
					command.extend(['-n', fs_label])
				if fs_uuid:
					# cannot delay
					command.extend(['-i', fs_uuid.lower().replace('-','')])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			elif fs_type == 'exfat':
				command = ['mkfs.exfat']
				if fs_label:
					command.extend(['-L', fs_label])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
				if fs_uuid:
					#run_command_in_multicmd_with_path_check(["exfatlabel", '-i', target_partition, fs_uuid],strict=False)
					delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))
			elif fs_type in ('hfsplus', 'hfs'):
				command = [f'mkfs.{fs_type}']
				if fs_label:
					command.extend(['-v', fs_label])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
				if fs_uuid:
					eprint(f"Cannot set fs uuid for hfs: {fs_type}. Skipping.")
			elif fs_type == 'udf':
				# command += f" {target_partition}"
				command = ['mkudffs', '--media-type=hd']
				if fs_label:
					command.extend(['--label', fs_label])
				if fs_uuid:
					#command.extend(['--uuid', fs_uuid])
					delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			elif fs_type == 'jfs':
				command = ['mkfs.jfs']
				if fs_label:
					command.extend(['-L', fs_label])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
				if fs_uuid:
					#run_command_in_multicmd_with_path_check(["jfs_tune", '-U', fs_uuid, target_partition],strict=False)
					delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))
			elif fs_type == 'reiserfs':
				command = ['mkfs.reiserfs']
				if fs_label:
					command.extend(['-l', fs_label])
				if fs_uuid:
					#command.extend(['-u', fs_uuid])
					delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			elif fs_type == 'zfs':
				eprint("Cannot create zfs file system: ZFS file system should be created using zpool command.")
			elif fs_type == 'ufs':
				command = ['newfs', '-t']
				if fs_label:
					command.extend(['-L', fs_label])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
				if fs_uuid:
					eprint("Cannot set fs uuid for ufs: Skipping.")
			elif fs_type == 'bfs':
				command = ['mkfs.bfs']
				if fs_label:
					command.extend(['-F', fs_label, '-V', fs_label])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
				if fs_uuid:
					eprint("Cannot set fs uuid for bfs: Skipping.")
			elif fs_type == 'cramfs':
				eprint("Cannot create cramfs file system: cramfs is read-only file system. You should create one with mkfs.cramfs.")
			elif fs_type == 'minix':
				command = ['mkfs.minix']
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
				if fs_label:
					eprint("Cannot set fs label for minix: Skipping.")
				if fs_uuid:
					eprint("Cannot set fs uuid for minix: Skipping.")
			elif fs_type == 'iso9660':
				eprint("Cannot create iso9660 file system: iso9660 is read-only file system. You should create one with mkfs.iso9660.")
			elif fs_type == 'swap':
				command = ['mkswap']
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					#command.extend(['-U', fs_uuid])
					delayed_commands.extend(_fs_identity_commands(fs_type, target_partition, uuid=fs_uuid))
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			elif fs_type == 'gpt':
				eprint("Skip creating gpt padding.")
			else:
				eprint(f"Not supported fs for creating: File system {fs_type} not currently supported by hpcp. Trying mkfs -t {fs_type} anyway...")
				command = ['mkfs', '-t', fs_type]
				if fs_label:
					command.extend(['-L', fs_label])
				if fs_uuid:
					command.extend(['-U', fs_uuid])
				_run_mkfs_with_fallback(command, param_args, target_partition, fs_type)
			if loop_device:
				#run_command_in_multicmd_with_path_check(["losetup", '--detach', loop_device])
				delayed_commands.append(['losetup', '--detach', loop_device])
		if partition_infos[partition_name]['partition_name']:
			run_command_in_multicmd_with_path_check(['sgdisk', f'--change-name={partition_name}:{partition_infos[partition_name]["partition_name"]}', image],strict=False)
	except Exception as e:
		eprint("Partition copy error: An error occurred while copying partition information:", e)
	return delayed_commands

def create_partition_table(image, partition_infos,sorted_partitions):
	"""Create a partition table in the image file that will match the source device and sort the partitions by size.

	Returns:
		tuple: (delayed_commands, loop_device). loop_device is the losetup device created for an
			image file, or None when image is already a block device. Caller owns the loop and
			must detach it (typically via loop_devices / clean_up); it is not queued for detach here.
	"""
	# Create a partition table in the image file
	run_command_in_multicmd_with_path_check(["sgdisk", '--clear', image])
	src_disk_path = sorted_partitions.pop()
	# set the disk identifier ( disk UUID )
	run_command_in_multicmd_with_path_check(["sgdisk", f'--disk-guid={partition_infos[src_disk_path]["disk_identifier"]}', image])
	# Create the filesystem, need to mount the image first if it is a image
	loop_device = None
	try:
		if not pathlib.Path(image).resolve().is_block_device():
			loop_device = create_loop_device(image)
			image = loop_device
		# Create the partitions
		delayed_commands = []
		for partition in sorted_partitions:
			start_sector = int(run_command_in_multicmd_with_path_check(["sgdisk", '--first-aligned-in-largest', image], extra_env=_C_LOCALE)[0].strip())
			end_sector = start_sector + int(partition_infos[partition]['size']/partition_infos[src_disk_path]['sector_size']) -1
			# Create the partition
			run_command_in_multicmd_with_path_check(["sgdisk", f'--new={partition}:{start_sector}:{end_sector}', image])
			# Copy the partition information
			delayed_commands.extend(write_partition_info(image, partition_infos,partition))
		# Fix the partition table
		run_command_in_multicmd_with_path_check(["sgdisk", '--verify', image])
		return delayed_commands, loop_device
	except Exception:
		# If setup fails after attaching a loop, detach it so the caller does not leak it.
		if loop_device:
			detach_loop_device(loop_device)
		raise

def resize_image(image_name, image_file_size):
	"""Resize the image file to the calculated size."""
	# use truncate to create a file if the image is not a block device
	if not pathlib.Path(image_name).resolve().is_block_device():
		image_file_size = format_bytes(image_file_size,to_int=True)
		try:
			run_command_in_multicmd_with_path_check(["truncate", f'--size={image_file_size}', image_name],strict=True)
		except Exception:
			# use python native method to allocate the space
			eprint("truncate not available, using python native method to allocate space")
			with open(image_name, 'wb') as f:
				f.seek(image_file_size-1)
				f.write(b'\0')

# Filesystem overhead reserved when sizing a -di dest image. Also the floor for
# -dis: an image smaller than this cannot hold a journal plus the copy.
_FS_IMAGE_SLAG = 50 * 1024 * 1024
_IMAGE_BLOCK = 4096

def _align_up(size, block=_IMAGE_BLOCK):
	"""Round size up to a multiple of block; already-aligned values are unchanged."""
	size = int(size or 0)
	if size <= 0:
		return 0
	return ((size + block - 1) // block) * block

def _split_dest_image_name(dest_image, index):
	"""Name for split volume `index` (0 is the original dest_image path)."""
	if index <= 0:
		return dest_image
	for suffix in ('.img', '.iso'):
		if dest_image.endswith(suffix):
			return dest_image[:-len(suffix)] + f'_{index}' + suffix
	return dest_image + f'_{index}'

def _plan_dest_images(init_size, dest_image_size=0):
	"""Plan how many -di images to create and how large each one is.

	Returns:
		tuple: (number_of_images, image_file_size). image_file_size is 4 KiB aligned.
	Raises:
		RuntimeError: dest_image_size is positive but too small to hold a filesystem.
	"""
	init_size = int(init_size or 0)
	needed = _align_up(int(1.05 * init_size + _FS_IMAGE_SLAG))
	dest_image_size = int(dest_image_size or 0)
	if dest_image_size <= 0:
		return 1, needed
	dest_image_size = _align_up(dest_image_size)
	if dest_image_size >= needed:
		return 1, dest_image_size
	if dest_image_size <= _FS_IMAGE_SLAG:
		raise RuntimeError(f"Destination image size too small: Estimated file system bloat size {format_bytes(_FS_IMAGE_SLAG)}B is larger than destination image size {format_bytes(dest_image_size)}B, exiting")
	# 50 MiB slag pads a *single* auto-sized image. Per split volume the journal
	# is much smaller; reserving 50 MiB on every piece over-counted volumes.
	usable = int(dest_image_size * 0.90) - (16 * 1024 * 1024)
	if usable <= 0:
		raise RuntimeError(f"Destination image size too small: Estimated file system bloat size {format_bytes(_FS_IMAGE_SLAG)}B is larger than destination image size {format_bytes(dest_image_size)}B, exiting")
	payload = max(int(1.05 * init_size), 1)
	number_of_images = (payload + usable - 1) // usable
	return max(number_of_images, 1), dest_image_size

def _apply_dis_to_dd_image(computed_size, dest_image_size, dest_path):
	"""Honor -dis for a -dd image-file destination.

	Block devices cannot be truncated to -dis, so they keep computed_size.
	A requested size below the packed partition layout cannot fit, so that
	is an error rather than a silent ignore.
	"""
	dest_image_size = int(dest_image_size or 0)
	if dest_image_size <= 0:
		return computed_size
	dest_image_size = _align_up(dest_image_size)
	dest_is_file = True
	if dest_path:
		try:
			if pathlib.Path(dest_path).exists() and pathlib.Path(dest_path).resolve().is_block_device():
				dest_is_file = False
		except Exception:
			dest_is_file = True
	if not dest_is_file:
		print(f"Ignoring dest_image_size {format_bytes(dest_image_size)}B: destination {dest_path} is a block device.")
		return computed_size
	if dest_image_size < computed_size:
		raise RuntimeError(f"Destination image size too small: Packed clone needs {format_bytes(computed_size)}B but dest_image_size is {format_bytes(dest_image_size)}B, exiting")
	print(f"Padding -dd destination image to dest_image_size {format_bytes(dest_image_size)}B (clone layout {format_bytes(computed_size)}B)")
	return dest_image_size

def is_device(path):
	mode = os.stat(path).st_mode
	return stat.S_ISCHR(mode) or stat.S_ISBLK(mode)

def get_mount_table():
	"""
	Get the mount table of the system.

	Returns:
		dict: A dictionary where the keys are device paths and the values are lists of mount points.
	"""
	mount_table = {}
	with open('/proc/mounts', 'r') as f:
		for line in f:
			parts = line.split()
			if len(parts) >= 2:
				mount_table.setdefault(parts[0], []).append(parts[1])
	return mount_table

#%% -- Symbolic Links --
def create_sym_links(symLinks,exclude=None,link_tracking=False,link_dirs=None):
	"""Recreate symLinks ({src link: [dest paths]}). Returns a (target's destination path, target's source path, link's
	destination path) triple for each relative link whose target is excluded, and with link tracking (-lt) for each one
	whose target exists at the source (copied now or already there): -rme keeps those targets, except where the copy
	itself covers them (_kept_paths). link_dirs (a set, in a move): each directory a link was made in is added, for the
	move's directory fsyncs (_sync_moved_directories). Each link's directory goes through the door first (_dest_directory,
	nothing made), and the link already there, its removal, the test for what holds its place and the new link go relative
	to what the door holds."""
	global RANDOM_DESTINATION_SELECTION
	tracked_targets = set()
	if len(symLinks) == 0:
		return tracked_targets
	nestedSymLinks = {}
	counter = 0
	print(f"\nFound Symbolic Links:   {len(symLinks)}")
	if link_tracking:
		print("Link tracking: copying the missing targets of relative links ...\n")
	#print(symLinks)
	start_time = time.monotonic()
	for src, dests in symLinks.items():
		_check_stop()
		# The door's result for each destination's directory (none: missing, or refused, kept in refusals); closed below.
		bases, refusals = {}, {}
		try:
			src = os.path.normpath(src)
			dests = [os.path.normpath(d) for d in dests]
			if exclude and is_excluded(src,exclude):
				print(f"\n{pformat(src)} is excluded, skipping...")
				continue
			# Read before anything is removed: a link whose destination is the link itself (a list, a move onto its own
			# place) would be gone, and unreadable.
			linkedTargetFile = os.readlink(src)
			source_stat = os.lstat(src)
			for d in dests:
				try:
					bases[d] = _dest_directory(os.path.dirname(d), make=False, source=os.path.dirname(os.path.abspath(src)))
				except (_Refused, OSError) as e:
					refusals[d] = e
			def look(d):
				"""What is at d now, by its name in what the door holds (lstat); None for nothing there."""
				if bases.get(d) is None:
					return None
				try:
					return os.lstat(*_at(bases[d], os.path.basename(d)), **_fd_of(bases[d]))
				except OSError:
					return None
			looks = {d: look(d) for d in dests}
			# The link is already there: it is its own copy.
			dests = [d for d in dests if not (looks[d] is not None and stat.S_ISLNK(looks[d].st_mode) and os.path.samestat(source_stat, looks[d]))]
			if not dests:
				continue
			for d in dests:
				if looks[d] is not None and stat.S_ISLNK(looks[d].st_mode):
					os.remove(*_at(bases[d], os.path.basename(d)), **_fd_of(bases[d]))
			# A real file or directory already at the link's path (in any dest: a file lives in one) keeps its place.
			occupied = next(((d, st) for d, st in ((d, look(d)) for d in dests) if st is not None), None)
			if occupied is not None:
				# An error, not a note: the destination does not match the source, and a move (-rm) must not go on.
				eprint(f"Failed to create symbolic link: {pformat(occupied[0])} is a {'directory' if stat.S_ISDIR(occupied[1].st_mode) else 'file'} "
					f"in the destination, skipping the link {pformat(src)}")
				continue
			# Determine if the link is a absolute link or relative link
			dest = ''
			# The link's target at the source (relative links); keep_target: its destination copy belongs to the copy (the
			# target exists at the source, with tracking) or is the copy of an excluded path, and -rme keeps it (_kept_paths).
			sourceLinkedFile = None
			keep_target = False
			if not os.path.isabs(linkedTargetFile):
				# Files are spread over the dests: a relative link goes where its target is, so that it resolves.
				destLinkedFiles = [os.path.join(os.path.dirname(d), linkedTargetFile) for d in dests]
				dest = next((d for d, destLinkedFile in zip(dests, destLinkedFiles) if os.path.exists(destLinkedFile)), '')
				# The target by the link's own spelling with '..' resolved, so -e patterns match it and the tracked tree;
				# used only when the kernel resolves the link to that same file (no '..' after a symlinked directory).
				linked = os.path.join(os.path.dirname(src), linkedTargetFile)
				sourceLinkedFile = os.path.normpath(linked)
				# Not copied, but an earlier copy of it in the destination is kept, as for any excluded path: with or
				# without tracking (an -e protects it either way).
				excluded_target = bool(exclude) and _excluded_or_below(sourceLinkedFile, exclude,
					stop=os.path.commonpath([sourceLinkedFile, os.path.dirname(os.path.abspath(src))]))
				# Two realpath calls, so only when tracking or an excluded target needs it.
				through_symlinked_dir = bool(link_tracking or excluded_target) and os.pardir in linkedTargetFile.split(os.sep) and os.path.realpath(sourceLinkedFile) != os.path.realpath(linked)
				keep_target = excluded_target and not through_symlinked_dir
				if link_tracking:
					note = None
					if through_symlinked_dir:
						# Said always: an earlier copy at the spelled path is not kept for this link.
						print(f"\n{pformat(src)} -> {pformat(linkedTargetFile)} goes through a symlinked directory: its target is not tracked", file=sys.stderr)
					elif keep_target:
						note = f"\n{pformat(sourceLinkedFile)}, linked by {pformat(src)}, is excluded: not copying it"
					elif not os.path.exists(sourceLinkedFile):
						# A dangling link is copied as it is: a note, not an error (eprint would record any path with a ':').
						note = f"\nFile {sourceLinkedFile} which is linked by {src} doesn't exist! \nSkipping copying original file..."
					else:
						keep_target = True
						if not dest:
							# we also copy the pointed file if the file doesn't exist
							destLinkedFiles = [os.path.normpath(destLinkedFile) for destLinkedFile in destLinkedFiles]
							_, _ , rtnSymLinks , _ = copy_files_serial(sourceLinkedFile, destLinkedFiles,exclude=exclude)
							nestedSymLinks.update(rtnSymLinks)
							dest = next((d for d, destLinkedFile in zip(dests, destLinkedFiles) if os.path.lexists(destLinkedFile)), '')
					# Only worth saying when the target is not already at a destination (a copy would have been tried).
					if note and not dest:
						print(note, file=sys.stderr)
			while dests:
				if not dest:
					if RANDOM_DESTINATION_SELECTION:
						idx = random.randrange(len(dests))
					else:
						idx = 0
					dest = dests.pop(idx)
				elif dest in dests:
					dests.remove(dest)
				try:
					if dest in refusals:
						raise refusals.pop(dest)
					if bases.get(dest) is None:
						# The link's directory, made as a file's is (-nds, or a link reached through a linked directory, has none
						# yet); under -ncd, a missing one by its path, which the link's make meets.
						bases[dest] = _dest_directory(os.path.dirname(dest), make=not NO_CREATE_DIR, source=os.path.dirname(os.path.abspath(src)))
					base = bases[dest] if bases[dest] is not None else os.path.dirname(dest)
					# A relative target is resolved from the link's directory (Windows makes directory and file links apart).
					os.symlink(linkedTargetFile, *_at(base, os.path.basename(dest)),
						target_is_directory=os.path.isdir(os.path.join(os.path.dirname(src), linkedTargetFile)), **_fd_of(base))
					if link_dirs is not None:
						link_dirs.add(os.path.dirname(os.path.abspath(dest)))
					if keep_target:
						tracked_targets.add((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(dest)), linkedTargetFile)),
							sourceLinkedFile, os.path.abspath(dest)))
					break
				except Exception as e:
					# Notes: only 'Failed to create symbolic link' below, when every destination failed, is the error.
					if isinstance(e, _Refused):
						_say_refused(e, f'the link {src}')
					print(f'Could not create symbolic link from {linkedTargetFile} to {dest}', file=sys.stderr)
					if dests:
						print("Trying next destination...", file=sys.stderr)
						dest = ''
						continue
					else:
						eprint("Failed to create symbolic link: All destinations failed, skipping...")
						break
			counter += 1
			# print the progress bar with the total count and the speed in F/s
			prefix = f'{counter} Symbolic Links Created'
			suffix = f'{counter / (time.monotonic() - start_time):.2f} F/s'
			multiCMD.print_progress_bar(counter, len(symLinks), prefix=prefix, suffix=suffix)
			# we catch the file name too long exception
		except OSError as e:
			eprint(f"""Create symlink os error:
Exception caught! Possibly file name too long!
{e}
{src} -> {dests}
Skipping...""")
			continue
		except Exception as e:
			eprint(f"""Create symlink general error:
Exception caught!
{e}
{src} -> {dests}
Skipping...""")
			continue
		finally:
			for base in bases.values():
				_close_base(base)


	endTime = time.monotonic()
	print(f"Time taken:             {endTime-start_time:0.4f} seconds")
	if len(nestedSymLinks) > 0:
		print(f"\nNested Symbolic Links:   {len(nestedSymLinks)}")
		tracked_targets |= create_sym_links(nestedSymLinks,exclude=exclude,link_tracking=link_tracking,link_dirs=link_dirs)
	return tracked_targets

#%% -- File list --
def _natural_key(text):
	"""Sort key splitting digit runs into ints. Odd positions of re.split are always ASCII digit runs."""
	return [int(part) if index % 2 else part.lower() for index, part in enumerate(re.split('([0-9]+)', text))]

def natural_sort(L):
	"""
	Sorts a list of strings naturally, considering both numeric and alphabetic characters.

	Args:
		L (list): The list of strings to be sorted.

	Returns:
		list: The sorted list of strings.

	Examples:
		>>> natural_sort(['file10.txt', 'file2.txt', 'file1.txt'])
		['file1.txt', 'file2.txt', 'file10.txt']
	"""
	return sorted(L, key=_natural_key)

#%% -- Format --
def format_bytes(size, use_1024_bytes=None, to_int=False, to_str=False,str_format='.2f'):
	"""
	Format the size in bytes to a human-readable format or vice versa.
	From hpcp: https://github.com/yufei-pan/hpcp

	Args:
		size (int or str): The size in bytes or a string representation of the size.
		use_1024_bytes (bool, optional): Whether to use 1024 bytes as the base for conversion. If None, it will be determined automatically. Default is None.
		to_int (bool, optional): Whether to convert the size to an integer. Default is False.
		to_str (bool, optional): Whether to convert the size to a string representation. Default is False.
		str_format (str, optional): The format string to use when converting the size to a string. Default is '.2f'.

	Returns:
		int or str: The formatted size based on the provided arguments.

	Examples:
		>>> format_bytes(1500, use_1024_bytes=False)
		'1.50 K'
		>>> format_bytes('1.5 GiB', to_int=True)
		1610612736
		>>> format_bytes('1.5 GiB', to_str=True)
		'1.50 Gi'
		>>> format_bytes(1610612736, use_1024_bytes=True, to_str=True)
		'1.50 Gi'
		>>> format_bytes(1610612736, use_1024_bytes=False, to_str=True)
		'1.61 G'
	"""
	return multiCMD.format_bytes(size, use_1024_bytes=use_1024_bytes, to_int=to_int, to_str=to_str,str_format=str_format)

def format_time(seconds):
	"""
	Format the time in seconds to a short, human-readable format.

	Args:
		seconds (int): The time in seconds.

	Returns:
		str: The formatted time in a human-readable format.

	Examples:
		>>> format_time(0)
		'0s'
		>>> format_time(65)
		'1m5s'
		>>> format_time(3661)
		'1h1m1s'
	"""
	try:
		seconds = int(seconds)
	except Exception:
		return seconds
	seconds_in_minute = 60
	seconds_in_hour = 60 * seconds_in_minute
	seconds_in_day = 24 * seconds_in_hour
	seconds_in_month = 30 * seconds_in_day
	seconds_in_year = 365 * seconds_in_day
	years, seconds = divmod(seconds, seconds_in_year)
	months, seconds = divmod(seconds, seconds_in_month)
	days, seconds = divmod(seconds, seconds_in_day)
	hours, seconds = divmod(seconds, seconds_in_hour)
	minutes, seconds = divmod(seconds, seconds_in_minute)
	parts = [
		f"{years:.0f}Y" if years else "",
		f"{months:.0f}M" if months else "",
		f"{days:.0f}D" if days else "",
		f"{hours:.0f}h" if hours else "",
		f"{minutes:.0f}m" if minutes else "",
		f"{seconds:.0f}s" if seconds else "0s",
	]
	return "".join(parts)

#%% -- Hash --
@functools.lru_cache(maxsize=None)
def hash_file(path,size = ...,full_hash=False):
	global xxhash_available
	global HASH_SIZE
	if HASH_SIZE <= 0:
		# Do not hash
		return ''
	if size == ...:
		size = os.path.getsize(path)
	with open(path, 'rb') as f:
		return _hash_stream(f, size, full_hash)

def _hash_stream(f, size, full_hash):
	"""hash_file's hash of f, an open binary file of size bytes: its last HASH_SIZE bytes, or all of it (full_hash)."""
	hasher = xxhash.xxh64() if xxhash_available else hashlib.blake2b()
	if not full_hash:
		# Only hash the last hash_size bytes
		f.seek(-min(HASH_SIZE,size), os.SEEK_END)
	for chunk in iter(lambda: f.read(4096), b''):
		hasher.update(chunk)
	return hasher.hexdigest()

def is_file_identical(src_path, dest_path,src_size,full_hash=False,compare_metadata=False,src_st=None,dest_st=None,dest_open=None):
	"""src_st, dest_st: their stats, when the caller has them (copy_file's looks): nothing is stat'ed again. dest_open: a
	callable that opens the destination as a binary file (copy_file's: relative to the directory the door holds), read
	uncached; without it the destination is read by its path, through hash_file's cache, as the source always is."""
	dst_size = dest_st.st_size if dest_st is not None else os.path.getsize(dest_path)
	# try to find the mtime are different
	if src_size != dst_size:
		return False
	src_st, dst_st = src_st or os.stat(src_path), dest_st or os.stat(dest_path)
	# A landing at a time its filesystem could not keep (a source from before 1980, on vfat at or after it) says nothing of its
	# contents: both are read in full, as the move's backstop does, for those files only. Two reads instead of a copy on
	# every rerun; reading on every other time that differs would read a changed file twice before copying it anyway.
	floored = src_st.st_mtime_ns < _FAT_TIME_FLOOR_NS <= dst_st.st_mtime_ns
	if not floored and not _mtimes_agree(src_st.st_mtime_ns, dst_st.st_mtime_ns):
		return False
	# A node has no contents to compare, and opening one is an act of its own (a FIFO waits for a writer): nodes of one kind for
	# the same device are the same, and a changed device number is not.
	if not (stat.S_ISREG(src_st.st_mode) and stat.S_ISREG(dst_st.st_mode)):
		return not floored and stat.S_IFMT(src_st.st_mode) == stat.S_IFMT(dst_st.st_mode) and src_st.st_rdev == dst_st.st_rdev
	if floored:
		if not _same_contents(src_path, dest_path, dest_open):
			return False
	elif hash_file(src_path,src_size,full_hash) != (hash_file(dest_path,dst_size,full_hash) if dest_open is None else
			'' if HASH_SIZE <= 0 else _hash_opened(dest_open, dst_size, full_hash)):
		return False
	if compare_metadata:
		src_st = os.lstat(src_path)
		dst_st = os.lstat(dest_path)
		if (src_st.st_mode, src_st.st_uid, src_st.st_gid) != (dst_st.st_mode, dst_st.st_uid, dst_st.st_gid):
			return False
		src_xattr = _xattr_digest(src_path)
		dst_xattr = _xattr_digest(dest_path)
		if src_xattr is not None and dst_xattr is not None and src_xattr != dst_xattr:
			return False
	return True

def _hash_opened(opener, size, full_hash):
	"""hash_file's hash of the file opener() opens: not cached, as what it opens is not named by a path (a key of
	/proc/self/fd/D/NAME would come back for another directory once D is reused)."""
	with opener() as f:
		return _hash_stream(f, size, full_hash)

def get_file_repr(filename,append_hash=False,full_hash=False):
	if append_hash:
		if os.path.islink(filename):
			return f'{filename}:{os.readlink(filename)}'
		return f'{filename}:{hash_file(filename,os.path.getsize(filename),full_hash)}'
	return filename

#%% -- Path --
def trim_paths(paths, baseDir):
	"""
	Convert a set of absolute paths to relative paths based on a base directory.
	
	Args:
		paths (set or list): A collection of absolute file paths.
		baseDir (str): The base directory to make paths relative to.
	
	Returns:
		set: A set of file paths, each relative to the parent directory of baseDir.
	
	Example:
		>>> sorted(trim_paths({'/home/user/project/file1.py', '/home/user/project/file2.py'}, '/home/user/project/main.py'))
		['file1.py', 'file2.py']
	"""
	return set([os.path.relpath(path,os.path.dirname(baseDir)) for path in paths])

#%% ---- Generate File List ----
_get_file_list_cache = {}

def _get_file_list_cache_key(path, exclude, append_hash, full_hash, boundary=None):
	exclude_key = frozenset(exclude) if isinstance(exclude, (list, set)) else exclude
	return (path, exclude_key, append_hash, full_hash, boundary)

def get_file_list(path, exclude=None, append_hash=False, full_hash=False, max_workers=56, drop_cache=False, parallel_file_listing=True, boundary=None):
	"""Unified file list getter with a single cache. Dispatches to get_file_list_serial or get_file_list_parallel based on serial.
	Listing only reads: -rm removes by descriptors (process_remove), not while listing. boundary (-rme's destination
	listing, _filesystem_boundary): stop at another filesystem below the root, and take no sizes."""
	cache_key = _get_file_list_cache_key(path, exclude, append_hash, full_hash, boundary)
	if drop_cache:
		_get_file_list_cache.pop(cache_key, None)
	if cache_key in _get_file_list_cache:
		return _get_file_list_cache[cache_key]
	start_time = time.monotonic()
	# Said for a directory only: a list (-sfl) gives one source per file, two lines each say nothing.
	announce = os.path.isdir(path)
	if announce:
		print(f"Getting file list for {path}")
	if parallel_file_listing:
		file_list, links, size, folders = get_file_list_parallel(path, max_workers=max_workers, exclude=exclude, append_hash=append_hash, full_hash=full_hash, boundary=boundary)
	else:
		file_list, links, size, folders = get_file_list_serial(path, exclude=exclude, append_hash=append_hash, full_hash=full_hash, boundary=boundary)
	result = (file_list, links, size, folders)
	_get_file_list_cache[cache_key] = result
	if announce:
		print(f"Time taken to get file list: {time.monotonic() - start_time:0.4f} seconds")
	return result

def _file_list_rate_limit(limiter_state, increment=1):
	"""Throttle file-list iteration based on FILES_RATE_LIMIT."""
	if not limiter_state:
		return
	increment_added = False
	while True:
		if not increment_added:
			limiter_state['count'] += increment
			increment_added = True
		elapsed = time.monotonic() - limiter_state['start_time']
		limit = limiter_state['limit']
		allowed = limit * elapsed
		count = limiter_state['count']
		if count <= allowed:
			return
		wait_for = (count - allowed) / limit
		time.sleep(min(0.1, max(0.01, wait_for)))

def _makedirs(name, mode=0o777, exist_ok=False):
	"""os.makedirs(name, mode, exist_ok) without its recursion (one frame per missing level: a tree deeper than about 1000
	levels raised RecursionError): the missing parents are made from the top (default mode; one made meanwhile is
	fine), then name with mode, unless it ends in '.' (made with them, as in os.makedirs); an existing name is fine only
	with exist_ok when it is a directory."""
	missing = []
	head = os.path.dirname(name.rstrip(os.sep))
	while head and not os.path.exists(head):
		missing.append(head)
		parent = os.path.dirname(head.rstrip(os.sep))
		if parent == head:
			break
		head = parent
	for directory in reversed(missing):
		try:
			os.mkdir(directory)
		except FileExistsError:
			pass
	if missing and os.path.basename(name.rstrip(os.sep)) == os.curdir:
		# xxx/newdir/. exists once xxx/newdir does
		return
	try:
		os.mkdir(name, mode)
	except OSError:
		if not exist_ok or not os.path.isdir(name):
			raise

def _walk(top, device=None, foreign=None):
	"""os.walk(top) top-down, symlinks to directories listed but not entered, unreadable directories skipped, without
	recursion (os.walk recursed once per level before Python 3.12). The caller may prune dirnames in place. With
	device (-rme), each directory below top is opened as it is entered (_enter_directory, which refuses a symlink) and
	one then on another device (a filesystem mounted on entering it: an NFSv4 submount, an automount; or, not opened,
	by its lstat) is not walked; foreign, when given, is called with its path. Each is read through its descriptor
	(_scan_target), and a symlink is among the filenames (its d_type: no lstat)."""
	enter = device is not None
	pending = [top]
	while pending:
		dirpath = pending.pop()
		dirnames, filenames = [], []
		fd = None
		try:
			if enter:
				# top is the destination (its device is device): opened, not checked.
				if dirpath is top:
					fd = os.open(dirpath, _DIRECTORY_FLAGS)
				else:
					try:
						fd = _enter_directory(dirpath, device)
					except OSError:
						# Not opened (EACCES, EIO, a mount that failed): its device as lstat says, as before 9.77.
						if foreign is not None and _lstat_device(dirpath) not in (None, device):
							foreign(dirpath)
						continue
					if fd is None:
						if foreign is not None:
							foreign(dirpath)
						continue
			target = _scan_target(fd, dirpath) if fd is not None else dirpath
			if target is None:
				continue
			with os.scandir(target) as entries:
				for entry in entries:
					try:
						is_dir = entry.is_dir(follow_symlinks=not enter)
					except OSError:
						is_dir = False
					(dirnames if is_dir else filenames).append(entry.name)
		except OSError:
			continue
		finally:
			if fd is not None:
				os.close(fd)
		yield dirpath, dirnames, filenames
		for name in reversed(dirnames):
			path = os.path.join(dirpath, name)
			if enter or not os.path.islink(path):
				pending.append(path)

def _rmtree(path, exclude=None, report=False):
	"""shutil.rmtree(path) without recursion (it recursed once per level before Python 3.13), by directory descriptors
	(_remove_tree_fd): a missing path raises FileNotFoundError, and a tree that is not gone raises the first error, with
	the full path as its filename (an ENOTEMPTY one when nothing failed). With report (delete_file_bulk) every error is
	recorded instead ('Remove process exception'). exclude: patterns matched against the paths as path spells them;
	what they match stays, with the directories holding it."""
	path = os.fspath(path)
	# As given ('link/..' is the link target's parent, as for shutil.rmtree; no lexical abspath); '' is nothing.
	absolute = path.rstrip(os.sep) or path
	if not path or not os.path.lexists(absolute):
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
	_, _, errors = _remove_tree_fd(absolute, _directory_identity(absolute), exclude, remove_files=True, spelled=path)
	if report:
		for failed, error in errors:
			eprint(f'Remove process exception:\n Deleting {failed} generated an exception: {error}')
		return
	if errors:
		failed, error = errors[0]
		raise type(error)(error.errno, error.strerror, failed) if isinstance(error.errno, int) else error
	if not exclude and os.path.lexists(absolute):
		raise OSError(errno.ENOTEMPTY, os.strerror(errno.ENOTEMPTY), path)

def _scan_dir(root, label=None):
	"""os.scandir(root)'s entries (root a path or a directory descriptor; label names it). A directory that cannot be
	opened or read on (EACCES, EIO) is recorded as a 'Listing error' and gives the entries read so far, so the rest of
	the tree is still listed."""
	try:
		with os.scandir(root) as entries:
			yield from entries
	except OSError as e:
		eprint(f'Listing error: {label if label is not None else root}: {e}')

# A directory opened to be listed, as given ('link/' is the link's target).
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)

def _lstat_device(path):
	"""path's st_dev as lstat says (a filesystem mounted on entering it not mounted yet: its parent's), or None."""
	try:
		return os.lstat(path).st_dev
	except OSError:
		return None

def _enter_directory(path, device):
	"""A descriptor of directory path opened as a listing enters it, or None when it is then on another device than
	device: a filesystem mounted on entering it (an automount, an NFSv4 submount) has its parent's device and no
	mountinfo entry until it is opened (lstat does not mount it). Raises OSError when it cannot be opened: ELOOP or
	ENOTDIR when it is no longer a directory (_BELOW_FLAGS: a symlink is not followed)."""
	fd = os.open(path, _BELOW_FLAGS)
	try:
		if os.fstat(fd).st_dev == device:
			return fd
	except BaseException:
		os.close(fd)
		raise
	os.close(fd)
	return None

def _named_special(path):
	"""True when path itself (lstat) is a FIFO, socket or device node."""
	try:
		mode = os.lstat(path).st_mode
	except OSError:
		return False
	return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISBLK(mode) or stat.S_ISCHR(mode)

def _special_file_kind(entry):
	"""'FIFO', 'socket', 'block device' or 'character device' for such a scandir entry, else None: a regular file,
	directory or link, or an entry deleted since the directory was read (without d_type, is_file() stats it, and a
	deleted entry is then neither file nor directory nor link; it is gone, nothing to copy)."""
	try:
		mode = entry.stat(follow_symlinks=False).st_mode
	except FileNotFoundError:
		return None
	for is_kind, kind in ((stat.S_ISFIFO, 'FIFO'), (stat.S_ISSOCK, 'socket'), (stat.S_ISBLK, 'block device'), (stat.S_ISCHR, 'character device')):
		if is_kind(mode):
			return kind
	return None

def _visible_mount_points(entries=None):
	"""The mount points of mountinfo (entries) that path lookup reaches: a mount hidden under a later one is left out."""
	entries = _mountinfo_entries() if entries is None else entries
	hidden = _hidden_mounts(entries)
	return frozenset(entry['mount_point'] for index, entry in enumerate(entries) if index not in hidden)

def _filesystem_boundary(listed_root, real_root, visible=None):
	"""What -rme leaves alone below a destination, as rsync -x: (the visible mount points strictly below real_root,
	spelled as a listing of listed_root names them (the destination as given: 'dst/share' for 'dst/'), real_root's
	device, None on Windows, whose DirEntry.stat reports no device). A listing given it skips those mount points (bind
	mounts, of files too) and the directories on another device (a btrfs subvolume, a mount mountinfo does not show)."""
	visible = _visible_mount_points() if visible is None else visible
	real_prefix = real_root.rstrip(os.sep) + os.sep
	listed_prefix = os.path.join(listed_root, '')
	points = frozenset(listed_prefix + point[len(real_prefix):] for point in visible if point.startswith(real_prefix))
	device = None
	if os.name != 'nt':
		# As opened (mounted, should it be mounted on entering it): what the listing and the walk find there.
		identity = _directory_identity(real_root)
		device = identity[0] if identity else None
	return (points, device)

def get_file_list_serial(root,exclude=None,append_hash=False,full_hash=False,recurse=True,_rate_limiter=None,respect_file_rate_limit=True,boundary=None,known_dir=False):
	"""(files, links, size, folders) below root. Iterative: a tree deeper than Python's recursion limit (about 1000
	levels) is listed too; each directory below root gets the checks a call for it as root would (length, exclude,
	link, file, directory). recurse False lists root's entries only (its directories in folders, unchecked: a listing of
	each says). boundary: see get_file_list. known_dir: root comes from a parent's listing (-pfl), which said directory:
	checked as a directory below root is."""
	global FILES_RATE_LIMIT
	if _rate_limiter is None and respect_file_rate_limit and FILES_RATE_LIMIT > 0:
		_rate_limiter = {'start_time': time.monotonic(), 'count': 0, 'limit': FILES_RATE_LIMIT}
	file_list = set()
	links = set()
	folders = set()
	size = 0
	iteration = 0
	start_time = time.monotonic()
	globalStartTIme = start_time
	# -rme's destination listing: each directory below the root is opened before anything else (_enter_directory), so a
	# filesystem mounted on entering it (an NFSv4 submount, an automount: its parent's device, and no mountinfo entry,
	# until then) is mounted and its device checked before anything in it is read; another device is another
	# filesystem, left alone. The root is the destination (its device is the boundary's): opened, not checked. Each is
	# read through its descriptor (_scan_target).
	enter = boundary is not None and boundary[1] is not None
	pending = [root]
	while pending:
		current = pending.pop()
		# skip if path is longer than 4096 characters: not listed, so not copied (an error: a move must not delete it).
		if len(current) > 4096:
			eprint(f'Listing error: {current}: path longer than 4096 characters, not listed')
			continue
		scan_fd = enter_error = None
		if current is root and not known_dir:
			if exclude and is_excluded(current,exclude):
				continue
			is_link, is_file, is_dir = os.path.islink(current), None, None
		else:
			if enter:
				try:
					scan_fd = _enter_directory(current, boundary[1])
					if scan_fd is None:
						continue
				except OSError as e:
					# No longer a directory (ELOOP, ENOTDIR: the lstat below says what it is), or one that cannot be
					# opened, reported once the lstat says directory.
					enter_error = e
			if scan_fd is not None:
				is_link, is_file, is_dir = False, False, True
			else:
				# Its parent's entry passed exclude and said directory: one lstat sees a swap since (to a link, a file).
				try:
					mode = os.lstat(current).st_mode
				except OSError:
					mode = 0
				is_link, is_file, is_dir = stat.S_ISLNK(mode), stat.S_ISREG(mode), stat.S_ISDIR(mode)
		if is_link:
			_file_list_rate_limit(_rate_limiter)
			links.add(get_file_repr(current,append_hash,full_hash))
			continue
		if is_file or (is_file is None and os.path.isfile(current)):
			_file_list_rate_limit(_rate_limiter)
			size += get_file_size(current)
			file_list.add(get_file_repr(current,append_hash,full_hash))
			continue
		if not (is_dir or (is_dir is None and os.path.isdir(current))):
			# A FIFO, socket or device the operator named is a source, copied as cp -a copies it (one met in a tree is
			# skipped by the walk); never hashed: opening it is an act of its own.
			if not (current == root and _named_special(current)):
				eprint(f'Source path type error: {current} is not a file or directory')
			file_list.add(current)
			continue
		folders.add(current)
		if enter and scan_fd is None:
			if enter_error is None:
				try:
					scan_fd = os.open(current, _DIRECTORY_FLAGS)
				except OSError as e:
					enter_error = e
			if enter_error is not None:
				eprint(f'Listing error: {current}: {enter_error}')
				continue
		try:
			target = _scan_target(scan_fd, current) if scan_fd is not None else current
			if target is None:
				eprint(f'Listing error: {current}: it was replaced since it was opened')
				continue
			for entry in _scan_dir(target, current):
				path = os.path.join(current, entry.name) if scan_fd is not None else entry.path
				# update the progress bar every 0.5 seconds
				currentTime = time.monotonic()
				if currentTime - start_time > 0.5:
					start_time = currentTime
					# use the time passed as the iteration number
					iteration = int(currentTime - globalStartTIme)
					suffix = ''
					if file_list:
						suffix += f'Files: {format_bytes(len(file_list),use_1024_bytes=False,to_str=True)} '
					if links:
						suffix += f'Links: {format_bytes(len(links),use_1024_bytes=False,to_str=True)} '
					if folders:
						suffix += f'Folders: {format_bytes(len(folders),use_1024_bytes=False,to_str=True)} '
					if size:
						suffix += f'Size: {format_bytes(size)}B '
					if _rate_limiter:
						suffix += f'| Overall: {format_bytes(_rate_limiter["count"],use_1024_bytes=False,to_str=True)} Files'
					# if the root is longer than 50 characters, we only show the last 50 characters
					multiCMD.print_progress_bar(iteration=iteration, total=0, prefix=f'{current}'[-50:], suffix=suffix)
				if exclude and is_excluded(path,exclude):
					continue
				try:
					# -rme's destination listing: another filesystem below the destination is left alone, unlisted: a
					# mount point by name here (nothing stat'ed), a directory on another device once entered (above).
					if boundary is not None and path in boundary[0]:
						continue
					if entry.is_symlink():
						_file_list_rate_limit(_rate_limiter)
						if boundary is None:
							size += get_file_size(path)
						links.add(get_file_repr(path,append_hash,full_hash))
					elif entry.is_file(follow_symlinks=False):
						_file_list_rate_limit(_rate_limiter)
						if boundary is None:
							size += get_file_size(path)
						file_list.add(get_file_repr(path,append_hash,full_hash))
					elif entry.is_dir(follow_symlinks=False):
						if recurse:
							pending.append(path)
						else:
							folders.add(path)
					else:
						kind = _special_file_kind(entry)
						if kind:
							# Not copied or compared; a move (-rm) leaves it in place (remove_moved_sources).
							eprint(f'Special file skipped: {path} is a {kind}; hpcp does not copy FIFOs, sockets or device nodes')
				except OSError as e:
					# One entry that cannot be read (its hash, a link target) is recorded; the others are still listed.
					eprint(f'Listing error: {path}: {e}')
		finally:
			if scan_fd is not None:
				os.close(scan_fd)
	return frozenset(file_list), frozenset(links) , size, frozenset(folders - set(['.', '..']))

def get_file_list_parallel(path,max_workers=56,exclude=None,append_hash=False,full_hash=False,boundary=None):
	global FILES_RATE_LIMIT
	# skip if path is longer than 4096 characters (an error, as in get_file_list_serial)
	if len(path) > 4096:
		eprint(f'Listing error: {path}: path longer than 4096 characters, not listed')
		return frozenset() ,frozenset(),0,frozenset()
	if exclude and is_excluded(path,exclude):
		return frozenset() ,frozenset(),0,frozenset()
	if os.path.islink(path):
		return frozenset() ,frozenset([get_file_repr(path,append_hash,full_hash)]),0,frozenset()
	if os.path.isfile(path):
		realSize = get_file_size(path)
		return frozenset([get_file_repr(path,append_hash,full_hash)]) ,frozenset(), realSize,frozenset()
	#print(f'Getting file list for {path}')
	if os.path.isdir(path):
		file_list, link_list , size, folder_list = get_file_list_serial(
			path,
			exclude=exclude,
			append_hash=append_hash,
			full_hash=full_hash,
			recurse=False,
			respect_file_rate_limit=False,
			boundary=boundary,
		)
		file_list = set(file_list)
		link_list = set(link_list)
		folder_list = set(folder_list)
		folders_to_expand = deque(folder_list - set([path]))
		enqueued_folders = set(folders_to_expand)
		futures = {}
		listing_start_time = time.monotonic()
		listed_items = len(file_list) + len(link_list)
		# A folder's listing that crashed (this default): its folder stays, as before 9.77.
		no_result = (frozenset(), frozenset(), 0, frozenset())
		with _pool(max_workers) as executor:
			while folders_to_expand or futures:
				under_rate_limit = (
					FILES_RATE_LIMIT <= 0
					or listed_items <= FILES_RATE_LIMIT * (time.monotonic() - listing_start_time)
				)
				while folders_to_expand and len(futures) < max_workers and under_rate_limit:
					folder = folders_to_expand.popleft()
					futures[_submit_task(
						executor,
						get_file_list_serial,
						folder,
						exclude=exclude,
						append_hash=append_hash,
						full_hash=full_hash,
						recurse=False,
						respect_file_rate_limit=False,
						boundary=boundary,
						known_dir=True,
					)] = folder
					under_rate_limit = (
						FILES_RATE_LIMIT <= 0
						or listed_items <= FILES_RATE_LIMIT * (time.monotonic() - listing_start_time)
					)
				if not futures:
					if folders_to_expand and FILES_RATE_LIMIT > 0:
						time.sleep(0.05)
					continue
				done = _wait(futures, executor)
				for doneTask in done:
					folder_path = futures.pop(doneTask)
					result = _task_result(doneTask, no_result, 'Listing error')
					dir_file_list, dir_link_list, dir_size, dir_folder_list = result
					if result is not no_result and folder_path not in dir_folder_list:
						# Its listing did not list it as a directory (another filesystem once entered, -rme; a link or
						# a file since; one too long): not a folder, as in a serial listing.
						folder_list.discard(folder_path)
					file_list.update(dir_file_list)
					link_list.update(dir_link_list)
					size += dir_size
					listed_items += len(dir_file_list) + len(dir_link_list)
					for sub_folder in dir_folder_list:
						if sub_folder != folder_path and sub_folder not in enqueued_folders:
							folders_to_expand.append(sub_folder)
							enqueued_folders.add(sub_folder)
					folder_list.update(dir_folder_list)
			
	else:
		if not _named_special(path):
			eprint(f'Source path type error: {path} is not a file or directory')
		return frozenset([path]) ,frozenset(), 0,frozenset()
	return frozenset(file_list), frozenset(link_list) ,size, frozenset(folder_list - set(['.', '..']))


#%% ---- Delete Files ----
def delete_file_bulk(paths, exclude=None):
	total_size = 0
	start_time = time.monotonic()
	for path in paths:
		if exclude and is_excluded(path, exclude):
			continue
		if os.path.lexists(path):
			try:
				total_size += os.lstat(path).st_size
				if os.path.isdir(path) and not os.path.islink(path):
					# By descriptors, without recursion, every failure recorded; excluded entries (matched as path spells
					# them) stay, with the directories holding them.
					_rmtree(path, exclude, report=True)
				else:
					os.remove(path)
			except Exception as exc:
				eprint(f'Remove process exception:\n Deleting {path} generated an exception: {exc}')
	endTime = time.monotonic()
	return total_size, endTime - start_time

def delete_file_list_parallel(file_list, max_workers, verbose=False,files_per_job=1,init_size=0,remover=None):
	"""Delete file_list in pool jobs; remover (a worker body taking a list of paths, returning (bytes, seconds)) is
	delete_file_bulk unless given."""
	remover = remover or delete_file_bulk
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	total_files = len(file_list)
	file_list_iterator = iter(file_list)
	start_time = time.monotonic()
	last_refresh_time = start_time
	futures = {}
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.2
	files_per_job = max(1,files_per_job)
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=init_size,last_num_job_for_stats=max(1,max_workers // 2),process_word='Deleted',
							 suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	last_transition_time = time.monotonic()
	last_under_limit = True
	with _pool(max_workers) as executor:
		while file_list_iterator or futures:
			# counter = 0
			under_limit = apb.under_rate_limit()
			while file_list_iterator and len(futures) < max_scheduled_jobs and under_limit:
				delete_files = []
				try:
					for _ in range(round(files_per_job)):
						delete_files.append(next(file_list_iterator))
					future = _submit_task(executor, remover, delete_files)
					futures[future] = delete_files
					# counter += 1
				except StopIteration:
					if delete_files:
						future = _submit_task(executor, remover, delete_files)
						futures[future] = delete_files
					file_list_iterator = None

			done = _wait(futures, executor)

			current_iteration_total_run_time = 0
			for doneTask in done:
				delete_files = futures.pop(doneTask)
				deleted_files_count_this_run = len(delete_files)
				rmSize, rmTime = _task_result(doneTask, (0, 0), 'Remove process exception')
				current_iteration_total_run_time += rmTime
				apb.update(num_files=deleted_files_count_this_run,cpSize=rmSize,cpTime=rmTime,files_per_job=files_per_job)
			apb.scheduled_jobs = len(futures)
			apb.print_progress()
			if done:
				if verbose:
					print(f'\nAverage rmtime is {current_iteration_total_run_time / len(done):0.2f} for {len(done)} jobs with {deleted_files_count_this_run} files each')
				if file_list_iterator and deleted_files_count_this_run == round(files_per_job) and (current_iteration_total_run_time / len(done) > 5 or time.monotonic() - last_refresh_time > 5):
					files_per_job /= MAGIC_NUMBER
					if verbose:
						print(f'\nCompletion time is long, changing files per job to {files_per_job}')
				elif file_list_iterator and deleted_files_count_this_run == round(files_per_job) and current_iteration_total_run_time / len(done) < 1:
					files_per_job *= MAGIC_NUMBER
					if verbose:
						print(f'\nCompletion time is short, changing files per job to {files_per_job}')
				if files_per_job < 1:
					files_per_job = 1
			else:
				max_scheduled_jobs = max(1, max_scheduled_jobs -1)
				if not under_limit:
					if verbose:
						print('\nWe had hit the rate limit and no job is running currently, setting files per job to 1')
					files_per_job = 1
				time.sleep(apb.refresh_interval)
			if under_limit != last_under_limit:
				last_transition_time = time.monotonic()
				last_under_limit = under_limit
			elif time.monotonic() - last_transition_time > 1:
				if under_limit:
					max_scheduled_jobs =  min(max_scheduled_jobs * MAGIC_NUMBER, max_workers * 1.2)
					if verbose:
						print(f'\nUnder rate limit for a while, increasing max scheduled jobs to {max_scheduled_jobs}')
				else:
					max_scheduled_jobs = max(1, max_scheduled_jobs / MAGIC_NUMBER)
					if verbose:
						print(f'\nOver rate limit for a while, decreasing max scheduled jobs to {max_scheduled_jobs}')
			last_refresh_time = time.monotonic()
	endTime = time.monotonic()
	apb.stop()
	_print_operation_summary(apb, endTime - start_time)
	return apb.item_counter, apb.size_counter

#%% ---- Copy Files ----
def _chown_like_cp(path, st, **how):
	"""Give path st's owner and group, as cp -a does: without privileges another user's ownership cannot be kept,
	so keep at least the group (one the user belongs to) and treat the rest as no error. Root failures raise.
	Returns (owner kept, group kept), as path has them now (it may have had the right owner already). how: os.chown's and
	os.stat's keywords (follow_symlinks=False: a link at the name is changed, never what it leads to)."""
	try:
		os.chown(path, st.st_uid, st.st_gid, **how)
		return True, True
	except PermissionError:
		if os.geteuid() == 0:
			raise
		try:
			os.chown(path, -1, st.st_gid, **how)
		except PermissionError:
			pass
	now = os.stat(path, **how)
	return now.st_uid == st.st_uid, now.st_gid == st.st_gid

def _named_root(path):
	"""The longest destination the operator named (NAMED_DESTS) that path, absolute, is or lies below; None when there is
	none (no run set them: a library call of copy_file, say)."""
	found = None
	for root in NAMED_DESTS:
		if (path == root or path.startswith(root.rstrip(os.sep) + os.sep)) and (found is None or len(root) > len(found)):
			found = root
	return found

def _landing_of(dest_path, base):
	"""(where a copy to dest_path writes, its lstat or None, named). A destination the operator named (-d) is followed when it
	is a link, as cp follows its destination argument (outside a run, every path counts as named); a landing below one is a
	name of the tree and never followed: a link, FIFO, socket or device node there is replaced, never opened (_cp_once).
	base: the door's result for the landing's directory (_dest_directory), which the landing is looked at relative to; None
	where the door found that directory missing: nothing is there."""
	target, named = _landing_path(dest_path)
	if base is None:
		return target, None, named
	try:
		return target, os.lstat(*_at(base, os.path.basename(target)), **_fd_of(base)), named
	except OSError:
		return target, None, named

def _landing_path(dest_path):
	"""(where a copy to dest_path writes, named), as _landing_of gives them, without its lstat of the landing: below a named
	destination nothing is looked at."""
	root = _named_root(os.path.abspath(dest_path))
	named = root is None or root == os.path.abspath(dest_path)
	target = dest_path
	if named and os.path.islink(dest_path):
		try:
			os.stat(dest_path)  # through the link, as cp's open goes: the kernel's own rules for following it apply
		except PermissionError:
			pass  # the kernel refuses it (fs.protected_symlinks), so it is not followed here: _cp_once keeps it
		except OSError:
			target = os.path.realpath(dest_path)  # dangling or a loop
		else:
			target = os.path.realpath(dest_path)
	return target, named

def _kind_of(st):
	"""What st (an lstat) is, in a message."""
	mode = st.st_mode
	for test, kind in ((stat.S_ISLNK, 'link'), (stat.S_ISFIFO, 'FIFO'), (stat.S_ISSOCK, 'socket'), (stat.S_ISBLK, 'block device'),
			(stat.S_ISCHR, 'character device'), (stat.S_ISDIR, 'directory')):
		if test(mode):
			return kind
	return 'file'

def _written_beside(st, named):
	"""True when what st (an lstat, or None) describes is a landing a copy replaces or writes in place (_cp_once): whatever is
	there and is not a directory, except a FIFO or device the operator named (a pipe feed, an image written to a disk), which
	cp writes through. False: cp writes the name."""
	if st is None or stat.S_ISDIR(st.st_mode):
		return False
	return not (named and (stat.S_ISFIFO(st.st_mode) or stat.S_ISCHR(st.st_mode) or stat.S_ISBLK(st.st_mode)))

# What a -co rewrite carries from the old file to the new one besides user.*: the POSIX ACLs and the SELinux label.
# security.capability is not carried: a write in place drops it (the kernel does), so the new content never gets one.
_CO_CARRIED = ('system.posix_acl_access', 'system.posix_acl_default', 'security.selinux')
_XFS_ACL_NAMES = {'trusted.SGI_ACL_FILE': _CO_CARRIED[0], 'trusted.SGI_ACL_DEFAULT': _CO_CARRIED[1]}  # root sees them on EL7

def _co_carry(target, st):
	"""-co: {name: value} of the xattrs a new file must get for it to stand where target (st: its lstat) is, or None when its
	metadata cannot be carried to a new file (_cp_beside): the runner is not root and does not own it, or
	cannot give a new file its group, or it holds an xattr hpcp cannot carry (trusted.*, security.ima, a filesystem's own; not
	XFS's own name for an ACL that its system.* name carries). listxattr failing with ENOTSUP means it has none."""
	if os.name != 'posix':
		return None
	euid = os.geteuid()
	if euid != 0 and not (st.st_uid == euid and (st.st_gid == os.getegid() or st.st_gid in os.getgroups())):
		return None
	if not hasattr(os, 'listxattr'):
		return {}
	try:
		names = os.listxattr(target, follow_symlinks=False)
	except OSError as e:
		return {} if e.errno in (errno.ENOTSUP, errno.EOPNOTSUPP) else None
	carried = {}
	for name in names:
		if name == 'security.capability' or _XFS_ACL_NAMES.get(name) in names:
			continue
		if not (name.startswith('user.') or name in _CO_CARRIED):
			return None
		try:
			carried[name] = os.getxattr(target, name, follow_symlinks=False)
		except OSError as e:
			if e.errno != errno.ENODATA:
				return None
	return carried

def _carry_over(st, temp, carried, held=None):
	"""Give temp, the new file of a -co rewrite, exactly what the old file (st, carried: _co_carry) had: its carried xattrs
	(each set, and any such name temp gained, as an ACL inherited from a default ACL, removed), owner and group, and mode, less
	set-user-ID, and set-group-ID where group-execute is set, when the runner is not root: the kernel clears those on a write
	by such a user, so the write in place lost them too. A label the filesystem gives itself (setting security.selinux fails
	with EOPNOTSUPP: vfat, a context= mount) is skipped. Anything else that fails raises. On Linux each step reaches what cp
	made through an O_PATH descriptor of it (never opened: a node made of a device is not acted on): a link or another file
	put at its name since (a user who can write the directory) is refused, so no chown or chmod goes through a link. held: the
	descriptor hpcp made temp by (_cp_beside), which every step goes through instead."""
	fd = os.open(temp, os.O_PATH | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0)) if _PATH_FDS and held is None else None
	try:
		if fd is not None:
			now = os.fstat(fd)
			if stat.S_ISLNK(now.st_mode) or now.st_nlink != 1:
				raise OSError(errno.EEXIST, 'it was replaced since cp made it', temp)
			temp = '/proc/self/fd/%d' % fd  # leads to the descriptor's file, and no further
		elif held is not None:
			temp = held
		how = {} if fd is not None or held is not None else {'follow_symlinks': False}
		if hasattr(os, 'listxattr'):
			try:
				gained = os.listxattr(temp, **how)
			except OSError as e:
				if e.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP):
					raise
				gained = []
			for name in gained:
				if name not in carried and (name.startswith('user.') or name in _CO_CARRIED[:2]):
					os.removexattr(temp, name, **how)
			for name, value in carried.items():
				try:
					os.setxattr(temp, name, value, **how)
				except OSError as e:
					if not (name == 'security.selinux' and e.errno in (errno.ENOTSUP, errno.EOPNOTSUPP)):
						raise
		os.chown(temp, st.st_uid, st.st_gid)
		mode = stat.S_IMODE(st.st_mode)
		if os.geteuid() != 0:
			mode &= ~stat.S_ISUID
			if mode & stat.S_IXGRP:
				mode &= ~stat.S_ISGID
		os.chmod(temp, mode)
	finally:
		if fd is not None:
			os.close(fd)

# Linux: a file reached through an O_PATH descriptor's /proc/self/fd/N name, never through a link at its own name.
_PATH_FDS = hasattr(os, 'O_PATH') and os.path.isdir('/proc/self/fd')

class _Refused(Exception):
	"""A landing this run will not write, for each file it would hold: a 'Destination clash' (213) naming what is there and
	the fix. text(file) is the message for one file. said: this process has printed it (a pinned destination that leads
	elsewhere, _dest_directory): recorded for each file, printed once (_say_refused)."""
	def __init__(self, before, after, said=False):
		super().__init__(before + '...' + after)
		self.before, self.after, self.said = before, after, said
	def text(self, path):
		return f'Destination clash: {self.before}{path}{self.after}'

def _say_refused(refusal, what):
	"""refusal's 213 line for what (eprint), or only its record where this process printed it already (refusal.said)."""
	if refusal.said:
		ERRORS.append(refusal.text(what))
	else:
		eprint(refusal.text(what))

class _KeptOld(Exception):
	"""A copy that left what is at its landing as it was (a rewrite beside whose new copy could not be made, filled, written to
	disk or renamed over it; -rbw's source failing on read; a named link that leads nowhere): 'leaving DEST as it is', 190."""

def _why(e):
	"""What failed, in a few words: an OSError's strerror, or what cp said (multiCMD's error ends with its last lines)."""
	if isinstance(e, OSError) and e.strerror:
		return e.strerror
	text = str(e)
	return text[text.rfind(': cp: ') + 2:] if ': cp: ' in text else text

# The directories the door holds (_dest_directory). Linux with /proc (_PATH_FDS): O_PATH, which search permission on the way is
# enough for, as for a path walk; a call with no dir_fd form reaches a held directory through its /proc/self/fd entry (_io).
# Elsewhere (macOS; Linux without /proc, where an O_PATH descriptor takes no fchmod, fchown, futimens, fsync or flistxattr):
# read, and O_NONBLOCK so a FIFO at the name is never waited on. The named destination is opened following a link (the
# operator named it); each step below it adds O_NOFOLLOW (_door_step).
_HOLD_PATH_FLAGS = getattr(os, 'O_PATH', 0) | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)
_HOLD_READ_FLAGS = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)

# Per process, in a run (RUN_ID): {named destination (absolute): the directories the door holds from it down to the last
# one asked for (_HeldDirs), or the refusal of one that no longer leads to its pin (NAMED_IDS)}. A forked worker inherits
# its parent's, whose descriptors stay valid in it. The cursor's own descriptors never leave it (the door hands out a dup):
# the next door call may close them, and a number reused then would lead a stale holder's write elsewhere.
_CURSORS = {}
_CURSORS_RUN = [None]
# How many directories the door may hold at once in a process: half its open-files limit, so the other half stays for what a
# copy opens beside them (its held file, cp's pipes, the pool's, the skip check's reads). Deeper levels go by path (macOS
# keeps its soft limit of 256: past 128 levels; Linux, where hpcp raises it to the hard limit at import, past 262,144, or
# 2,048 on EL7, deeper than the 4,096 characters a listed path holds). (pid, the budget) per process.
_HOLD_BUDGET = [None, None]

def _hold_budget():
	"""This process's _HOLD_BUDGET: half the soft open-files limit (none where it is unlimited or cannot be read)."""
	if _HOLD_BUDGET[0] != os.getpid():
		budget = None
		with contextlib.suppress(ImportError, ValueError, OSError):
			import resource
			soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
			if soft != resource.RLIM_INFINITY and soft > 0:
				budget = soft // 2
		_HOLD_BUDGET[:] = [os.getpid(), budget]
	return _HOLD_BUDGET[1]

def _close_cursors():
	"""Close every descriptor the door holds in this process (a new run, the run's end)."""
	while _CURSORS:
		held = _CURSORS.popitem()[1]
		if isinstance(held, _HeldDirs):
			held.close_all()

def _close_base(base):
	"""Close a door's result (_dest_directory) where it is a descriptor: the caller's, always in a finally."""
	if isinstance(base, int):
		os.close(base)

def _io(base, name, path):
	"""A path for a call that takes no dir_fd (xattrs, an ioctl's open, chflags, cp's argument) on name in base, the door's
	result for path's directory: on Linux with /proc name below base's /proc/self/fd entry, which leads to the directory the
	door holds (only name is looked up there); else path, the logical one (macOS; Linux without /proc)."""
	return '/proc/self/fd/%d/%s' % (base, name) if isinstance(base, int) and _PATH_FDS else path

def _cp_io(base, name, path):
	"""cp's argument for name in base, the door's result for path's directory: below base's /proc/self/fd entry (_io) where cp
	writes a held file (_held_route: it opens its own /proc/self/fd/N), else path. A cp the probe did not prove able to open
	its own /proc/self/fd (a cp other than GNU cp, an LSM rule, a /proc that does not show it) gets path, as in 9.87."""
	return _io(base, name, path) if _held_route() else path

def _io_fds(base, given, path):
	"""The descriptors a child given given (_cp_io) for path inherits (pass_fds): base's, which /proc/self/fd/BASE names in it."""
	return (base,) if given != path and isinstance(base, int) else ()

def _shown(e, directory, base):
	"""e, an OSError of a call made relative to base (a directory's descriptor) on a name in directory, naming its file(s) by
	the logical path: a bare name joined to directory, a /proc/self/fd/BASE/ prefix (_io) replaced by it. Returns e."""
	if isinstance(base, int):
		prefix = '/proc/self/fd/%d/' % base
		for attribute in ('filename', 'filename2'):
			name = getattr(e, attribute, None)
			if isinstance(name, str):
				if name.startswith(prefix):
					setattr(e, attribute, os.path.join(directory, name[len(prefix):]))
				elif name and os.sep not in name:
					setattr(e, attribute, os.path.join(directory, name))
	return e

def _cp_failed(e, given, target):
	"""Raise e, cp's failure on target, naming target where cp was given another name for it (given: the held file's
	/proc/self/fd/N or /dev/fd/N; the landing below its held directory, /proc/self/fd/D/NAME), in the error the runner
	recorded (ERRORS[-1]) and in the one raised; a stop (no Exception) as it is."""
	if given == target:
		raise e
	pairs = [(re.compile(re.escape(given) + '(?![0-9])'), target)]
	name = os.path.basename(target)
	if given.endswith(os.sep + name) and given.startswith('/proc/self/fd/'):
		pairs.append((re.compile(re.escape(given[:-len(name)])), os.path.dirname(target) + os.sep))
	def mapped(text):
		for pattern, replacement in pairs:
			text = pattern.sub(lambda match: replacement, text)
		return text
	if ERRORS and given in ERRORS[-1]:
		ERRORS[-1] = mapped(ERRORS[-1])  # cp's return-code error, as the runner recorded it
	if not isinstance(e, Exception):
		raise e
	raise RuntimeError(mapped(str(e))) from e

def _by_path_error(e):
	"""Whether the door finishes by path from where e (an OSError of an open) met it: the open-files limit anywhere, and a
	directory the runner may not read where the door holds read descriptors (macOS; Linux without /proc), as 9.87 did. On
	Linux with /proc an O_PATH open needs only search, so EACCES there is the error a path walk meets."""
	return e.errno in (errno.EMFILE, errno.ENFILE) or (e.errno == errno.EACCES and not _PATH_FDS)

def _named_ids(roots):
	"""{root: (st_dev, st_ino)} for each of roots (absolute) that is a directory now, as named (a link followed): the run's pin
	(NAMED_IDS). One stat each, once a run."""
	ids = {}
	for root in roots:
		with contextlib.suppress(OSError):
			st = os.stat(root)
			if stat.S_ISDIR(st.st_mode):
				ids[root] = (st.st_dev, st.st_ino)
	return ids

def _door_step(base, name, current, make, source, below):
	"""One component of the way to a landing: name in base (a held directory's descriptor, or its path: _at), current its
	path (messages), below how many components lie below it (the Note's source). Returns name's descriptor (base a
	descriptor; the caller holds it) or current (base a path) once it is a directory; None when it is missing and make is
	False. 9.87's rules, relative to base: a missing one is made (mkdir meeting EEXIST where lstat then finds nothing, on a
	second attempt: vfat iocharset=utf8, ENOENT); a link, FIFO, socket or device node is a name of the tree, unlinked and
	made a directory (a note for a link; _Refused under -ncd, which makes no directory in place of anything, and where its
	directory refuses the unlink); a regular file raises NotADirectoryError, the clash it always was. Three attempts at it,
	the race's bound since 9.85 (workers racing on one path: an unlink meeting ENOENT or EISDIR, a mkdir meeting EEXIST, are
	fine when the next look says directory), and a descriptor's last open after them. An open meeting a link (O_NOFOLLOW) or
	another kind of entry says ENOTDIR (Linux) or ELOOP (macOS): it is looked at by lstat."""
	held = isinstance(base, int)
	how = _fd_of(base)
	for attempt in range(4):
		missing = False
		if held:
			try:
				return os.open(name, (_HOLD_PATH_FLAGS if _PATH_FDS else _HOLD_READ_FLAGS) | getattr(os, 'O_NOFOLLOW', 0), dir_fd=base)
			except OSError as e:
				if e.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
					raise
				missing = e.errno == errno.ENOENT
		if attempt == 3:
			break
		st = None
		if not missing:
			try:
				st = os.lstat(*_at(base, name), **how)
			except FileNotFoundError:
				pass
			else:
				if stat.S_ISDIR(st.st_mode):
					if held:
						continue  # a directory since the open: open it again
					return current
		if st is None:
			if not make:
				return None
			try:
				os.mkdir(*_at(base, name), **how)
			except FileExistsError:
				# Another worker may have made it since the look: look again before calling it vfat's.
				try:
					again = os.lstat(*_at(base, name), **how)
				except FileNotFoundError:
					again = None
				if again is not None and stat.S_ISDIR(again.st_mode):
					if held:
						continue
					return current
				if again is None and attempt:
					raise OSError(errno.ENOENT, 'the directory holds this name in another case and does not find it in this '
						'one (vfat mounted iocharset=utf8: mount with utf8 instead)', current)
				continue
			if held:
				continue
			return current
		if stat.S_ISREG(st.st_mode):
			raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), current)
		kind = _kind_of(st)
		link = os.readlink(*_at(base, name), **how) if stat.S_ISLNK(st.st_mode) else None
		above = source
		for _ in range(below):
			above = above and os.path.dirname(above.rstrip(os.sep))
		named_form = f'name it with -d {_shell_word(current + os.sep)} to write through it' if link is not None else f'remove {current}'
		if NO_CREATE_DIR:
			raise _Refused(f'{current} is a {kind}' + (f' to {link}' if link is not None else '') +
				' and -ncd makes no directory in its place: not copying ', f' below it; {named_form}')
		try:
			os.unlink(*_at(base, name), **how)
		except (FileNotFoundError, IsADirectoryError):
			continue
		except OSError as e:
			parent = os.path.dirname(current)
			raise _Refused(f'cannot replace the {kind} {current}' + (f' (to {link})' if link is not None else '') +
				f' by a directory ({parent}: {e.strerror}): not copying ', f' below it; make {parent} writable, or remove {current}')
		try:
			os.mkdir(*_at(base, name), **how)
		except FileExistsError:
			continue
		if link is not None:
			print(f'Note: {current} was a link to {link} and is now a directory; nothing was written to {link}. To write through '
				f'a link, name it as the destination: hpcp {_shell_word(above.rstrip(os.sep) + os.sep) if above else "SRC/"} -d '
				f'{_shell_word(current + os.sep)}')
		if not held:
			return current
	raise OSError(errno.EAGAIN, 'it kept changing while it was made a directory', current)

def _door_by_path(start, parts, make, source):
	"""The door's walk by path (9.87's) from start, a directory, through parts: the last path, or None when one is missing and
	make is False. Where no descriptor can be had (_DIR_FD_WALK, _by_path_error)."""
	current = start
	for index, part in enumerate(parts):
		current = os.path.join(current, part)
		if _door_step(os.path.dirname(current), part, current, make, source, len(parts) - 1 - index) is None:
			return None
	return current

def _dest_directory(path, make=True, source=None):
	"""The one door for a directory below a destination: path as a real directory, and where its names are made from: a
	descriptor of it (a dup of the cursor's, the caller's to close in a finally: _close_base), or path itself where nothing
	is held (no named destination above it: a library call; no descriptor calls, _DIR_FD_WALK; _by_path_error); None when
	path or a directory on its way is missing and make is False (not under -ncd for a file's parent). The destination the
	operator named and what is above it are made as os.makedirs makes them, through links (it is named), and it is opened
	once per process in a run, following a link, and checked against the run's pin (NAMED_IDS): one that leads to another
	directory now is refused for everything below it (_Refused, said once a process). Below it each component is opened one level at a time without following a link (_door_step: its rules for
	what is there; source: the source directory that lands at path, to name the -d form in a note), and the levels shared
	with the last path asked for are kept open (_HeldDirs): a directory swapped for a link after it was opened is no longer
	on the way, as hpcp writes into the one it holds, wherever it is now."""
	path = os.path.abspath(path)
	root = _named_root(path)
	if root is None:
		if make:
			_makedirs(path, exist_ok=True)
			return path
		return path if os.path.isdir(path) else None
	if _CURSORS_RUN[0] != RUN_ID:
		_close_cursors()
		_CURSORS_RUN[0] = RUN_ID
	parts = path[len(root.rstrip(os.sep)) + 1:].split(os.sep) if path != root else []
	held = _CURSORS.get(root)
	if isinstance(held, _Refused):
		raise _Refused(held.before, held.after, said=True)
	if held is None:
		pinned = NAMED_IDS.get(root)
		if make and pinned is None:
			_makedirs(root, exist_ok=True)
		if not _DIR_FD_WALK:
			if not make and not os.path.isdir(root):
				return None
			if pinned is not None and os.path.isdir(root):
				st = os.stat(root)
				if (st.st_dev, st.st_ino) != pinned:
					refusal = _CURSORS[root] = _Refused(f'{root} now leads to another directory than the one this run started writing '
						'in: not copying ', '; run hpcp again to write where it leads now')
					raise refusal
			return _door_by_path(root, parts, make, source)
		try:
			fd = os.open(root, _HOLD_PATH_FLAGS if _PATH_FDS else _HOLD_READ_FLAGS)
		except OSError as e:
			if _by_path_error(e):
				return _door_by_path(root, parts, make, source)
			if not make and e.errno in (errno.ENOENT, errno.ENOTDIR):
				return None
			raise
		if pinned is not None:
			try:
				st = os.fstat(fd)
			except BaseException:
				os.close(fd)
				raise
			if (st.st_dev, st.st_ino) != pinned:
				os.close(fd)
				refusal = _CURSORS[root] = _Refused(f'{root} now leads to another directory than the one this run started writing '
					'in: not copying ', '; run hpcp again to write where it leads now')
				raise refusal
		held = _CURSORS[root] = _HeldDirs(fd)
	depth = 0
	while depth < len(parts) and depth + 1 < len(held) and held.names[depth + 1] == parts[depth]:
		depth += 1
	held.truncate(depth)
	budget = _hold_budget()
	for index in range(depth, len(parts)):
		current = os.path.join(root, *parts[:index + 1])
		if budget is not None and sum(len(other) for other in _CURSORS.values() if isinstance(other, _HeldDirs)) >= budget:
			return _door_by_path(os.path.dirname(current), parts[index:], make, source)
		try:
			fd = _door_step(held.top(), parts[index], current, make, source, len(parts) - 1 - index)
		except OSError as e:
			if _by_path_error(e):
				return _door_by_path(os.path.dirname(current), parts[index:], make, source)
			raise
		if fd is None:
			return None
		held.push(parts[index], fd)
	try:
		return os.dup(held.top())
	except OSError as e:
		if e.errno in (errno.EMFILE, errno.ENFILE):
			return path  # nothing to hold it with: by path, as 9.87
		raise

def _read_through(path):
	"""None when path, a regular file, reads to its end now (1 MiB reads, discarded), else the error: under -rbw a source that
	fails on read keeps the old copy a rewrite in place would truncate. Anything but a regular file is never read (cp -a makes
	a new node of it)."""
	try:
		if not stat.S_ISREG(os.stat(path).st_mode):
			return None
		with open(path, 'rb', buffering=0) as f:
			while f.read(1 << 20):
				pass
	except OSError as e:
		return e
	return None

def _claim_device(target, st, named, src_path):
	"""For a block device the operator named (-d), a descriptor of it opened O_EXCL, held until cp has written it: Linux fails
	that with EBUSY for one that is mounted, used as swap, or held by device-mapper, md or LVM, and no mount can start while
	it is held. None for anything else. A plain open by another program (a dd) is not seen."""
	if not (named and st is not None and stat.S_ISBLK(st.st_mode)) or os.name != 'posix':
		return None
	try:
		return os.open(target, os.O_WRONLY | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0))
	except OSError as e:
		if e.errno == errno.EBUSY:
			raise _Refused(f'{target} is in use (mounted, swap, or held by device-mapper, md or LVM): not writing ', ' over it')
		raise

# The full-disk warning (_space_warning), where statvfs's free room is what a copy can take: btrfs, ZFS and bcachefs compress
# or share blocks, NFS, SMB, Lustre and GPFS report pooled space, overlay hides what is below it.
_SPACE_TYPES = frozenset(('ext2', 'ext3', 'ext4', 'xfs', 'tmpfs'))
# One slot per destination filesystem that warns in a run; past them one warns once per file: it bounds extra lines only.
_SPACE_SLOTS = 64
# A run's slots, shared by its copy workers (lock, slots, the Array for a pool's initializer or None); None outside a run.
_SPACE_WARNED = None
_MOUNT_TYPES = {}  # (major, minor): its type, per process (_mount_type)

def _space_share():
	"""New slots for a run, or None where nothing can be shared. Forked workers (Python 3.6 to 3.13 on Linux) inherit an
	anonymous shared mapping and its lock, which leave no file (an Array on 3.6 and 3.7 leaves a pymp- directory when stopped,
	and 3.6's takes no 'Q'); others (forkserver, 3.14's default; spawn) get an Array through the initializer (_copy_pool)."""
	with contextlib.suppress(OSError, ImportError, ValueError, TypeError):
		if multiprocessing.get_start_method() == 'fork':
			import mmap
			return multiprocessing.Lock(), memoryview(mmap.mmap(-1, 8 * _SPACE_SLOTS)).cast('Q'), None
		if sys.version_info >= (3, 7):
			array = multiprocessing.Array('Q', _SPACE_SLOTS)
			return array.get_lock(), array.get_obj(), array
	return None

def _init_space_share(array):
	global _SPACE_WARNED
	_SPACE_WARNED = (array.get_lock(), array.get_obj(), array)

def _copy_pool(max_workers):
	"""The pool for jobs that may run copy_file, its workers sharing the run's slots (forked ones inherit them). Asked here, in
	the parent and outside the pool: how cp writes a held file (_held_route), which its workers are handed (_submit_task), so
	none makes a probe file or runs a cp for it (an early Ctrl-C met each worker's probe, and left its file)."""
	_held_route()
	return _pool(max_workers, space=True)

def _mount_type(dev):
	"""dev's filesystem type (_mountinfo_entries), None where no mount lists it (off Linux, a btrfs subvolume's st_dev): read
	once per process and again at a miss (a mount made since), and the miss is kept."""
	key = (os.major(dev), os.minor(dev))
	if key not in _MOUNT_TYPES:
		_MOUNT_TYPES.update((entry['devnum'], entry['fstype']) for entry in _mountinfo_entries())
		_MOUNT_TYPES.setdefault(key, None)
	return _MOUNT_TYPES[key]

def _claim_space(dev):
	"""True when this process prints dev's warning: the run's first claim of it, across its workers. Without slots, with all
	taken, or without the lock within 1 s (far longer than a scan of them takes: a holder died, and printing again beats
	hanging the copy), it prints each time."""
	share = _SPACE_WARNED
	if share is None or not share[0].acquire(timeout=1):
		return True
	try:
		slots, mark = share[1], dev + 1  # 0 is an empty slot
		for index in range(len(slots)):
			if slots[index] in (0, mark):
				claimed, slots[index] = slots[index] == 0, mark
				return claimed
		return True
	finally:
		share[0].release()

def _space_warning(target, src_path, free, need, old):
	"""Before cp writes target, whose directory has free bytes, less than need (src_path's allocated size): one line per
	filesystem in a run, then cp tries, as GNU cp does; nothing is refused, and the exit code is the copy's. old: the room an
	old copy that cp frees first holds (in place), else 0. Only for _SPACE_TYPES, and not on an XFS the source is on too (cp may
	clone it: GNU cp 9 does by default; ext and tmpfs never clone). A 'Space warning' (code 0) in ERRORS too, which a task hands back, repeated at the run's end."""
	directory = os.path.dirname(os.path.abspath(target))
	try:
		dev = os.stat(directory).st_dev
		kind = _mount_type(dev)
		if kind not in _SPACE_TYPES or (kind == 'xfs' and dev == os.lstat(src_path).st_dev) or not _claim_space(dev):
			return
	except OSError:
		return
	start = f'Space warning: {directory} has {format_bytes(free)}B free, less than the {format_bytes(need)}B that {src_path} needs'
	end = 'hpcp copies anyway, as cp does; other files short of room on this filesystem are not warned about again.'
	if old > 0 and need <= free + old:
		text = (f"{start}; it fits only by reusing its old copy's room, which cp frees first, so if another writer takes that "
			f'room the copy fails and the old copy is lost (the source is kept). {end}')
	else:
		lost = "a rewrite's old copy stays (-wb)" if WRITE_BESIDE else "a rewrite's old copy is lost"
		text = f'{start}, so it may not fit; if the disk runs out the copy fails, and {lost} (the source is kept). {end}'
	# One write, flushed before cp: its start leaves the bar's line (_line_start), and no other output lands in it (3.6 to 3.8
	# buffer stderr).
	with contextlib.suppress(Exception):
		sys.stdout.flush()
		sys.stderr.write(_line_start() + text + '\n')
		sys.stderr.flush()
	ERRORS.append(text)

def _repeat_space_warnings():
	"""At a run's end: its space warnings again, each once, then out of ERRORS, so a run with nothing else ends 'Done.'."""
	warnings = list(dict.fromkeys(error for error in ERRORS if error.startswith('Space warning:')))
	if warnings:
		ERRORS[:] = [error for error in ERRORS if not error.startswith('Space warning:')]
		sys.stdout.flush()
		print('\n'.join(['', 'Space warnings in this run (repeated from above):'] + warnings), file=sys.stderr)

def _cp_to(command, src_path, target, space=None, base=None):
	"""cp src_path to target by name (a first copy; a FIFO or device the operator named, written through); its size. base:
	the door's result for its directory (None: its path), cp given the landing below it (_io). A part it leaves goes
	(_remove_what_failed)."""
	base = os.path.dirname(target) if base is None else base
	name = os.path.basename(target)
	if space is not None:
		_space_warning(target, src_path, space[0], space[1], 0)
	given = _cp_io(base, name, target)
	before = _copy_state(base, name)
	try:
		run_command_in_multicmd_with_path_check(command + [src_path, given], quiet=True, strict=True, pass_fds=_io_fds(base, given, target))
	except BaseException as e:
		_remove_what_failed(base, name, before)
		_cp_failed(e, given, target)
	return get_file_size(*_at(base, name), **_fd_of(base))

# A landing opened to be written in place: never through a link put there (ELOOP), never waiting on a FIFO (ENXIO) or a
# lease (EAGAIN), not inherited by cp; no O_TRUNC (cp truncates it).
_LANDING_FLAGS = os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0)
_HELD = [None, None]  # (pid, _held_route's answer): asked once per process, so a forked worker asks for itself
_HELD_GIVEN = None  # (_held_route's answer,), a worker's, handed by its parent's task (_submit_task): the worker asks no cp
# The name under which cp opens a descriptor it inherited (pass_fds): its own process's, so no other process's /proc is asked.
_HELD_NAME = '/dev/fd/%d' if sys.platform == 'darwin' else '/proc/self/fd/%d'

# Where the held-file probe makes its two files (_held_route): tempfile's candidates after the environment's, in its order,
# then the working directory.
_TEMP_DIRS = ('/tmp', '/var/tmp', '/usr/tmp')

def _temp_candidates():
	"""The directories the held-file probe tries, in tempfile.gettempdir's order: tempfile.tempdir alone when a program set it
	(gettempdir returns it untried); else TMPDIR, TEMP and TMP, then _TEMP_DIRS, then the working directory."""
	if tempfile.tempdir is not None:
		return [tempfile.tempdir]
	found = [os.environ[name] for name in ('TMPDIR', 'TEMP', 'TMP') if os.environ.get(name)]
	found.extend(_TEMP_DIRS)
	with contextlib.suppress(OSError):
		found.append(os.getcwd())
	return [os.path.abspath(directory) for directory in found]

def _held_route():
	"""How cp writes a file this process holds open, through the descriptor it inherits (_HELD_NAME): 'c' when its write leaves
	the file at the source's size (Linux reopens /proc/self/fd/N and truncates; macOS's copyfile truncates after its copy),
	'cb' when it writes through it but leaves an old tail (a dup opened without truncating: hpcp empties the held file before
	each cp), None when it does not: cp then gets the name, as 9.84 gave it. Asked once per process by doing it: cp, with the
	run's flags, copies a 1-byte source holding c onto a held file holding ab, which is read back (a pool's workers are
	handed their parent's answer, _HELD_GIVEN, and ask nothing). Only on Linux with GNU cp (BusyBox's makes every destination
	anew) and on macOS; elsewhere None. A cp that times out or cannot start keeps no answer (asked again at the next copy, as
	cp_supports_sparse); a cp that fails or writes elsewhere, or no directory to make the files in (_temp_candidates), is
	None. The source is made where the first candidate takes it, and the held file beside it with no name (O_TMPFILE), or
	named and unlinked at once where it cannot be (a kernel before 3.11, which ignores the flag; a filesystem without it;
	macOS), as tempfile.TemporaryFile does. Each is held by C as it is made (_create), and each name made goes whatever
	happens: tempfile's own calls left their files to a stop landing between its create and its unlink (gettempdir's test
	file, mkstemp's; 9.87 deferred stops there, for hpcp's own handler only)."""
	pid = os.getpid()
	if _HELD[0] == pid:
		return _HELD[1]
	if _HELD_GIVEN is not None:
		return _HELD_GIVEN[0]
	_HELD[:] = [pid, None]
	if sys.platform.startswith('linux'):
		gnu = cp_supports_sparse()
		if gnu is None:
			_HELD[0] = None  # cp gave no answer: asked again at the next copy
			return None
		if not gnu or not os.path.isdir('/proc/self/fd'):
			return None
	elif sys.platform != 'darwin':
		return None
	# Each name is set before its create, and removed as soon as it has served: by the finally only where a stop or an error
	# came first (a stop that lands in a finally the run reached normally cuts it), and then only once the create has stored its
	# descriptor (_create).
	source = held_name = None
	sources, helds = [], []
	try:
		for directory in _temp_candidates():
			source = os.path.join(directory, 'hpcp-held-probe-%d-%s' % (pid, uuid.uuid4().hex[:16]))
			try:
				_create(directory, os.path.basename(source), _NEW_FILE_FLAGS, 0o600, sources)
				# Written here: a full filesystem makes the name but takes no byte, and the next candidate is tried, as
				# tempfile.gettempdir tries its candidates by writing (9.87 went on to /var/tmp where /tmp was full).
				os.write(sources[-1], b'c')
				break
			except OSError:
				if sources:
					with contextlib.suppress(OSError):
						os.unlink(source)
					os.close(sources.pop())
				continue
		else:
			return None  # no directory to make the probe's files in: no route
		try:
			helds.extend(map(functools.partial(os.open, flags=os.O_TMPFILE | os.O_RDWR | getattr(os, 'O_CLOEXEC', 0), mode=0o600),
				(directory,)))
		except (AttributeError, OSError):
			held_name = source + '-held'
			_create(directory, os.path.basename(held_name), _NEW_FILE_FLAGS & ~os.O_WRONLY | os.O_RDWR, 0o600, helds)
			os.unlink(held_name)
			held_name = None
		fd = helds[0]
		os.write(fd, b'ab')
		os.lseek(fd, 0, os.SEEK_SET)  # as a landing's fresh descriptor: macOS's /dev/fd/N shares its offset
		# an empty copy takes milliseconds; a cp that hangs this long gets no route (_CP_ASK_TIMEOUT)
		task = multiCMD.run_commands([[_bin('cp'), '-f' if CONTENT_ONLY else '-af', source, _HELD_NAME % fd]],
			timeout=_CP_ASK_TIMEOUT, quiet=True, return_object=True, pass_fds=(fd,))[0]
		os.unlink(source)
		source = None
		if task.returncode in _NO_ANSWER:
			_HELD[0] = None
		elif task.returncode == 0:
			_HELD[1] = {b'c': 'c', b'cb': 'cb'}.get(os.pread(fd, 8, 0))
	except Exception:
		pass  # no file to probe with: no route
	finally:
		# The names first: a second stop landing in here cuts what follows it (a descriptor left open in a process that is ending).
		if sources and source is not None:
			with contextlib.suppress(OSError):
				os.unlink(source)
		if helds and held_name is not None:
			with contextlib.suppress(OSError):
				os.unlink(held_name)
		for opened in (sources, helds):
			while opened:
				os.close(opened.pop())
	return _HELD[1]

def _cp_held(command, src_path, target, fd, fst, space, through_link, new=False, base=None):
	"""cp src_path into target, the regular file with one name this process holds open (fd, fst its fstat); its size. cp
	writes the held file through the descriptor it inherits (_held_route), so a link put at the name since the open is never
	written through; with no route, the name, as in 9.84. The inode stays. A cp that fails or is stopped after it changed the
	file takes the torn name if it is still this file, as 9.84 did, except where a link the operator named led there
	(through_link: kept for the retry, 9.84). -rbw reads the source first; its failure keeps the file (_KeptOld). new: the
	file is one hpcp just made (_cp_made): no -rbw read, no old copy, and a failure takes the name while it leads to
	it. Errors name target, not the descriptor's name. base: the door's result for target's directory (None: its path), where
	the name is looked at and taken, and cp with no route gets the landing below it (_io)."""
	base = os.path.dirname(target) if base is None else base
	name = os.path.basename(target)
	if READ_FIRST and not new:
		failed = _read_through(src_path)
		if failed is not None:
			raise _KeptOld(f'{src_path} failed on read ({_why(failed)})')
	if space is not None:
		_space_warning(target, src_path, space[0], space[1], 0 if new else fst.st_blocks * 512)
	route = _held_route()
	held = _HELD_NAME % fd if route else None
	given = held or _cp_io(base, name, target)
	if route == 'cb' and not new:
		os.ftruncate(fd, 0)  # cp's open will not truncate it, and a shorter source would leave the old tail
	try:
		run_command_in_multicmd_with_path_check(command + [src_path, given], quiet=True, strict=True,
			pass_fds=(fd,) if held else _io_fds(base, given, target))
	except BaseException as e:
		with contextlib.suppress(OSError):
			now, named = os.fstat(fd), os.lstat(*_at(base, name), **_fd_of(base))
			if (not through_link and (new or (now.st_size, now.st_mtime_ns, now.st_ctime_ns) != (fst.st_size, fst.st_mtime_ns, fst.st_ctime_ns))
					and (named.st_dev, named.st_ino) == (fst.st_dev, fst.st_ino)):
				os.unlink(*_at(base, name), **_fd_of(base))
		_cp_failed(e, given, target)
	if not CONTENT_ONLY and not new:
		_remove_stale_xattrs(src_path, fd)
		if held is not None:
			_label_like(src_path, fd)
	return os.fstat(fd).st_blocks * 512

# A file hpcp makes for cp to write (_cp_made: a first copy, the beside route's new file): never through a link or onto a
# name there.
_NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)

def _new_file_mode(src_st):
	"""The mode a file hpcp makes for cp starts with (_cp_made): 0600 for cp -a, which gives it the source's mode through the
	descriptor when it ends (no reader sees it wider than the source); under -co (cp -f, which keeps an existing file's mode)
	the source's permission bits, which the umask and a default ACL then make what cp -f makes."""
	return stat.S_IMODE(src_st.st_mode) & 0o777 if CONTENT_ONLY and src_st is not None else 0o600

def _create(base, name, flags, mode, opened):
	"""Make name in base (a directory's descriptor, or its path: _at) with flags (O_CREAT | O_EXCL ...) and mode, its
	descriptor appended to opened (a list) by C: list.extend over map stores what os.open returns before Python runs another
	instruction, so a stop's handler (hpcp's _stop, or Python's own in a program that imports hpcp) never lands between the
	create and the store, where `fd = os.open(...)` loses the descriptor and leaves the file. The caller's cleanup closes
	and removes what opened holds."""
	opened.extend(map(functools.partial(os.open, flags=flags, mode=mode, **_fd_of(base)), _at(base, name)))
	return opened[-1]

def _drop_made(base, name, made):
	"""Remove name in base (_at) while it still leads to made, the file hpcp made there (its fstat): never what was put there
	since."""
	with contextlib.suppress(OSError):
		named = os.lstat(*_at(base, name), **_fd_of(base))
		if (named.st_dev, named.st_ino) == (made.st_dev, made.st_ino):
			os.unlink(*_at(base, name), **_fd_of(base))

def _has_acl(fd):
	"""Whether the file fd is open on carries POSIX ACL entries beyond its mode (Linux; False where it cannot be asked)."""
	try:
		return 'system.posix_acl_access' in os.listxattr(fd)
	except (OSError, AttributeError, TypeError):
		return False

def _cp_made(command, src_path, target, src_st, space, base=None):
	"""cp src_path, a regular file, into target, a new file hpcp makes for it (O_EXCL: never through a link put at the name;
	FileExistsError when a name is there) and cp writes through its own descriptor (_cp_held): (that descriptor, still open
	for the caller to close, the made file's fstat, the size); None where cp has to make it by name (below). A failure
	removes the made file while its name still leads to it.
	A plain run gives the new file the source's SELinux label (_label_like) before cp writes it, and its owner and group too
	where a default ACL gave it entries beyond its mode: cp -a gives a file it did not make another owner only after
	narrowing its mode and ACL to the owner's (the entries go), where a file it makes keeps them. Where they cannot be given
	(not root) and the file has such entries: None, the file removed, for cp by name as in 9.86; and so for a root run of
	a file whose owner or group is not the made file's (another user's or another group's), where giving it first charges
	root's write to that owner's or group's quota (below). A root run of a file whose owner and group are the made file's
	(root:root, or root and the group of a setgid directory) gives nothing and keeps the held route. Without such entries
	cp -a gives the owner once it has written, as by name: given first, the data would be charged to the owner while it is
	written (XFS enforces an owner's or a group's quota on a root write, not on root's chown: a user, or a group, at the hard
	limit got Disk quota exceeded where 9.86 completed). A mode its owner cannot write (-co's 0444 source, a umask taking
	owner write) has owner write while cp writes, which cp's reopen needs; -co puts the mode back. base: the door's result for
	target's directory (None: its path), which the file is made in."""
	base = os.path.dirname(target) if base is None else base
	name = os.path.basename(target)
	src_st = src_st or os.lstat(src_path)
	# the descriptor, held from its create on (_create): a stop's cleanup below always finds the file it made
	opened, made = [], None
	try:
		fd = _create(base, name, _NEW_FILE_FLAGS, _new_file_mode(src_st), opened)
		made = os.fstat(fd)
		if not CONTENT_ONLY:
			_label_like(src_path, fd)
			if _has_acl(fd):
				# Only where a default ACL gave the file entries, which cp -a would narrow away once it gives a file it did not
				# make another owner. Elsewhere cp -a gives the owner once it has written, as by name. A root run does not give
				# another user or another group the file first: the data would be charged to that owner's or group's quota while
				# root writes it (XFS enforces both on a root write, not on root's chown), so there cp makes the file by name,
				# as in 9.86. Only a change of owner or group is charged differently: a file with the made file's owner and group keeps the route.
				by_name = os.geteuid() == 0 and (made.st_uid, made.st_gid) != (src_st.st_uid, src_st.st_gid)
				if not by_name:
					try:
						os.fchown(fd, src_st.st_uid, src_st.st_gid)
					except PermissionError:
						by_name = True
				if by_name:
					_drop_made(base, name, made)
					made = None
					# out of opened before the close: a stop raised as it returns would have the handler below close it again,
					# whose EBADF took the stop's place (a false Copy failed, the stop left to the task's end)
					os.close(opened.pop())
					return None
		narrow = not made.st_mode & stat.S_IWUSR
		if narrow:
			os.fchmod(fd, stat.S_IMODE(made.st_mode) | stat.S_IWUSR)
		size = _cp_held(command, src_path, target, fd, made, space, False, new=True, base=base)
		if narrow and CONTENT_ONLY:
			os.fchmod(fd, stat.S_IMODE(made.st_mode))
		return fd, made, size
	except BaseException:
		if opened:
			if made is None:
				# stopped before the fstat stored it
				with contextlib.suppress(OSError):
					made = os.fstat(opened[0])
			if made is not None:
				_drop_made(base, name, made)
			os.close(opened.pop())
		raise

def _cp_new(command, src_path, target, src_st, space, base=None):
	"""A first copy (nothing at target) of a regular source, where cp writes a held file (_held_route): made by hpcp in base
	(the door's result for its directory) and written through cp's own descriptor (_cp_made), else by name (_cp_to).
	FileExistsError when a name appeared there since the look, for the caller to look again."""
	made = _cp_made(command, src_path, target, src_st, space, base)
	if made is None:
		return _cp_to(command, src_path, target, space, base)
	os.close(made[0])
	return made[2]

def _label_like(src_path, fd):
	"""The source's SELinux label on fd, a held file cp -a writes through its descriptor's name (after a rewrite, before a first
	copy). cp -a gives an existing destination the source's label by name (lsetfilecon, coreutils 8.22 on), which through
	/proc/self/fd/N reaches the /proc link, not the file, and it says nothing when that fails; by name (9.84) and on a new file
	(9.85) the label came along.
	Set through the descriptor, only where it differs, and as cp -a does it: silently where it cannot be (no label, not
	permitted, a filesystem that labels itself)."""
	with contextlib.suppress(OSError, AttributeError):
		label = os.getxattr(src_path, 'security.selinux', follow_symlinks=False)
		if os.getxattr(fd, 'security.selinux') != label:
			os.setxattr(fd, 'security.selinux', label)

def _in_place_hint(target, st, src_path, base=None):
	"""Under -wb, the end of a failed rewrite's reason where the default would have written it in place (one name, and it
	opens for writing: asked by opening it in base, the door's result for its directory, nothing written; and a regular
	source, as cp -a makes the rest anew)."""
	if WRITE_BESIDE and stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and _regular_source(src_path):
		base = os.path.dirname(target) if base is None else base
		with contextlib.suppress(OSError):
			os.close(os.open(*_at(base, os.path.basename(target)), _LANDING_FLAGS, **_fd_of(base)))
			return '; without -wb it would be written in place'
	return ''

def _cp_beside(command, src_path, target, st, why, space, src_st=None, base=None):
	"""cp src_path to a new name beside target, write it to disk and rename it over target once whole; its size. st: what is
	at target (lstat, or the held file's fstat); why: why it goes beside (None for a non-file). Never in place: a failure
	keeps what is there, a file as _KeptOld (190) naming why and what failed, a FIFO, socket, device or link as _Refused
	(213). Under -co the new file gets the old one's owner, group, mode, carried xattrs and FAT bits (where they cannot be
	carried: one name keeps its old copy, 190; other names get a file the runner owns, and keep the old content). base: the
	door's result for target's directory (None: its path): the new name is made, looked at and renamed in it, and a call
	with no dir_fd form goes below it (_io)."""
	base = os.path.dirname(target) if base is None else base
	name = os.path.basename(target)
	regular = stat.S_ISREG(st.st_mode)
	carried = _co_carry(_io(base, name, target), st) if CONTENT_ONLY and regular else None
	if CONTENT_ONLY and regular and carried is None and st.st_nlink < 2:
		raise _KeptOld(f'{why}; its owner, group or an xattr cannot be given to a new file{_in_place_hint(target, st, src_path, base)}')
	if space is not None:
		_space_warning(target, src_path, space[0], space[1], 0)
	temp, renamed, what = _temp_sibling(target), False, 'the new copy beside it failed: %s'
	tname = os.path.basename(temp)
	# A regular source's new file is made by hpcp and written by cp through its descriptor (_held_route), and what follows goes
	# through that descriptor too: a second name of another file put at the temporary name is never what hpcp holds. A node
	# source is made by cp -a (mknod), by name.
	fd = None
	try:
		try:
			if _regular_source(src_path, src_st) and _held_route():
				made = _cp_made(command, src_path, temp, src_st, None, base)
				if made is not None:
					fd, made = made[:2]
			if fd is None:
				given = _cp_io(base, tname, temp)
				try:
					run_command_in_multicmd_with_path_check(command + [src_path, given], quiet=True, strict=True, pass_fds=_io_fds(base, given, temp))
				except BaseException as e:
					_cp_failed(e, given, temp)
			attributes = None
			if carried is not None:
				what = 'its owner, group or an xattr cannot be given to the new copy (%s)'
				_carry_over(st, _io(base, tname, temp), carried, fd)
				attributes = _fat_attributes(_io(base, name, target))
			elif not CONTENT_ONLY:
				_remove_stale_xattrs(src_path, _io(base, tname, temp) if fd is None else fd)
			# Renamed unsynced, a power cut could leave nothing where the old copy was.
			what = 'the new copy cannot be written to disk (%s)'
			if fd is None:
				_fsync_file(temp, base, tname)
			else:
				try:
					os.fsync(fd)
				except OSError as e:
					if e.errno not in _NO_FSYNC:
						raise
			what = 'it could not be replaced by the new copy (%s)'
			if fd is not None:
				# the rename is by name: something put there since (a user who can write the directory) is not renamed over
				# the old copy
				now = os.lstat(*_at(base, tname), **_fd_of(base))
				if (now.st_dev, now.st_ino) != (made.st_dev, made.st_ino):
					raise OSError(errno.EEXIST, 'something else was put at its name', temp)
			os.rename(*_at(base, tname), *_at(base, name), **({'src_dir_fd': base, 'dst_dir_fd': base} if isinstance(base, int) else {}))
			renamed = True  # nothing left to discard: an unlink of the gone name costs a round trip on NFS
		except Exception as e:
			if not regular:
				parent = os.path.dirname(target)
				raise _Refused(f'cannot replace the {_kind_of(st)} {target} ({parent}: {_why(_shown(e, parent, base))}): not writing ',
					f' into it; make {parent} writable, or remove {target}')
			raise _KeptOld(f'{why}; {what % _why(_shown(e, os.path.dirname(target), base))}{_in_place_hint(target, st, src_path, base)}')
		if attributes is not None and attributes != _fat_attributes(_io(base, name, target)):
			_set_fat_attributes(_io(base, name, target), attributes, target)
		return get_file_size(*_at(base, name), **_fd_of(base))
	finally:
		if fd is not None:
			os.close(fd)
		if not renamed:
			_discard_temp(temp, base)

def _cp_once(src_path, dest_path, cp_flags, sparse, space=None, look=None, src_st=None, base=None):
	"""One attempt at copying src_path to dest_path (copy_file); the size copied. Raises _Refused for what this run will not
	write, _KeptOld for a rewrite that kept what is there. space: (free, need) when the room is short (_space_warning). By
	what is at the landing (_landing_of): nothing, or a FIFO or device the operator named: cp writes the name (_cp_to). A
	regular file with one name that opens for writing, of a regular source: in place, through the held file (_cp_held).
	Anything else (other names, not writable, a running program, a source cp makes anew; a FIFO, socket, device or link below
	a destination, never opened), anything the open finds changed since the look, and every rewrite under -wb: beside it,
	renamed over it once whole (_cp_beside); never to cp by name. look: _landing_of(dest_path, base) when the caller just
	took it (copy_file's first attempt); src_st: the source's lstat. base: the door's result for the landing's directory
	(copy_file's, which closes it), every name below it made, looked at and written relative to it; None: the door is asked
	here (made, not under -ncd), its result closed here. An OSError raised out names its file by the logical path."""
	opened = None
	try:
		if base is None:
			directory = os.path.dirname(_landing_path(dest_path)[0])
			opened = _dest_directory(directory, make=not NO_CREATE_DIR, source=os.path.dirname(os.path.abspath(src_path)))
			base = directory if opened is None else opened
		target, st, named = look or _landing_of(dest_path, base)
		_judge_landing(dest_path, target, st, named)
		command = ["cp", cp_flags] + (["--sparse=always"] if sparse and cp_supports_sparse() else [])
		# A name that appears at a first copy's landing between the look and hpcp's create is looked at again, judged as the
		# first look was, and the route that fits it taken; 3 times, the door's bound for the same race.
		for _ in range(3):
			try:
				return _cp_landing(command, src_path, dest_path, target, st, named, space, src_st, base)
			except FileExistsError:
				target, st, named = _landing_of(dest_path, base)
				_judge_landing(dest_path, target, st, named)
		raise OSError(errno.EAGAIN, 'it kept changing while it was copied', target)
	except OSError as e:
		raise _shown(e, os.path.dirname(os.path.abspath(dest_path)), base)
	finally:
		_close_base(opened)

def _judge_landing(dest_path, target, st, named):
	"""What no look at a landing lets through (_cp_once), at every look: a dangling link the operator named (cp writes through
	none, nor does hpcp: a planted one would have it make directories and a file anywhere), one the kernel will not follow
	for this run, and a directory (cp would copy into it, through whatever is there)."""
	if named and st is None and os.path.islink(dest_path):
		raise _KeptOld(f'{dest_path} is a link to {os.readlink(dest_path)}, which does not exist (cp writes through no dangling link)')
	if named and st is not None and stat.S_ISLNK(st.st_mode):
		# _landing_of left it: the kernel will not follow it for this run (another user's link in a sticky directory:
		# fs.protected_symlinks), as cp's open meets it; a loop is replaced, as in 9.85.
		try:
			os.stat(dest_path)
		except PermissionError as e:
			raise _KeptOld(f'{dest_path} is a link to {os.readlink(dest_path)}, which cannot be followed ({_why(e)})')
		except OSError:
			pass
	if st is not None and stat.S_ISDIR(st.st_mode):
		raise _Refused(f'{target} is a directory, not copying ', ' over it')

def _cp_landing(command, src_path, dest_path, target, st, named, space, src_st, base):
	"""_cp_once's copy onto target, dest_path's landing, by what its look found there (st, named), relative to base (the door's
	result for its directory). FileExistsError: a first copy's name taken since."""
	held = _claim_device(target, st, named, src_path)
	try:
		if not _written_beside(st, named):
			if st is None and _regular_source(src_path, src_st) and _held_route():
				return _cp_new(command, src_path, target, src_st, space, base)
			return _cp_to(command, src_path, target, space if st is None else None, base)
		if not WRITE_BESIDE and stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and _regular_source(src_path, src_st):
			try:
				fd = os.open(*_at(base, os.path.basename(target)), _LANDING_FLAGS, **_fd_of(base))
			except FileNotFoundError:
				# gone since the lstat: a first copy
				if _held_route():
					return _cp_new(command, src_path, target, src_st, space, base)
				return _cp_to(command, src_path, target, space, base)
			except OSError as e:
				return _cp_beside(command, src_path, target, st, f'it cannot be opened for writing ({_why(e)})', space, src_st, base)
			try:
				st = os.fstat(fd)  # what the descriptor holds decides from here on
				if stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
					return _cp_held(command, src_path, target, fd, st, space, target != dest_path, base=base)
			finally:
				os.close(fd)
			# Changed since the lstat (another name, a FIFO someone holds open): beside, never cp by name.
		# Why it goes beside: the start of the reason a failure gives.
		why = ('-wb writes beside it' if WRITE_BESIDE else None if not stat.S_ISREG(st.st_mode) else
			'it has other names, which a write in place would change' if st.st_nlink > 1 else
			f'{src_path} is no regular file, which cp makes anew')
		return _cp_beside(command, src_path, target, st, why, space, src_st, base)
	finally:
		if held is not None:
			os.close(held)

def _regular_source(src_path, src_st=None):
	"""False for a source that cp -a makes anew (a FIFO, socket or device node: it unlinks what is at the landing, and cannot
	unlink a held file's /proc name); a source that is gone is cp's to answer. src_st: its lstat, when the caller has it."""
	try:
		return stat.S_ISREG((src_st or os.lstat(src_path)).st_mode)
	except OSError:
		return True

def _cp_flags(src_path, src_st=None):
	"""How cp copies src_path. Under -co cp -f reads what it is given and writes through what is there, so only a regular file
	goes that way: cp -a makes a new node for anything else and never opens it, so a device is not read to its end and a FIFO
	does not wait for a writer. A source that is gone is cp's to answer."""
	return "-f" if CONTENT_ONLY and _regular_source(src_path, src_st) else "-af"

def _can_open(path, st=None):
	"""False when cp could not read the regular file path. Asked by opening, with the ids cp runs with: access() answers for
	the real uid, and a service with a read capability or a setuid wrapper can open more. Only a regular file: cp -a makes a
	new node for the rest, and opening one is an act of its own (a FIFO's writer wakes, a watchdog arms). No search
	permission on the way answers as a file does."""
	try:
		# st: the caller's lstat (its source is no link), else a stat: a link to a file is read as cp -a's caller reads it.
		if stat.S_ISREG((st or os.stat(path)).st_mode):
			os.close(os.open(path, os.O_RDONLY))
	except PermissionError:
		return False
	except OSError:
		pass
	return True

def _can_make_node(src_path, dest_paths, src_st=None, bases=None):
	"""False when src_path is a device node that this user cannot make, and a destination holds something. cp -a unlinks
	what is there before its mknod, so a user who may not make nodes (not root) would lose the old file and get nothing.
	Asked by doing it: a node of src_path's kind and number is made under a name of its own beside a destination that
	exists, and removed again; PermissionError is the answer. Any other error (EROFS, say) is no answer: the next existing
	destination is asked, and when none answers it is cp's to answer. Nothing is made for a
	source that is no device, or when no destination exists (cp then has nothing to lose). bases: {dest: the door's result
	for its directory} (copy_file's), which each is looked at and asked in."""
	try:
		st = src_st or os.lstat(src_path)
	except OSError:
		return True
	if not (stat.S_ISBLK(st.st_mode) or stat.S_ISCHR(st.st_mode)):
		return True
	bases = bases or {}
	def held(dest):
		return bases.get(dest) if bases.get(dest) is not None else os.path.dirname(dest)
	existing = []
	for dest in dest_paths:
		with contextlib.suppress(OSError):
			os.lstat(*_at(held(dest), os.path.basename(dest)), **_fd_of(held(dest)))
			existing.append(dest)
	if not existing:
		return True
	for dest in existing:
		name = '.hpcp-node-probe-%d-%s' % (os.getpid(), uuid.uuid4().hex)
		base = None
		try:
			# Relative to the directory's descriptor where it can be (_ask_names): the name costs its own length. The door's
			# own is never closed here: a dup of it.
			if _PROBE_BY_FD and os.mknod in os.supports_dir_fd:
				base = os.dup(held(dest)) if isinstance(held(dest), int) else os.open(os.path.dirname(dest) or os.curdir, _PARENT_FLAGS)
			else:
				base = held(dest) if not isinstance(held(dest), int) else os.path.dirname(dest)
			os.mknod(*_at(base, name), 0o600 | stat.S_IFMT(st.st_mode), st.st_rdev, **_fd_of(base))
		except PermissionError:
			return False
		except OSError:
			# This directory cannot answer (a read-only mount, say); the next destination may.
			continue
		else:
			try:
				os.unlink(*_at(base, name), **_fd_of(base))
			except OSError:
				pass
			return True
		finally:
			if isinstance(base, int):
				os.close(base)
	return True

def _copy_state(base, name):
	"""What a copy to name in base (_at) would change at that name itself: a link stays whatever cp did through it, so a retry
	writes through it again and its target ends whole or the file fails. None when nothing is there."""
	try:
		st = os.lstat(*_at(base, name), **_fd_of(base))
	except OSError:
		return None
	return st.st_ino, st.st_size, st.st_mtime_ns

def _remove_what_failed(base, name, before):
	"""After a failed copy to name in base (_at), whose _copy_state was before: its part goes. What it never touched is the
	destination's old copy, maybe the only good one, so that stays."""
	if _copy_state(base, name) not in (None, before):
		try:
			os.remove(*_at(base, name), **_fd_of(base))
		except OSError:
			pass  # its directory will not let it go (0555, sticky, a bind mount): the copy's own error is the one to report

# A landing opened to be read (the skip check): never through a link put there, never waiting on a FIFO.
_LANDING_READ_FLAGS = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0)

def _known_identical(src_path, dest_path, src_size, full_hash, src_st=None, dest_st=None, base=None):
	"""is_file_identical, where a side that may not be read is not known to be the same: the copy then goes on, and a source
	stops at _can_open, its old copy kept. Only that: any other read error stays a crash, because cp would truncate first.

	Under -co, which carries no times, a regular landing of the source's size is the same when it reads the same to the end
	(_same_contents), whatever its time: the time test never matched there, so every rerun rewrote every file, and the hash
	reads only the last HASH_SIZE bytes, missing a change before them. Under -hs 0 ("file attributes only") it is rewritten.

	base: the door's result for the landing's directory (copy_file's): the landing is read by its name there, never through
	a link put at it since the look (O_NOFOLLOW; one there, or the name gone, is not known to be the same)."""
	dest_open = None
	if base is not None:
		name = os.path.basename(dest_path)
		dest_open = lambda: os.fdopen(os.open(*_at(base, name), _LANDING_READ_FLAGS, **_fd_of(base)), 'rb')
	try:
		if CONTENT_ONLY and HASH_SIZE > 0:
			src_st, dest_st = src_st or os.stat(src_path), dest_st or os.stat(dest_path)
			if stat.S_ISREG(src_st.st_mode) and stat.S_ISREG(dest_st.st_mode):
				return src_st.st_size == dest_st.st_size and _same_contents(src_path, dest_path, dest_open)
		return is_file_identical(src_path, dest_path, src_size, full_hash, src_st=src_st, dest_st=dest_st, dest_open=dest_open)
	except PermissionError:
		return False
	except OSError as e:
		if dest_open is not None and e.errno in (errno.ELOOP, errno.ENXIO):
			return False  # a link (O_NOFOLLOW) or a FIFO put at the landing since the look: the copy looks at it again
		raise

# The errors shutil.copystat's xattr copy ignores (3.6 to 3.14, together): a name the destination or the runner cannot have.
_XATTR_SKIPPED = frozenset((errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENODATA, errno.EINVAL, errno.EACCES))

def _settle_identical(src_path, src_st, dest, look, base=None):
	"""Give dest, a landing the skip check found identical to src_path (look: its lstat then; src_st: the source's, taken before
	the check read it), what the skip has always given it (9.86's _remove_stale_xattrs, copystat, _chown_like_cp, chmod and
	utime): stale portable xattrs removed and the source's set, owner and group as cp -a keeps them, the mode less set-user-ID
	or set-group-ID where the owner or group is not kept, the times where the modification time differs (a read time alone
	is left: reads move it), BSD flags last. Each only where it differs, all through one descriptor checked against the
	look. False, with nothing set, when dest is no longer the file looked at (a link, or
	another file's name, put there since: neither gets the source's owner or mode), for the caller to look again.

	On Linux an O_PATH descriptor, reached by its /proc/self/fd name (as _carry_over); elsewhere a regular file opened for
	reading without following a link (the check just read it), and a node by its name in base, its owner, times and (where
	chmod can be told so) mode set relative to base without following a link put at it, its xattrs (Linux without /proc)
	and BSD flags (macOS) by name (no dir_fd form). base: the door's result for dest's directory (None: its path), which the
	landing is opened in."""
	base = os.path.dirname(dest) if base is None else base
	leaf = os.path.basename(dest)
	fd = None
	try:
		if _PATH_FDS:
			fd = os.open(*_at(base, leaf), os.O_PATH | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0), **_fd_of(base))
		elif stat.S_ISREG(look.st_mode):
			fd = os.open(*_at(base, leaf), os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0),
				**_fd_of(base))
	except OSError as e:
		if e.errno in (errno.ENOENT, errno.ELOOP):
			return False  # gone, or a link put there (O_NOFOLLOW)
		raise
	try:
		now = os.fstat(fd) if fd is not None else os.lstat(*_at(base, leaf), **_fd_of(base))
		if (now.st_dev, now.st_ino, stat.S_IFMT(now.st_mode)) != (look.st_dev, look.st_ino, stat.S_IFMT(look.st_mode)):
			return False
		if fd is None:
			# a node, by its name in base: calls with a dir_fd form never follow a link put at it; xattrs have none
			path, how = _at(base, leaf)[0], dict(_fd_of(base), follow_symlinks=False)
			xpath, xhow = _io(base, leaf, dest), {'follow_symlinks': False}
		elif _PATH_FDS:
			path, how = '/proc/self/fd/%d' % fd, {}  # leads to the descriptor's file, and no further
			xpath, xhow = path, how
		else:
			path, how = fd, {}
			xpath, xhow = path, how
		owner_kept = group_kept = True
		chowned = os.name == 'posix' and (now.st_uid, now.st_gid) != (src_st.st_uid, src_st.st_gid)
		if chowned:
			owner_kept, group_kept = _chown_like_cp(path, src_st, **how)
		# xattrs after the owner, as cp -a: a chown clears a file capability (security.capability), and before the mode, which
		# may make the file read-only
		if hasattr(os, 'listxattr'):
			try:
				src_names = os.listxattr(src_path, follow_symlinks=False)
				dest_names = os.listxattr(xpath, **xhow)
			except OSError as e:
				if e.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP):
					raise
				src_names = dest_names = ()
			for name in dest_names:
				if name not in src_names and _is_portable_xattr(name):
					try:
						os.removexattr(xpath, name, **xhow)
					except OSError as e:
						if e.errno not in (errno.ENOTSUP, errno.EPERM, errno.ENODATA, errno.EACCES, errno.EINVAL, errno.ENOENT):
							raise
			for name in src_names:
				try:
					value = os.getxattr(src_path, name, follow_symlinks=False)
					if name not in dest_names or os.getxattr(xpath, name, **xhow) != value:
						os.setxattr(xpath, name, value, **xhow)
				except OSError as e:
					if e.errno not in _XATTR_SKIPPED:
						raise
		# As cp -a: no set-user-ID / set-group-ID bit for an owner / group not kept. A chown may have cleared them: set again.
		mode = stat.S_IMODE(src_st.st_mode) & ~((0 if owner_kept else stat.S_ISUID) | (0 if group_kept else stat.S_ISGID))
		if chowned or stat.S_IMODE(now.st_mode) != mode:
			# a node by its name in base: not through a link put at it where chmod can say so (macOS's lchmod; Linux has none,
			# so there, without /proc, the mode follows one)
			os.chmod(path, mode, **(how if os.chmod in os.supports_follow_symlinks else _fd_of(base) if fd is None else {}))
		if now.st_mtime_ns != src_st.st_mtime_ns:
			# The read time goes with it, as cp -a gives it; alone it is not chased: a read moves it (relatime moves one older
			# than the change time, which setting it makes it), so the skip check's own read would have it set on every rerun.
			os.utime(path, ns=(src_st.st_atime_ns, src_st.st_mtime_ns), **how)
		if getattr(src_st, 'st_flags', 0) != getattr(now, 'st_flags', 0):
			try:
				os.chflags(_io(base, leaf, dest), src_st.st_flags, follow_symlinks=False)
			except OSError as e:
				if e.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP):
					raise
		return True
	finally:
		if fd is not None:
			os.close(fd)

class _CopyFailure(str):
	"""A 'Copy failed' error in ERRORS that names its source (.source): the copy's check (_not_held) asks which sources
	failed. It goes back from a worker with the task's errors, as text does."""
	def __new__(cls, text, source):
		error = str.__new__(cls, text)
		error.source = source
		return error
	def __reduce__(self):
		return _CopyFailure, (str(self), self.source)

def _copy_failed(source, text, said=False):
	"""eprint(text), recorded as the failure of source's copy; said: recorded only (a refusal this process printed already:
	_Refused.said)."""
	if said:
		ERRORS.append(_CopyFailure(text, source))
		return
	eprint(text)
	if ERRORS and ERRORS[-1] == text:
		ERRORS[-1] = _CopyFailure(text, source)

def copy_file(src_path, dest_paths, full_hash=False, verbose=False, concurrent_processes=0):
	"""
	Copy a file from the source path to the destination path.

	Args:
		src_path (str): The path of the source file.
		dest_paths (list): The list of paths of the destination file.
		full_hash (bool, optional): Whether to perform a full hash comparison to determine if the files are identical. Defaults to False.
		verbose (bool, optional): Whether to print verbose output. Defaults to False.
		concurrent_processes (int, optional): The number of concurrent processes running in parralle. Used to calculate preemptive back-off to next dest threashold. Defaults to 0 ( Do not back-off ).

	Returns:
		tuple: A tuple containing the size of the copied file, the time taken for the copy operation, and a dictionary of symbolic links encountered during the copy.
	"""
	# The door's result for each landing's directory (_dest_directory): every name below a destination is looked at, made and
	# written relative to it; closed here whatever happens.
	bases = {}
	try:
		return _copy_file(src_path, dest_paths, full_hash, verbose, concurrent_processes, bases)
	finally:
		for base in bases.values():
			_close_base(base)

def _copy_file(src_path, dest_paths, full_hash, verbose, concurrent_processes, bases):
	"""copy_file's body; bases: {dest: the door's result for its landing's directory}, filled here, closed by copy_file."""
	global RANDOM_DESTINATION_SELECTION
	global NO_CREATE_DIR
	global CONTENT_ONLY
	symLinks = {}
	#task_to_run = []
	inDests = dest_paths
	dest_paths = _validate_paths(src_path, dest_paths)
	if dest_paths is None:
		return 0, 0, symLinks
	start_time = time.monotonic()
	# The source is looked at once (its kind, size, times), and each landing once (_landing_of): what the skip check and the
	# first attempt act on. A later attempt looks again.
	try:
		src_st = os.lstat(src_path)
	except OSError:
		src_st = None  # gone: cp says so
	src_is_link = src_st is not None and stat.S_ISLNK(src_st.st_mode)
	# Each destination's directories go through the door first (_dest_directory, nothing made: the file lands on one of
	# them), so a link there is replaced before anything looks below it, and what follows (the look, the skip check and its
	# metadata, the node probe, the copy) goes relative to the directory it holds (bases). A destination the door refuses is
	# left out, its refusal kept; one whose directory is missing (None) is made by the door at the copy (_cp_once).
	refused = {}
	for dest in dest_paths:
		try:
			bases[dest] = _dest_directory(os.path.dirname(_landing_path(dest)[0]), make=False, source=os.path.dirname(os.path.abspath(src_path)))
		except _Refused as e:
			refused[dest] = e
		except OSError:
			pass  # _cp_once meets it again, as the copy's error
	if refused:
		dest_paths = [dest for dest in dest_paths if dest not in refused]
		if not dest_paths:
			for refusal in refused.values():
				_copy_failed(src_path, refusal.text(src_path), refusal.said)
			return 0, time.monotonic() - start_time, symLinks
	looks = {dest: _landing_of(dest, bases.get(dest)) for dest in dest_paths}
	if not src_is_link:
		# cp would put the file inside a directory at its place (dest/x/x), and a move would then lose the source. A link there
		# is a name of the tree, replaced, unless the operator named it (_landing_of).
		usable = [dest for dest in dest_paths if not (looks[dest][1] is not None and stat.S_ISDIR(looks[dest][1].st_mode))]
		if not usable:
			eprint(f'Destination clash: {dest_paths[0]} is a directory, not copying {src_path} over it')
			return 0, 0, symLinks
		dest_paths = usable
	try:
		src_size = _allocated_size(src_st) if src_st is not None else get_file_size(src_path)
		copiedSize = 0
		src_kind = stat.S_IFMT(src_st.st_mode) if src_st is not None else None
		for dest in dest_paths:
			# A landing that changed between the skip check's look and its settling is looked at and judged again, as the door
			# bounds its own race.
			for _ in range(3):
				target, dest_st, _ = looks[dest]
				# Only a regular file, or a node of the source's own kind, can be the same: a link is never identical, and its
				# target's mode and owner are never touched.
				if src_st is None or src_is_link or dest_st is None or not (stat.S_ISREG(dest_st.st_mode) or
						(not stat.S_ISDIR(dest_st.st_mode) and stat.S_IFMT(dest_st.st_mode) == src_kind)):
					break
				if not _known_identical(src_path, target, src_st.st_size, full_hash, src_st=src_st, dest_st=dest_st, base=bases.get(dest)):
					break
				try:
					settled = CONTENT_ONLY or _settle_identical(src_path, src_st, target, dest_st, bases.get(dest))
				except OSError as e:
					# Identical, but what the skip gives it cannot be set (a read-only mount, another user's file): a failed
					# copy that names the landing.
					_copy_failed(src_path, f'Copy failed: ERROR copying {src_path} to {target}: it holds the source\'s content, but its owner, mode, times or xattrs cannot be set ({_why(e)})')
					return 0, time.monotonic() - start_time, symLinks
				if not settled:
					looks[dest] = _landing_of(dest, bases.get(dest))
					continue
				if SYNC_COPIES:
					# A copy an earlier run left (or a plain copy before this move) may still be in memory.
					try:
						_sync_landing(target, bases.get(dest))
					except _NotOnDisk as e:
						_copy_failed(src_path, f'Copy failed: ERROR copying {src_path} to {target}: {e}')
						return 0, time.monotonic() - start_time, symLinks
				endTime = time.monotonic()
				return 0, endTime - start_time , symLinks #, task_to_run
			else:
				_copy_failed(src_path, f'Copy failed: ERROR copying {src_path} to {target}: it kept changing while it was looked at')
				return 0, time.monotonic() - start_time, symLinks
		bak_dest_paths = dest_paths.copy()
		using_bak_paths = False
		if src_is_link:
			symLinks[src_path] = dest_paths
		else:
			if not _can_open(src_path, src_st):
				# No cp at all: cp -a takes away a destination with other names before it finds it cannot read.
				_copy_failed(src_path, f'Copy failed: cannot open {src_path}, leaving {dest_paths} as they are')
				return 0, time.monotonic() - start_time, symLinks
			if not _can_make_node(src_path, dest_paths, src_st, bases):
				_copy_failed(src_path, f'Copy failed: cannot make a device node like {src_path} here, leaving {dest_paths} as they are')
				return 0, time.monotonic() - start_time, symLinks
			copied = False
			copy_error_start = len(ERRORS)
			cp_flags = _cp_flags(src_path, src_st)
			# Why each destination failed, the last attempt's (the door's refusals first).
			failures = dict(refused)
			while (not copied) and dest_paths:
				if RANDOM_DESTINATION_SELECTION:
					idx = random.randrange(len(dest_paths))
				else:
					idx = 0
				dest_path = dest_paths.pop(idx)
				dest_free_space = get_free_space_bytes(os.path.dirname(dest_path))
				if not using_bak_paths:
					to_skip = False
					if dest_free_space < src_size:
						if verbose:
							print(f'\nEstimation: not enough space on {dest_path} to copy {src_path}')
							print(f'Free space: {format_bytes(dest_free_space)}B, Required: {format_bytes(src_size)}B')
						to_skip = True
					if not to_skip and concurrent_processes > 0:
						estimated_concurrent_write_size = max(src_size * log(concurrent_processes),1)
						backoff_threashold = dest_free_space / estimated_concurrent_write_size
						if backoff_threashold < random.random():
							if verbose:
								print(f'\nPreemptively backing off on {dest_path} to copy {src_path} with chance {1- backoff_threashold:.2f}')
								print(f'Estimated concurrent write size: {format_bytes(estimated_concurrent_write_size)}B, Free space: {format_bytes(dest_free_space)}B')
							to_skip = True
						elif verbose:
							print(f'\nLottery Win. Continuing on {dest_path} to copy {src_path} with chance {backoff_threashold:.2f}')
							print(f'Estimated concurrent write size: {format_bytes(estimated_concurrent_write_size)}B, Free space: {format_bytes(dest_free_space)}B')
					if to_skip:
						if not dest_paths:
							using_bak_paths = True
							dest_paths = bak_dest_paths
							if verbose:
								print(f'Pre Gracious copy pass failed. Force trying all destination paths for {src_path}')
						continue
				if verbose:
					print(f'\nTrying to copy from {src_path} to {dest_path}')
				# The forced pass tries once more without --sparse. Its attempts may run short of room: the warning (_space_warning)
				# comes before cp, and cp still tries, as GNU cp does.
				space = (dest_free_space, src_size) if dest_free_space < src_size else None
				for sparse in [True] + ([False] if using_bak_paths else []):
					try:
						if os.name == 'posix':
							if bases.get(dest_path) is None:
								# Missing at the look: judged as the look is (a dangling link the operator named makes no directory),
								# then the door makes it (not under -ncd) and holds it for the copy (one door call a landing otherwise).
								_judge_landing(dest_path, *(looks.get(dest_path) or _landing_of(dest_path, None)))
								bases[dest_path] = _dest_directory(os.path.dirname(_landing_path(dest_path)[0]), make=not NO_CREATE_DIR,
									source=os.path.dirname(os.path.abspath(src_path)))
							# The first attempt at a destination acts on its look; a later one looks again.
							copiedSize = _cp_once(src_path, dest_path, cp_flags, sparse, space, look=looks.pop(dest_path, None), src_st=src_st,
								base=bases[dest_path])
						elif CONTENT_ONLY:
							shutil.copy(src_path, dest_path, follow_symlinks=False)
						else:
							shutil.copy2(src_path, dest_path, follow_symlinks=False)
						if SYNC_COPIES:
							_sync_landing(dest_path, bases.get(dest_path))
						copied = True
						break
					except _Refused as e:
						# What is there will not be written, by any attempt.
						failures[dest_path] = e
						break
					except Exception as e:
						failures[dest_path] = e
						if os.name != 'posix':
							break
						if sparse and using_bak_paths and not dest_paths and not verbose:
							print(f'Error copying {src_path} to {dest_path}, {e}')
							print(f'Trying to copy from {src_path} to {dest_path} without sparse')
						elif verbose:
							print(f'Error copying {src_path} to {dest_path}, {e}')
				if copied:
					break
				if using_bak_paths and not dest_paths:
					for failure in failures.values():
						if isinstance(failure, _Refused):
							_copy_failed(src_path, failure.text(src_path), failure.said)
					last = failures[dest_path]
					if isinstance(last, _KeptOld):
						_copy_failed(src_path, f'Copy failed: cannot copy {src_path} to {dest_path}, leaving {dest_path} as it is: {last}')
					elif not isinstance(last, _Refused):
						_copy_failed(src_path, f'Copy failed: ERROR copying {src_path} to {dest_path}, No more destination paths to try: {last}')
					return 0, time.monotonic() - start_time, symLinks
				elif verbose:
					print(f'Retrying with a different destination path in {dest_paths}')
				if not copied and not using_bak_paths and not dest_paths:
					if verbose:
						print(f'Gracious copy pass failed. Force trying all destination paths for {src_path}')
					using_bak_paths = True
					dest_paths = bak_dest_paths
			if not copied:
				_copy_failed(src_path, f'Copy failed: FAILED to copy {src_path} to {dest_paths}')
				return 0, time.monotonic() - start_time, symLinks
			# A successful retry resolves this file's failed attempts; a space warning stays (it is repeated at the run's end).
			# Preserve errors from earlier files and from path validation above.
			ERRORS[copy_error_start:] = [error for error in ERRORS[copy_error_start:] if error.startswith('Space warning:')]
			if verbose:
				print(f'Copied {src_path} to {dest_path}')
				print(f'Estimated remaining size: {format_bytes(dest_free_space - copiedSize)}B')
	except Exception as e:
		# recorded as this source's failure (_CopyFailure): the copy check then leaves it out, as it does a failed copy
		_copy_failed(src_path, f'Copy process crashed: Fatal Error copying {src_path} to {inDests}: {e}')
		if verbose or not isinstance(e, OSError):
			# a file that cannot be read or reached says so in one line; anything else is a bug and keeps its traceback
			import traceback
			print(traceback.format_exc())
		return 0, time.monotonic() - start_time, symLinks #, task_to_run
	if not copiedSize:
		copiedSize = src_size
	endTime = time.monotonic()
	return copiedSize, endTime - start_time , symLinks #, task_to_run

def copy_files_bulk(src_files, dst_files,src_path, full_hash=False, verbose=False, concurrent_processes=0):
	"""
	Copy multiple files from source to destination.

	Args:
		src_files (list): List of source file paths.
		dst_files (list): List of original destination file paths.
		src_path (str): Source directory path.
		full_hash (bool, optional): Whether to calculate full hash of files. Defaults to False.
		verbose (bool, optional): Whether to display verbose output. Defaults to False.
		concurrent_processes (int, optional): Number of concurrent processes to pass to copy_file. Defaults to 0.

	Returns:
		tuple: A tuple containing the total size of copied files, total time taken for copying, and a dictionary of symbolic links.

	"""
	total_size = 0
	total_time = 0
	symLinks = {}
	# tasks_to_run = []
	for src in src_files:
		source_relative_path = os.path.relpath(src, src_path)
		dests = [os.path.join(dest_path, source_relative_path) for dest_path in dst_files]
		size , cpTime , rtnSymLinks = copy_file(src, dests, full_hash, verbose,concurrent_processes)
		total_size += size
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_size , total_time, symLinks

def copy_files_bulk_batch(current_jobs, full_hash=False, verbose=False, concurrent_processes=0):
	total_size = 0
	total_time = 0
	symLinks = {}
	# tasks_to_run = []
	for (src_path, dest_paths, f) in current_jobs:
		source_relative_path = os.path.relpath(f, src_path)
		if source_relative_path == '.':
			dests = dest_paths
		else:
			dests = [os.path.join(dest_path, source_relative_path) for dest_path in dest_paths]
		size , cpTime , rtnSymLinks = copy_file(f, dests, full_hash, verbose,concurrent_processes)
		total_size += size
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_size , total_time, symLinks

def copy_file_list_parallel(file_list, links, src_path, dest_paths, max_workers, full_hash=False, verbose=False, files_per_job=1,estimated_size = 0):
	"""
	Copy a list of files in parallel using multiple workers.

	Args:
		file_list (list): List of file paths to be copied.
		links (list): List of symbolic links to be created.
		src_path (str): Source directory path.
		dest_paths (list): Destination directory path lists.
		max_workers (int): Maximum number of worker processes to use.
		full_hash (bool, optional): Whether to perform full file hash comparison. Defaults to False.
		verbose (bool, optional): Whether to print verbose output. Defaults to False.
		files_per_job (int, optional): Number of files to be copied per job. Defaults to 1.

	Returns:
		tuple: A tuple containing the following information:
			- copy_counter (int): Total number of files copied.
			- copy_size_counter (int): Total size of files copied in bytes.
			- symLinks (dict): Dictionary mapping symbolic links to their destination paths.
			- file_list (frozenset): Frozen set of remaining files to be copied.
	"""
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.1
	dest_paths = _validate_paths(src_path, dest_paths)
	if dest_paths is None:
		return 0, 0, {}, frozenset()
	file_list_iterator = iter(_path_order(file_list))
	start_time = time.monotonic()
	lastRefreshTime = start_time
	total_files = len(file_list)
	futures = {}
	files_per_job = max(1,files_per_job)
	symLinks = _link_dests(links, src_path, dest_paths)
	if len(file_list) == 0:
		return 0, 0, symLinks , frozenset()
	print(f"Processing {len(file_list)} files with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=estimated_size,last_num_job_for_stats=max(1,max_workers//10),process_word='Copied',suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	last_transition_time = time.monotonic()
	last_under_limit = True
	with _copy_pool(max_workers) as executor:
		while file_list_iterator or futures:
			# counter = 0
			under_limit = apb.under_rate_limit()
			while file_list_iterator and len(futures) < max_scheduled_jobs and under_limit:
				src_files = []
				try:
					# generate some noise from 0.9 to 1.1 to apply to the files per job to attempt spreading out the job scheduling
					noise = random.uniform(0.9, 1.1)
					for _ in range(max(1,round(files_per_job * noise))):
						src_file = next(file_list_iterator)
						src_files.append(src_file)
					future = _submit_task(executor, copy_files_bulk, src_files, dest_paths,src_path, full_hash, verbose,len(futures))
					futures[future] = src_files
					# counter += 1
				except StopIteration:
					if src_files:
						future = _submit_task(executor, copy_files_bulk, src_files, dest_paths,src_path, full_hash, verbose,len(futures))
						futures[future] = src_files
					file_list_iterator = None

			done = _wait(futures, executor)

			#print('\n',len(done) ,'\t', len(futures))
			if file_list_iterator and len(done) > 1 and len(done) / len(futures) > 0.1:
				if verbose:
					print(f'\nTransfer is fast, doubling files per job to {files_per_job * 2}')
				files_per_job *= 2

			current_iteration_total_run_time = 0
			for future in done:
				src_files = futures.pop(future)
				copied_file_count_this_run = len(src_files)
				cpSize, cpTime, rtnSymLinks = _task_result(future, (0, 0, {}), 'Copy process crashed')
				current_iteration_total_run_time += cpTime
				apb.update(num_files=copied_file_count_this_run,cpSize=cpSize,cpTime=cpTime,files_per_job=files_per_job)
				symLinks.update(rtnSymLinks)
			apb.scheduled_jobs = len(futures)
			apb.print_progress()
			if done:
				if verbose:
						print(f'\nAverage cptime is {current_iteration_total_run_time / len(done):0.2f} for {len(done)} jobs with {copied_file_count_this_run} files each')
				if file_list_iterator and copied_file_count_this_run == round(files_per_job) and time.monotonic() - lastRefreshTime > 1:
					files_per_job //= MAGIC_NUMBER
					if verbose:
						print(f'\nCompletion time is long, changing files per job to {files_per_job}')
				#elif file_list_iterator and copied_file_count_this_run == files_per_job and current_iteration_total_run_time / len(done) < 1:
				elif file_list_iterator and copied_file_count_this_run == round(files_per_job) and time.monotonic() - lastRefreshTime < 0.1:
					files_per_job *= MAGIC_NUMBER
					if verbose:
						print(f'\nCompletion time is short, changing files per job to {files_per_job}')
				if files_per_job < 1:
					files_per_job = 1
			else:
				if verbose:
					print('\nNo job is done yet, sleeping for a while and reducing max scheduled jobs by 1')
				max_scheduled_jobs = max(1, max_scheduled_jobs - 1)
				if not under_limit:
					if verbose:
						print('\nWe had hit the rate limit and no job is running currently, setting files per job to 1')
					files_per_job = 1
				time.sleep(apb.refresh_interval)
			if under_limit != last_under_limit:
				last_transition_time = time.monotonic()
				last_under_limit = under_limit
			elif time.monotonic() - last_transition_time > 1:
				if under_limit:
					max_scheduled_jobs =  min(max_scheduled_jobs * MAGIC_NUMBER, max_workers * 1.2)
					if verbose:
						print(f'\nUnder rate limit for a while, increasing max scheduled jobs to {max_scheduled_jobs}')
				else:
					max_scheduled_jobs = max(1, max_scheduled_jobs / MAGIC_NUMBER)
					if verbose:
						print(f'\nOver rate limit for a while, decreasing max scheduled jobs to {max_scheduled_jobs}')
			lastRefreshTime = time.monotonic()
	endTime = time.monotonic()
	apb.stop()
	_print_operation_summary(apb, endTime - start_time, show_bandwidth=True)
	return apb.item_counter, apb.size_counter , symLinks ,file_list

def get_copy_file_job_iter(jobs):
	"""(src_path, dest_paths, file) for each file of each job, a job's files in path order (_path_order)."""
	for src_path, dest_paths, file_list in jobs:
		for f in _path_order(file_list):
			yield (src_path, dest_paths, f)

def _path_order(paths):
	"""paths sorted as strings: every directory's subtree is one run (the paths that start with its own and a separator), so a
	worker's consecutive landings share their directories, and the door (_dest_directory) enters each directory of a run
	once. 0.07 s a 100,000 paths on 3.14, on a loaded host (the 9.88 plan: 0.37 s a million), once a source."""
	return sorted(paths)

def copy_file_list_parallel_batch(jobs, max_workers, full_hash=False, verbose=False, files_per_job=1,total_item_count=0,total_size_count=0):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.1
	symLinks = {}
	# src_path, dest_paths, file_list
	job_list_iterator = get_copy_file_job_iter(jobs)
	start_time = time.monotonic()
	lastRefreshTime = start_time
	futures = {}
	files_per_job = max(1,files_per_job)

	print(f"Processing {total_item_count} files with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=total_item_count,total_size=total_size_count,last_num_job_for_stats=max(1,max_workers//10),process_word='Copied',suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	last_transition_time = time.monotonic()
	last_under_limit = True
	with _copy_pool(max_workers) as executor:
		while job_list_iterator or futures:
			# counter = 0
			under_limit = apb.under_rate_limit()
			while job_list_iterator and len(futures) < max_scheduled_jobs and under_limit:
				current_jobs = []
				try:
					# generate some noise from 0.9 to 1.1 to apply to the files per job to attempt spreading out the job scheduling
					noise = random.uniform(0.9, 1.1)
					for _ in range(max(1,round(files_per_job * noise))):
						job = next(job_list_iterator)
						current_jobs.append(job)
					future = _submit_task(executor, copy_files_bulk_batch, current_jobs, full_hash, verbose,len(futures))
					futures[future] = current_jobs
					# counter += 1
				except StopIteration:
					if current_jobs:
						future = _submit_task(executor, copy_files_bulk_batch, current_jobs, full_hash, verbose,len(futures))
						futures[future] = current_jobs
					job_list_iterator = None

			done = _wait(futures, executor)

			#print('\n',len(done) ,'\t', len(futures))
			if job_list_iterator and len(done) > 1 and len(done) / len(futures) > 0.1:
				if verbose:
					print(f'\nTransfer is fast, doubling files per job to {files_per_job * 2}')
				files_per_job *= 2

			current_iteration_total_run_time = 0
			for future in done:
				current_jobs = futures.pop(future)
				copied_file_count_this_run = len(current_jobs)
				cpSize, cpTime, rtnSymLinks = _task_result(future, (0, 0, {}), 'Copy process crashed')
				current_iteration_total_run_time += cpTime
				apb.update(num_files=copied_file_count_this_run,cpSize=cpSize,cpTime=cpTime,files_per_job=files_per_job)
				symLinks.update(rtnSymLinks)
			apb.scheduled_jobs = len(futures)
			apb.print_progress()
			if done:
				if verbose:
						print(f'\nAverage cptime is {current_iteration_total_run_time / len(done):0.2f} for {len(done)} jobs with {copied_file_count_this_run} files each')
				if job_list_iterator and copied_file_count_this_run == round(files_per_job) and time.monotonic() - lastRefreshTime > 1:
					files_per_job //= MAGIC_NUMBER
					if verbose:
						print(f'\nCompletion time is long, changing files per job to {files_per_job}')
				#elif file_list_iterator and copied_file_count_this_run == files_per_job and current_iteration_total_run_time / len(done) < 1:
				elif job_list_iterator and copied_file_count_this_run == round(files_per_job) and time.monotonic() - lastRefreshTime < 0.1:
					files_per_job *= MAGIC_NUMBER
					if verbose:
						print(f'\nCompletion time is short, changing files per job to {files_per_job}')
				if files_per_job < 1:
					files_per_job = 1
			else:
				if verbose:
					print('\nNo job is done yet, sleeping for a while and reducing max scheduled jobs by 1')
				max_scheduled_jobs = max(1, max_scheduled_jobs - 1)
				if not under_limit:
					if verbose:
						print('\nWe had hit the rate limit and no job is running currently, setting files per job to 1')
					files_per_job = 1
				time.sleep(apb.refresh_interval)
			if under_limit != last_under_limit:
				last_transition_time = time.monotonic()
				last_under_limit = under_limit
			elif time.monotonic() - last_transition_time > 1:
				if under_limit:
					max_scheduled_jobs =  min(max_scheduled_jobs * MAGIC_NUMBER, max_workers * 1.2)
					if verbose:
						print(f'\nUnder rate limit for a while, increasing max scheduled jobs to {max_scheduled_jobs}')
				else:
					max_scheduled_jobs = max(1, max_scheduled_jobs / MAGIC_NUMBER)
					if verbose:
						print(f'\nOver rate limit for a while, decreasing max scheduled jobs to {max_scheduled_jobs}')
			lastRefreshTime = time.monotonic()
	endTime = time.monotonic()
	apb.stop()
	_print_operation_summary(apb, endTime - start_time, show_bandwidth=True)
	return apb.item_counter, apb.size_counter , symLinks 

def copy_files_parallel(src_path, dest_paths, max_workers, full_hash=False, verbose=False,files_per_job=1,parallel_file_listing=False,exclude=None):
	dest_paths = _validate_paths(src_path, dest_paths)
	if dest_paths is None:
		return 0, 0, set(), frozenset()
	if exclude and is_excluded(src_path,exclude):
		return 0, 0 , set(), frozenset()

	if not os.path.isdir(src_path):
		src_size, _ , symLinks = copy_file(src_path, dest_paths,full_hash=full_hash, verbose=verbose)
		return 1, src_size , symLinks , frozenset([src_path])
	file_list, links, init_size, folders = get_file_list(src_path, max_workers=max_workers, exclude=exclude, parallel_file_listing=parallel_file_listing)
	total_files = len(file_list)
	print(f"Number of files: {total_files}")
	print(f"Number of links: {len(links)}")
	print(f"Number of folders: {len(folders)}")
	print(f"Estimated size: {format_bytes(init_size)}B")
	return copy_file_list_parallel(file_list=file_list,links=links,src_path=src_path, dest_paths=dest_paths, max_workers=max_workers, full_hash=full_hash, verbose=verbose,files_per_job=files_per_job,estimated_size = init_size)

def _space_needed(file_list, src_path, dest_paths, max_workers):
	"""The bytes a copy of file_list (src_path's files) into dest_paths will take, as -enes counts it: one lstat of each source
	and of its landing at the first destination holding one. No landing: the source's allocated size; a regular file of the
	same size whose time agrees (any time under -co): nothing, the copy skips it; one rewritten in place (a regular file with one
	name, not under -wb): what the source takes beyond it; any other landing (a rewrite beside it, under -wb or for a file of
	several names, or not a regular file): the source's allocated size. A source gone since the listing takes nothing."""
	def look(chunk):
		need = 0
		for file in chunk:
			try:
				st = os.lstat(file)
			except OSError:
				continue
			relative = os.path.relpath(file, src_path)
			for dest_path in dest_paths:
				try:
					landing = os.lstat(dest_path if relative == '.' else os.path.join(dest_path, relative))
					break
				except OSError:
					pass
			else:
				need += _allocated_size(st)
				continue
			regular = stat.S_ISREG(landing.st_mode)
			if regular and landing.st_size == st.st_size and (CONTENT_ONLY or _mtimes_agree(st.st_mtime_ns, landing.st_mtime_ns)):
				continue
			if regular and landing.st_nlink == 1 and not WRITE_BESIDE:
				need += max(0, _allocated_size(st) - _allocated_size(landing))
			else:
				need += _allocated_size(st)
		return need
	it = iter(file_list)
	return sum(_looked_at(iter(lambda: list(itertools.islice(it, 1024)), []), look, max_workers))

def _not_enough_space(file_list, src_path, dest_paths, listed_size, max_workers):
	"""-enes: whether to skip src_path, whose copy into dest_paths does not fit their free space. Only where its listed size does
	not fit is what the copy will write counted (_space_needed): a rerun that has little or nothing to copy goes on."""
	free = sum(get_free_space_bytes(dest_path) for dest_path in dest_paths)
	if free >= listed_size:
		return False
	need = _space_needed(file_list, src_path, dest_paths, max_workers)
	if free >= need:
		return False
	eprint(f'Not enough space: free space on {dest_paths} too small, skipping copy {src_path}')
	print(f'Free space: {format_bytes(free)}B, Required: {format_bytes(need)}B')
	return True

def copy_files_parallel_batch(jobs, max_workers, full_hash=False, verbose=False,files_per_job=1,parallel_file_listing=False,exclude=None,
							  exit_not_enough_space=False,dropped=None):
	newJobs = []
	total_symLinks = {}
	total_files = []
	total_item_count = 0
	total_size_count = 0
	total_link_count = 0
	total_folder_count = 0
	for src_path, dest_paths in jobs:
		lost = dropped.get((src_path, tuple(dest_paths))) if dropped else None
		dest_paths = _validate_paths(src_path, dest_paths)
		if dest_paths is None:
			continue
		if exclude and is_excluded(src_path,exclude):
			continue
		file_list , links,init_size,folders  = get_file_list(src_path, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)
		if lost:
			file_list, links = file_list - lost, links - lost
		if exit_not_enough_space and _not_enough_space(file_list, src_path, dest_paths, init_size, max_workers):
			continue
		newJobs.append((src_path, dest_paths, file_list))
		total_item_count += len(file_list)
		total_size_count += init_size
		total_link_count += len(links)
		total_folder_count += len(folders)
		total_files.extend(file_list)
		total_symLinks.update(_link_dests(links, src_path, dest_paths))

	print(f"Number of files: {total_item_count}")
	print(f"Number of links: {total_link_count}")
	print(f"Number of folders: {total_folder_count}")
	print(f"Estimated size: {format_bytes(total_size_count)}B")
	item_counter, size_counter , symLinks =  copy_file_list_parallel_batch(newJobs, max_workers=max_workers, full_hash=full_hash, verbose=verbose,files_per_job=files_per_job,total_item_count=total_item_count,total_size_count=total_size_count)
	total_symLinks.update(symLinks)
	return item_counter, size_counter , total_symLinks ,frozenset(total_files)

def copy_files_serial(src_path, dest_paths, full_hash=False, verbose=False,exclude=None):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	dest_paths = _validate_paths(src_path, dest_paths)
	if dest_paths is None:
		return 0, 0 , set(), frozenset()
	if exclude and is_excluded(src_path,exclude):
		return 0, 0 , set(), frozenset()
	if not os.path.isdir(src_path):
		src_size, _ , symLinks = copy_file(src_path, dest_paths,full_hash=full_hash, verbose=verbose)
		return 1, src_size , symLinks , frozenset([src_path])
	file_list,links,init_size,_ = get_file_list(src_path,exclude=exclude,parallel_file_listing=False)
	links = set(links)
	total_files = len(file_list)
	print(f"Number of files: {total_files}")
	start_time = time.monotonic()
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=init_size,last_num_job_for_stats=1,process_word='Copied',suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	for file in _path_order(file_list):
		_check_stop()
		srcRelativePath = os.path.relpath(file, src_path)
		size, cpTime ,rtnSymLinks = copy_file(file, [os.path.join(dest_path, srcRelativePath) for dest_path in dest_paths],full_hash = full_hash, verbose=verbose)
		#update_progress_bar(copy_counter, copy_size_counter, total_files, start_time)
		apb.update(num_files=1,cpSize=size,cpTime=cpTime,files_per_job=1)
		links.update(rtnSymLinks)
		apb.rate_limit()
	symLinks = _link_dests(links, src_path, dest_paths)
	endTime = time.monotonic()
	apb.stop()
	_print_operation_summary(apb, endTime - start_time, show_bandwidth=True)
	return apb.item_counter, apb.size_counter , symLinks , frozenset(file_list)

def _link_dests(links, src_path, dest_paths):
	"""{link: [where it goes in each dest]} for the symlinks a listing of src_path found."""
	return {link: [os.path.join(dest_path, os.path.relpath(link, src_path)) for dest_path in dest_paths] for link in links}

def copy_files_serial_batch(jobs, full_hash=False, verbose=False,exclude=None,exit_not_enough_space=False,dropped=None):
	# skip if src path or dest path is longer than 4096 characters
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	newJobs = []
	total_symLinks = {}
	total_files = []
	copied_count = 0
	copied_size = 0
	total_item_count = 0
	total_size_count = 0
	for src_path, dest_paths in jobs:
		lost = dropped.get((src_path, tuple(dest_paths))) if dropped else None
		dest_paths = _validate_paths(src_path, dest_paths)
		if dest_paths is None:
			continue
		if exclude and is_excluded(src_path,exclude):
			continue
		if not os.path.isdir(src_path) or os.path.islink(src_path):
			if lost and src_path in lost:
				continue
			_check_stop()
			src_size, _ , symLinks = copy_file(src_path, dest_paths,full_hash=full_hash, verbose=verbose)
			#return 1, src_size , symLinks , frozenset([src_path])
			copied_count += 1
			copied_size += src_size
			total_item_count += 1
			total_size_count += src_size
			total_symLinks.update(symLinks)
			total_files.append(src_path)
			continue
		file_list,links,init_size,_ = get_file_list(src_path,exclude=exclude,parallel_file_listing=False)
		if lost:
			file_list, links = file_list - lost, links - lost
		if exit_not_enough_space and _not_enough_space(file_list, src_path, dest_paths, init_size, 1):
			continue
		newJobs.append((src_path, dest_paths, file_list, set(links), init_size))
		total_item_count += len(file_list)
		total_size_count += init_size
		total_files.extend(file_list)
	print(f"Number of files: {total_item_count}")
	print(f"Estimated size: {format_bytes(total_size_count)}B")
	if total_item_count == 0:
		# Nothing to copy, but the directories may still hold symlinks to recreate.
		for src_path, dest_paths, _, links, _ in newJobs:
			total_symLinks.update(_link_dests(links, src_path, dest_paths))
		return 0, 0, total_symLinks, frozenset()
	start_time = time.monotonic()
	apb = Adaptive_Progress_Bar(total_count=total_item_count,total_size=total_size_count,last_num_job_for_stats=1,process_word='Copied',suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	apb.item_counter = copied_count
	apb.size_counter = copied_size
	for src_path, dest_paths, file_list, links, init_size in newJobs:
		for file in _path_order(file_list):
			_check_stop()
			srcRelativePath = os.path.relpath(file, src_path)
			size, cpTime ,rtnSymLinks = copy_file(file, [os.path.join(dest_path, srcRelativePath) for dest_path in dest_paths],full_hash = full_hash, verbose=verbose)
			#update_progress_bar(copy_counter, copy_size_counter, total_files, start_time)
			apb.update(num_files=1,cpSize=size,cpTime=cpTime,files_per_job=1)
			links.update(rtnSymLinks)
			apb.rate_limit()
		total_symLinks.update(_link_dests(links, src_path, dest_paths))
	endTime = time.monotonic()
	apb.stop()
	_print_operation_summary(apb, endTime - start_time, show_bandwidth=True)
	return apb.item_counter, apb.size_counter , total_symLinks , frozenset(total_files)

class copy_scheduler:
	def __init__(self, max_workers = multiprocessing.cpu_count(), full_hash=False, verbose=False,files_per_job=1,
			  parallel_file_listing=False,exclude=None, exit_not_enough_space=False, dropped=None):
		self.max_workers = max_workers
		self.full_hash = full_hash
		self.verbose = verbose	
		self.files_per_job = files_per_job
		self.parallel_file_listing = parallel_file_listing
		self.exclude = exclude
		self.dir_sync_job = []
		self.copy_job = []
		#copy_counter, copy_size_counter , rtnSymLinks , file_list 
		self.copy_counter = 0
		self.copy_size_counter = 0
		self.total_sym_links = {}
		self.total_file_list = []
		# The directory sync's records, for the finish (finish_directories)
		self.dir_records = []
		self.exit_not_enough_space = exit_not_enough_space
		# See _settle_destinations.
		self.dropped = dropped
	def add_dir_sync(self, src, dests):
		self.dir_sync_job.append((src, dests))
	def add_copy(self, src, dests):
		self.copy_job.append((src, dests))
	def process(self):
		if self.dir_sync_job:
			start_time = time.monotonic()
			if self.max_workers == 1:
				links, records = sync_directories_serial_batch(self.dir_sync_job,exclude=self.exclude)
			else:
				links, records = sync_directories_parallel_batch(self.dir_sync_job, max_workers=self.max_workers,verbose=self.verbose,exclude=self.exclude,parallel_file_listing=self.parallel_file_listing)
			self.total_sym_links.update(links)
			self.dir_records += records
			endTime = time.monotonic()
			print(f"\nTime taken to sync directory: {endTime-start_time:0.4f} seconds")
			self.dir_sync_job = []
		if self.copy_job:
			global HASH_SIZE
			if HASH_SIZE == 0:
				print("Using file attributes only for skipping")
			elif xxhash_available:
				print("Using xxhash for skipping")
			else:
				print("Using blake2b for skipping")
			if self.max_workers == 1:
				copy_counter, copy_size_counter , rtnSymLinks , file_list = copy_files_serial_batch(self.copy_job, full_hash = self.full_hash,verbose=self.verbose,exclude=self.exclude,exit_not_enough_space=self.exit_not_enough_space,dropped=self.dropped)
				#total_file_list.update(trim_paths(file_list,src))
				#self.total_file_list.update(trim_paths(rtnSymLinks.keys(),src))
			else:
				copy_counter, copy_size_counter , rtnSymLinks , file_list = copy_files_parallel_batch(self.copy_job, max_workers=self.max_workers,full_hash = self.full_hash,verbose=self.verbose,files_per_job=self.files_per_job,parallel_file_listing=self.parallel_file_listing,exclude=self.exclude,exit_not_enough_space=self.exit_not_enough_space,dropped=self.dropped)
				#total_file_list.update(trim_paths(file_list,src))
				#self.total_file_list.update(trim_paths(rtnSymLinks.keys(),src))
			print(f'Total files copied:     {copy_counter}')
			print(f'Total size copied:      {format_bytes(copy_size_counter)}B')
			print(f'Total files discovered: {len(self.total_file_list)}')
			self.copy_counter += copy_counter
			self.copy_size_counter += copy_size_counter
			self.total_sym_links.update(rtnSymLinks)
			self.total_file_list.extend(file_list)
			self.copy_job = []
		return self.total_file_list, self.total_sym_links, self.dir_records
	def clear(self,clear_pending_jobs=False):
		self.copy_counter = 0
		self.copy_size_counter = 0
		self.total_sym_links = {}
		self.total_file_list = []
		self.dir_records = []
		if clear_pending_jobs:
			self.dir_sync_job = []
			self.copy_job = []


#%% ---- Copy Directories ----
def _below(dest, relative):
	"""dest/relative; the directory itself (relative '.') is dest, not 'dest/.', which does not exist when dest is a file."""
	return dest if relative == os.curdir else os.path.join(dest, relative)

def _make_dest_directory(src_path, dest_path):
	"""dest_path as the directory src_path lands on, made if missing, through the one door (_dest_directory): below a
	named destination a link, FIFO or node there is replaced by the directory. Returns the door's result for it (a
	descriptor the caller closes, _close_base, or its path); None when a file is in the way, or what is there cannot be
	replaced, which is reported."""
	named = _landing_path(dest_path)[1]
	if os.path.lexists(dest_path) and not os.path.isdir(dest_path) and (named or stat.S_ISREG(os.lstat(dest_path).st_mode)):
		# A file (or a link to one the operator named) is no directory's to take: its mode and owner stay, and nothing can be
		# made below it.
		eprint(f'Destination clash: {dest_path} is not a directory, not syncing the directory {src_path} onto it')
		return None
	try:
		return _dest_directory(dest_path, source=src_path)
	except _Refused as e:
		_say_refused(e, f'the directory {src_path}')
		return None
	except FileExistsError:
		print(f"Destination path {dest_path} maybe a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...", file=sys.stderr)
		return dest_path

def _sync_xattrs(src_path, dest):
	"""Give the directory dest src_path's xattrs: stale portable ones removed (_remove_stale_xattrs' rule), and each of the
	source's set where dest lacks it or holds another value, so a rerun sets none. The errors copystat ignores are ignored
	(Python 3.14's set: a name the destination's filesystem or the runner may not have). Returns whether it tried to change
	any (an ACL is an xattr, and setting one sets the mode's group bits)."""
	if not hasattr(os, 'listxattr'):
		return False
	try:
		src_names = os.listxattr(src_path)
		dest_names = set(os.listxattr(dest))
	except OSError as e:
		if e.errno not in (errno.ENOTSUP, errno.ENODATA, errno.EINVAL):
			raise
		return False
	changed = False
	wanted = set(src_names)
	for name in dest_names:
		if name in wanted or not _is_portable_xattr(name):
			continue
		changed = True
		try:
			os.removexattr(dest, name)
		except OSError as e:
			if e.errno not in (errno.ENOTSUP, errno.EPERM, errno.ENODATA, errno.EACCES, errno.EINVAL, errno.ENOENT):
				raise
	for name in src_names:
		try:
			value = os.getxattr(src_path, name)
			if name in dest_names and os.getxattr(dest, name) == value:
				continue
			changed = True
			os.setxattr(dest, name, value)
		except OSError as e:
			if e.errno not in (errno.EPERM, errno.ENOTSUP, errno.ENODATA, errno.EINVAL, errno.EACCES):
				raise
	return changed

def _held_directory(held, dest_path):
	"""What a call that acts on a directory itself takes for it (os.stat, chmod, chown, the xattr calls): its /proc/self/fd
	entry for a held O_PATH descriptor (Linux with /proc: an O_PATH descriptor takes no fchmod, fchown or flistxattr), the
	descriptor itself for a held read one (macOS, Linux without /proc), else dest_path (held a path: nothing held)."""
	if isinstance(held, int):
		return '/proc/self/fd/%d' % held if _PATH_FDS else held
	return dest_path

def _start_directory(src_path, dest_path, st, held=None):
	"""The directory sync's part of dest_path's metadata (src_path's lstat st): what does not stop the copy writing into it,
	each made only where dest_path differs. Its xattrs, its owner and group (_chown_like_cp), and its mode, plus read, write
	and search for its owner when the runner is not root and owns it (rsync's rule; root needs none): a 0555 source
	directory still takes its files. Its times and BSD flags, and the exact mode, wait for the finish, after what goes in it
	(finish_directories), as cp -a and rsync set a directory's last. Returns the finish's record: (src_path, dest_path, the
	mode, atime_ns, mtime_ns, BSD flags or None); the mode is None under -co, which sets no directory metadata. held: the
	door's result for dest_path (_make_dest_directory), which every call acts on (_held_directory): never a link put on
	the way since the door walked it."""
	if CONTENT_ONLY:
		return (src_path, dest_path, None, None, None, None)
	directory = _held_directory(held, dest_path)
	mode = stat.S_IMODE(st.st_mode)
	now = os.stat(directory)
	current = stat.S_IMODE(now.st_mode)
	runner = os.geteuid() if hasattr(os, 'geteuid') else 0
	if runner and now.st_uid == runner and current & stat.S_IRWXU != stat.S_IRWXU:
		# A rerun meets the restrictive mode the finish gave it: opened before its xattrs, or its owner may not set them
		# (EACCES, which copystat's rule ignores) and a source's changed xattr would never come.
		current |= stat.S_IRWXU
		os.chmod(directory, current)
	if _sync_xattrs(src_path, directory):
		now = os.stat(directory)
		current = stat.S_IMODE(now.st_mode)
	if os.name == 'posix':
		if (now.st_uid, now.st_gid) != (st.st_uid, st.st_gid):
			owner_kept, group_kept = _chown_like_cp(directory, st)
			# chmod after chown: no set-user-ID / set-group-ID bit for an owner / group not kept (9.86's rule).
			mode &= ~((0 if owner_kept else stat.S_ISUID) | (0 if group_kept else stat.S_ISGID))
			now = os.stat(directory)
			current = stat.S_IMODE(now.st_mode)
		opened = mode
		if runner and now.st_uid == runner:
			opened |= stat.S_IRWXU
		if current != opened:
			os.chmod(directory, opened)
	return (src_path, dest_path, mode, st.st_atime_ns, st.st_mtime_ns, getattr(st, 'st_flags', None))

def _sync_directory(src_path, dest_paths):
	"""The directory sync of src_path onto dest_paths: each made (_make_dest_directory) and given its start
	(_start_directory). (count, seconds, {link: dests} for a source that is a link, the finish's records); a source that is
	no directory is copied as a file (copy_file), with no records."""
	start_time = time.monotonic()
	dest_paths = _validate_paths(src_path, dest_paths)
	if dest_paths is None:
		return 0, 0, {}, []
	if os.path.islink(src_path):
		return 0, time.monotonic() - start_time, {src_path: dest_paths}, []
	if not os.path.lexists(src_path):
		# A live tree: the directory went away after it was listed. The destination keeps what it has.
		eprint(f'Directory sync error: {src_path} no longer exists')
		return 1, time.monotonic() - start_time, {}, []
	if not os.path.isdir(src_path):
		return copy_file(src_path, dest_paths) + ([],)
	try:
		st = os.stat(src_path)
	except OSError as e:
		# A live tree: the directory went away (or became unreadable) after it was listed.
		eprint(f'Directory sync error: {src_path}: {e}')
		return 1, time.monotonic() - start_time, {}, []
	records = []
	for dest_path in dest_paths:
		# One directory that cannot be created or synced is recorded; the others (and the other dests) go on.
		held = None
		try:
			held = _make_dest_directory(src_path, dest_path)
			if held is not None:
				records.append(_start_directory(src_path, dest_path, st, held))
		except OSError as e:
			eprint(f'Directory sync error: {src_path} to {dest_path}: {e}')
		finally:
			_close_base(held)
	return 1, time.monotonic() - start_time, {}, records

def sync_directory_metadata(src_path, dest_paths):
	"""src_path's directory onto each of dest_paths, its metadata whole now: the sync and its finish at once, for a directory
	whose contents are in place already (a patch's, an image's parents). (count, seconds, {link: dests})."""
	count, seconds, links, records = _sync_directory(src_path, dest_paths)
	for record in records:
		_finish_directory(record)
	return count, seconds, links

def _first_mode(mode):
	"""The mode the finish gives a directory whose final mode is mode: in a move by a non-root runner, one without owner
	search keeps owner read and search until the move's check and removal are done (_finish_late), or every name below it
	would be out of the runner's reach (its fsyncs, the check's looks: 212 or 213 where the copy was whole)."""
	if mode is not None and SYNC_COPIES and hasattr(os, 'geteuid') and os.geteuid() != 0 and not mode & stat.S_IXUSR:
		return mode | stat.S_IRUSR | stat.S_IXUSR
	return mode

def _synced_at_finish(mode):
	"""Whether the finish fsyncs a directory of this final mode itself (a move): one whose first mode (_first_mode) lets a
	non-root owner not read it, which no later open by the runner reaches (_sync_moved_directories passes it over)."""
	first = _first_mode(mode)
	return first is not None and hasattr(os, 'geteuid') and os.geteuid() != 0 and not first & stat.S_IRUSR

def _finish_late(records):
	"""After a non-root move's removal step (ran or skipped): the exact mode of each directory _first_mode left open to its
	owner, deepest first (each parent still searchable while its children are set), each fsynced through its own descriptor
	(_finish_directory, late)."""
	late = [record for record in records if record[2] is not None and _first_mode(record[2]) != record[2]]
	for record in sorted(late, key=lambda record: -os.path.abspath(record[1]).count(os.sep)):
		_finish_directory(record, sync=True, late=True)

def _finish_directory(record, sync=False, late=False):
	"""The rest of a synced directory's metadata, once what goes in it is in (_start_directory's record): its exact mode (in a
	non-root move first _first_mode's, and the exact one in the late pass, late: _finish_late), its times and its BSD flags,
	each set only where the directory differs, through one descriptor opened without following a link (a link the operator
	named with -d is followed, as everywhere). sync (a move): a directory its mode shuts to its owner (_synced_at_finish), or
	any in the late pass, is fsynced through that descriptor, opened while the mode still let its owner in; the others are
	fsynced after the whole finish, once each (an fsync right after each change would commit the journal once per
	directory). A failure is a Directory sync error, or the move's copy failure when only the fsync failed. Below a named
	destination the directory is opened by its name in its parent as the door holds it (_dest_directory, nothing made), the
	named destination itself through the door (its pin), so a directory on the way swapped for a link since is not followed;
	BSD flags (macOS: no descriptor form in Python) still go by name."""
	src_path, dest_path, mode, atime_ns, mtime_ns, flags = record
	if mode is None:
		return  # -co: no directory metadata
	if os.name != 'posix':
		try:
			os.chmod(dest_path, mode)
			os.utime(dest_path, ns=(atime_ns, mtime_ns))
		except OSError as e:
			eprint(f'Directory sync error: {src_path} to {dest_path}: {e}')
		return
	how = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_DIRECTORY', 0)
	path = os.path.abspath(dest_path)
	base = None
	try:
		try:
			if path in NAMED_DESTS:
				# the named destination, followed as named, through the door's open of it
				base, name = _dest_directory(path, make=False), os.curdir
			else:
				how |= getattr(os, 'O_NOFOLLOW', 0)
				base, name = _dest_directory(os.path.dirname(path), make=False), os.path.basename(path)
			if base is None:
				raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dest_path)
			fd = os.open(*_at(base, name), how, **_fd_of(base))
		except _Refused as e:
			_say_refused(e, f'the directory {src_path}')
			return
		except OSError as e:
			eprint(f'Directory sync error: {src_path} to {dest_path}: {_shown(e, os.path.dirname(path), base)}')
			return
	finally:
		_close_base(base)
	try:
		try:
			now = os.fstat(fd)
			wanted = mode if late else _first_mode(mode)
			if stat.S_IMODE(now.st_mode) != wanted:
				os.fchmod(fd, wanted)
			# The read time with the modification time, as cp -a gives both; one that alone differs is left, as a file's
			# (_settle_identical): a read moves it where it is not after the change time, which any set makes now, so a rerun
			# chasing it would set it and move the change time on every run (EL8, relatime: the rerun's own listing did).
			# Within the step the destination keeps (_time_stored_as): vfat keeps 2 s, exFAT 10 ms, HFS+ 1 s, and an exact compare
			# set every directory's times again on each rerun there.
			if not _time_stored_as(now.st_mtime_ns, mtime_ns):
				os.utime(fd if os.utime in os.supports_fd else dest_path, ns=(atime_ns, mtime_ns))
			if flags is not None and hasattr(os, 'chflags') and getattr(now, 'st_flags', flags) != flags:
				os.chflags(dest_path, flags, follow_symlinks=False)
		except OSError as e:
			if late:
				eprint(f"Directory sync error: {src_path} to {dest_path}: its mode, set after the move's removal: {e}")
			else:
				eprint(f'Directory sync error: {src_path} to {dest_path}: {e}')
		if sync and (late or _synced_at_finish(mode)):
			try:
				os.fsync(fd)
			except OSError as e:
				if e.errno not in _NO_FSYNC:
					eprint(f'Copy failed: the directory {dest_path} cannot be written to disk ({_why(e)})')
	finally:
		os.close(fd)

def _finish_directories_job(records, sync):
	"""A finish job in a pool worker: finish each record (_finish_directory). (count, 0, {})."""
	for record in records:
		_finish_directory(record, sync)
	return len(records), 0, {}

# A finish job's directories: the folder sync's folder_per_job default, for the same per-job overhead. A pass of at most one
# job's directories runs in the parent: a pool would run it in one worker, after a pool start (about 23 ms).
_FINISH_PER_JOB = 64

def finish_directories(records, max_workers, sync=False):
	"""Give each directory the copy synced the rest of its metadata (_finish_directory), deepest first, once what goes in it
	(its files, links, subdirectories and -rme's removals) is in: a directory written into after its times were set keeps the
	write's time, and a 0555 one would refuse its files. One pool for the whole pass, each depth done before the next starts,
	so a child's times are set through its parent while the parent's own mode still lets its owner search it; in the parent
	under -s or for one job's worth. sync (a move, M2): a directory its mode shuts to its owner is fsynced through the
	descriptor that set it (_finish_directory); _sync_moved_directories fsyncs the rest."""
	records = [record for record in records if record[2] is not None]
	if not records:
		return
	by_depth = {}
	for record in records:
		by_depth.setdefault(os.path.abspath(record[1]).count(os.sep), []).append(record)
	depths = sorted(by_depth, reverse=True)
	if max_workers <= 1 or len(records) <= _FINISH_PER_JOB:
		for depth in depths:
			for record in by_depth[depth]:
				_check_stop()
				_finish_directory(record, sync)
		return
	with _pool(max(1, min(max_workers, -(-len(records) // _FINISH_PER_JOB)))) as executor:
		for depth in depths:
			level = by_depth[depth]
			futures = {_submit_task(executor, _finish_directories_job, level[i:i + _FINISH_PER_JOB], sync)
				for i in range(0, len(level), _FINISH_PER_JOB)}
			while futures:
				for future in _wait(futures, executor):
					futures.discard(future)
					_task_result(future, (0, 0, {}), 'Copy process crashed')

def _first_missing(path):
	"""The topmost directory on the way to path (made absolute) that is not there yet, path itself when only it is missing;
	None when path is there. A run makes it, and a name made in a directory is on disk only once that directory is fsynced."""
	path, missing = os.path.abspath(path), None
	while not os.path.lexists(path):
		missing, parent = path, os.path.dirname(path)
		if parent == path:
			break
		path = parent
	return missing

def _sync_moved_directories(files, link_dirs, src_paths, dests, dest_image, records, max_workers, made_tops=()):
	"""A move's directory fsyncs, once the finish (finish_directories) is done: each directory the copy synced (records), but
	those the finish fsynced (_synced_at_finish), the directory of every file landing (files: the copy's source paths, one
	directory map per source directory, so a -nds move is covered), those a link was made in (link_dirs), and the parent of
	each source's landing, and for each directory a -d path made (made_tops, _first_missing) its parent and every directory
	from it down to the destination, once each. fsync(2) promises a new name only with its directory's fsync: on ext4 and XFS a file's
	own fsync commits it, elsewhere not; there the first fsync here commits every change before it, so the rest cost little
	(2,040 directories fsynced each right after its own change took 1.4 to 3.1 s on EL7 and EL8, against 9.86's 0.4 to 0.7 s
	for the move). In max_workers threads: an fsync waits on the disk, not the CPU. A directory that is not there (a -nds landing directory in a destination its files did not go to)
	has nothing to write, and another user's that this run may not read (a write-only drop box) is passed over (_fsync_file).
	A failure is a failed copy: Copy failed: the directory DIR cannot be written to disk (why), and the move removes nothing.
	Returns the directories it fsynced (or tried to)."""
	by_source = _by_source(_src_to_dest_map(src_paths, dests, dest_image))
	directories = set(link_dirs or ())
	directories.update(record[1] for record in records)
	finished = {os.path.abspath(record[1]) for record in records if _synced_at_finish(record[2])}
	for parent in {os.path.dirname(path) for path in files}:
		directories.update(_landings(parent, by_source))
	for _, landings in _src_to_dest_map(src_paths, dests, dest_image):
		directories.update(os.path.dirname(os.path.normpath(landing)) for landing in landings)
	for top in made_tops:
		# its parent holds its name; below it, every directory on the way to a destination it holds (a file destination's
		# own directory up)
		directories.add(os.path.dirname(top))
		for dest in dests:
			path = os.path.abspath(dest)
			if not os.path.isdir(path):
				path = os.path.dirname(path)
			while path == top or path.startswith(top.rstrip(os.sep) + os.sep):
				directories.add(path)
				path = os.path.dirname(path)
	directories = {os.path.abspath(directory) for directory in directories}
	# The directories between a destination and each landing directory: under -nds no folder sync made them records, and a
	# name made in one is promised to disk only by its own fsync. With the folder sync each parent is a record already: one
	# look each.
	roots = [os.path.abspath(dest).rstrip(os.sep) + os.sep for dest in dests]
	for directory in sorted(directories):
		parent = os.path.dirname(directory)
		while parent != directory and parent not in directories and any((parent + os.sep).startswith(root) for root in roots):
			directories.add(parent)
			directory, parent = parent, os.path.dirname(parent)
	# One spelling each; a destination named as a link to a directory is fsynced where it leads (_fsync_file passes a link).
	directories -= finished
	directories = sorted({os.path.realpath(directory) if directory in NAMED_DESTS else directory for directory in directories})
	def sync(directory):
		if _STOPPED_PID is not None:
			return  # stopped (the parent raises it): the rest is left, and the move removes nothing
		try:
			_fsync_file(directory)
		except FileNotFoundError:
			pass
		except OSError as e:
			eprint(f'Copy failed: the directory {directory} cannot be written to disk ({_why(e)})')
	if max_workers <= 1 or len(directories) <= 1:
		for directory in directories:
			sync(directory)
	else:
		with concurrent.futures.ThreadPoolExecutor(min(max_workers, len(directories))) as threads:
			for _ in threads.map(sync, directories):
				pass
	return directories

# fcntl(F_FULLFSYNC)'s errors that say the drive cannot be asked (a filesystem without it, such as SMB or FUSE), not that the
# flush failed
_NO_FULL_FLUSH = frozenset(getattr(errno, name) for name in ('ENOTSUP', 'EOPNOTSUPP', 'EINVAL', 'ENOTTY', 'EBADF') if hasattr(errno, name))

def _flush_drives(files, directories, src_paths, dests, dest_image):
	"""A move's last write before its removal, where fcntl has F_FULLFSYNC (macOS): fsync there returns once the data is in
	the drive's own cache, and F_FULLFSYNC "drains the entire queue of the device and acts as a barrier" (xnu's fcntl(2)), so
	one call per destination filesystem the move wrote to (st_dev; one per filesystem is at least one per device) makes every
	earlier fsync there durable. On one landing file there (SQLite's call), or on a directory the move fsynced (directories)
	where no file landed. A drive that cannot be asked (_NO_FULL_FLUSH) gets a Note and the move goes on, as 9.86 never asked;
	a failed flush is a failed copy (Copy failed: the drive under DIR could not flush its cache), and the move removes nothing."""
	try:
		import fcntl
	except ImportError:
		return
	command = getattr(fcntl, 'F_FULLFSYNC', None)
	if command is None:
		return
	by_source = _by_source(_src_to_dest_map(src_paths, dests, dest_image))
	targets = {}
	seen = set()
	# One landing file per source directory: a directory's files are on its filesystem.
	for path in files:
		parent = os.path.dirname(path)
		if parent in seen:
			continue
		seen.add(parent)
		for landing in _landings(path, by_source):
			with contextlib.suppress(OSError):
				st = os.stat(landing)
				if stat.S_ISREG(st.st_mode):
					targets.setdefault(st.st_dev, landing)
	for directory in sorted(directories):
		with contextlib.suppress(OSError):
			targets.setdefault(os.stat(directory).st_dev, directory)
	for path in targets.values():
		_check_stop()
		where = path if os.path.isdir(path) else os.path.dirname(path)
		try:
			try:
				fd = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
			except PermissionError:
				if where == path:
					raise
				# A landing its owner may not read (a copy of another user's file of mode 0044, which _fsync_file opens by a
				# chmod): its directory is on the same filesystem, and the flush is the whole device's.
				fd = os.open(where, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
			try:
				fcntl.fcntl(fd, command)
			finally:
				os.close(fd)
		except OSError as e:
			# PermissionError: neither that landing nor its directory opens for this run (another user's write-only drop box),
			# so the drive cannot be asked through it, as the approved drop-box entry passes over its fsync.
			if e.errno in _NO_FULL_FLUSH or isinstance(e, PermissionError):
				print(f'Note: the drive under {where} could not be asked to flush its cache ({_why(e)}): a power cut soon after '
					'this move can lose files whose sources are removed', file=sys.stderr)
			else:
				eprint(f'Copy failed: the drive under {where} could not flush its cache ({_why(e)})')

def sync_directory_metadata_bulk(src_paths, dest_paths,src_path):
	"""_sync_directory for each of src_paths (below src_path) onto its place below each of dest_paths: (count, seconds,
	{link: dests}, the finish's records)."""
	total_count = 0
	total_time = 0
	symLinks = {}
	records = []
	for src in src_paths:
		source_relative_path = os.path.relpath(src, src_path)
		dests = [_below(dest_path, source_relative_path) for dest_path in dest_paths]
		count, cpTime, rtnSymLinks, done = _sync_directory(src, dests)
		total_count += count
		total_time += cpTime
		symLinks.update(rtnSymLinks)
		records += done
	return total_count, total_time, symLinks, records

def sync_directory_metadata_bulk_batch(jobs):
	"""sync_directory_metadata_bulk for (src_path, folder, dest_paths) jobs."""
	total_count = 0
	total_time = 0
	symLinks = {}
	records = []
	for (src_path,folder,dest_paths) in jobs:
		source_relative_path = os.path.relpath(folder, src_path)
		dests = [_below(dest_path, source_relative_path) for dest_path in dest_paths]
		count, cpTime, rtnSymLinks, done = _sync_directory(folder, dests)
		total_count += count
		total_time += cpTime
		symLinks.update(rtnSymLinks)
		records += done
	return total_count, total_time, symLinks, records

def _folder_syncs(jobs, list_folders, copy_link_or_file, exclude):
	"""The folder syncs of a batch of (src, dests) jobs: (src, folder, dests) for each job's root, then each folder its listing
	has (list_folders(src); the root among them) in path order (_path_order), in job order, one per destination directory:
	the last, as merged sources' files go (a later source's replaces an earlier one's), so a directory several jobs share is
	synced once, from the later source; and the root once. A job whose src is no directory is handed to
	copy_link_or_file(src, dests), as before; one excluded or with no valid destination is left out. One source shares
	nothing but its root, and is not keyed: a key a folder (made absolute) cost 8 to 28 s per million folders on 3.14 to 3.6,
	in this process, before any sync starts (9.87 0.7 to 1.8 s)."""
	listed = []
	for src, dests in jobs:
		dests = _validate_paths(src, dests)
		if dests is None:
			continue
		if exclude and is_excluded(src, exclude):
			continue
		if not os.path.isdir(src):
			copy_link_or_file(src, dests)
			continue
		listed.append((src, dests, _path_order(list_folders(src))))
	if len(listed) == 1:
		src, dests, folders = listed[0]
		return [(src, src, dests)] + [(src, folder, dests) for folder in folders if folder != src]
	syncs, keys = [], []
	for src, dests, folders in listed:
		# Each destination made absolute once; a folder's place below src from its name (a listing spells each folder as src,
		# as given, with the names below it), relpath only where it does not.
		bases = [os.path.abspath(dest) for dest in dests]
		head = src.rstrip(os.sep) + os.sep
		for folder in itertools.chain((src,), folders):
			below = folder[len(head):].strip(os.sep) if folder.startswith(head) else os.path.relpath(folder, src)
			below = '' if below == os.curdir else below
			syncs.append((src, folder, dests))
			keys.append(tuple(os.path.join(base, below) if below else base for base in bases))
	last = {key: index for index, key in enumerate(keys)}
	return [sync for index, sync in enumerate(syncs) if last[keys[index]] == index]

def sync_directories_parallel_batch(jobs, max_workers, verbose=False,folder_per_job=64,exclude=None,parallel_file_listing=False):
	"""Sync the directories of (src, dests) jobs (a pool of max_workers / 4, at least 2): ({link: dests}, the finish's records).
	A destination directory that several jobs share (merged sources), or a job's root and its listing, is synced once, from
	the last job that has it (_folder_syncs), and gives one record. The roots are synced here first, the rest in the pool."""
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.1
	symLinks = {}
	records = []
	def copy_link_or_file(src, dests):
		symLinks.update(copy_file(src, dests)[2])
	syncs = _folder_syncs(jobs, lambda src: get_file_list(src, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)[3],
		copy_link_or_file, exclude)
	# Destinations are unique: which runs first decides nothing.
	for src, folder, dests in syncs:
		if folder == src:
			records += _sync_directory(src, dests)[3]
	allFolderToSync = [sync for sync in syncs if sync[1] != sync[0]]
	if not allFolderToSync:
		return symLinks, records
	print(f"Syncing Dir for {len(allFolderToSync)} folders with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=len(allFolderToSync),total_size=len(allFolderToSync),suppress_all_output=verbose,process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	job_iterator = iter(allFolderToSync)
	futures = {}
	start_time = time.monotonic()
	last_refresh_time = start_time
	max_workers = max(2,int(max_workers / 4))
	folder_per_job = max(1,folder_per_job)
	num_folders_copied_this_job = 0
	
	with _copy_pool(max_workers) as executor:
		while job_iterator or futures:
			while job_iterator and len(futures) <  max_scheduled_jobs and apb.under_rate_limit():
				currentJobs = []
				try:
					for _ in range(folder_per_job):
						currentJobs.append(next(job_iterator))
					future = _submit_task(executor, sync_directory_metadata_bulk_batch, currentJobs)
					futures[future] = currentJobs
				except StopIteration:
					if currentJobs:
						future = _submit_task(executor, sync_directory_metadata_bulk_batch, currentJobs)
						futures[future] = currentJobs
					job_iterator = None

			done = _wait(futures, executor)
			time_spent_this_iter = 0
			for future in done:
				currentJobs = futures.pop(future)
				num_folders_copied_this_job = len(currentJobs)
				cpSize, cpTime, rtnSymLinks, done_records = _task_result(future, (0, 0, {}, []), 'Copy process crashed')
				time_spent_this_iter += cpTime
				apb.update(num_files=num_folders_copied_this_job, cpSize=cpSize, cpTime=cpTime , files_per_job=folder_per_job)
				symLinks.update(rtnSymLinks)
				records += done_records
			apb.scheduled_jobs = len(futures)
			apb.print_progress()
			if done:
				if verbose:
						print(f'\nAverage cptime is {time_spent_this_iter / len(done):0.2f} for {len(done)} jobs with {num_folders_copied_this_job} folders each')
				if job_iterator and num_folders_copied_this_job == folder_per_job and (time_spent_this_iter / len(done) > 5 or time.monotonic() - last_refresh_time > 5):
					folder_per_job //= MAGIC_NUMBER
					folder_per_job = round(folder_per_job)
					if verbose:
						print(f'\nCompletion time is long, changing folders per job to {folder_per_job}')
				elif job_iterator and num_folders_copied_this_job == folder_per_job and time_spent_this_iter / len(done) < 1:
					folder_per_job *= MAGIC_NUMBER
					folder_per_job = round(folder_per_job)
					if verbose:
						print(f'\nCompletion time is short, changing folders per job to {folder_per_job}')
				if folder_per_job < 1:
					folder_per_job = 1
			else:
				if not apb.under_rate_limit():
					if verbose:
						print('\nWe had hit the rate limit and no job is running currently, setting folders per job to 1')
					folder_per_job = 1
				time.sleep(apb.refresh_interval)
			last_refresh_time = time.monotonic()

	endTime = time.monotonic()
	apb.stop()
	_print_operation_summary(apb, endTime - start_time, unit='folder', show_bandwidth=True)
	return symLinks, records

def sync_directories_serial_batch(jobs,exclude=None):
	"""sync_directories_parallel_batch in this process: ({link: dests}, the finish's records), one sync and one record per
	destination directory (_folder_syncs), in job order."""
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	symLinks = {}
	records = []
	def copy_link_or_file(src, dests):
		symLinks.update(copy_file(src, dests)[2])
	syncs = _folder_syncs(jobs, lambda src: get_file_list(src,exclude=exclude,parallel_file_listing=False)[3], copy_link_or_file, exclude)
	if not syncs:
		return symLinks, records
	print(f"Syncing Dir from {len({(src, tuple(dests)) for src, _, dests in syncs})} in single thread")
	apb = Adaptive_Progress_Bar(total_count=len(syncs),total_size=len(syncs),process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	for src, folder, dests in syncs:
		source_relative_path = os.path.relpath(folder, src)
		count, cpTime, _, done = _sync_directory(folder, [_below(dest, source_relative_path) for dest in dests])
		records += done
		apb.update(num_files=1, cpSize=count, cpTime=cpTime , files_per_job=1)
		apb.rate_limit()
	apb.stop()
	return symLinks, records

#%% ---- Compare Files ----
def _compare_file_lists(file_list, file_list2, diff_file_list=None, tar_diff_file_list = False):
	"""Compare two -fl style entry sets (src, dest) and report the rows that turn dest into src."""
	new_entries = _parse_file_list_lines(file_list)
	base_entries = _parse_file_list_lines(file_list2, live_keys=new_entries)
	print('-'*80)
	print(f"Number of files in src: {len(new_entries)}")
	print(f"Number of files in dest: {len(base_entries)}")
	rows = _rows_from_list_entries(new_entries, base_entries)
	report_changes(rows, diff_file_list, tar_diff_file_list)
	return rows

# Preserve the public name while avoiding the boolean option's name in store_file_list.
compare_file_list = _compare_file_lists

#%% ---- Diff Image ----
DIFF_DIR_NAME = '.hpcp_diff'
DIFF_FORMAT_VERSION = 3
# Single-filesystem patches keep the version hpcp 9.65 reads; disk patches (layout disk) are version 2; a patch whose
# states carry the full-hash mark (_FULL_HASH_MARK, FAT files, hpcp 9.69) is version 3, which 9.68 and older refuse.
_SINGLE_FS_FORMAT_VERSION = 1
_DISK_FORMAT_VERSION = 2
_FULL_HASH_FORMAT_VERSION = 3

# A hash over the whole file in a patch hashed by its tail (manifest full_hash 0): 'F' + the hex digest. Files on FAT
# carry it (see compute_changes); digests are lowercase hex, so the mark cannot be one.
_FULL_HASH_MARK = 'F'

def _full_marked(state):
	"""True when state is a file state whose hash carries the full-hash mark (_FULL_HASH_MARK)."""
	if not state.startswith('f:'):
		return False
	fields = state.split(':')
	return len(fields) == 8 and fields[6].startswith(_FULL_HASH_MARK)
CHANGES_HEADER = '#hpcp-changes'
CHANGES_VERSION = 1
ChangeRow = namedtuple('ChangeRow', 'op path old new')
_FIELD_UNESCAPES = {'\\': '\\', 't': '\t', 'n': '\n', 'r': '\r'}

def _escape_field(text):
	"""Escape backslash, tab, newline and carriage return so text fits in one TSV field."""
	return text.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')

def _unescape_field(text):
	"""Exact inverse of _escape_field. Raises ValueError on an unknown or dangling escape."""
	if '\\' not in text:
		return text
	out = []
	index = 0
	while index < len(text):
		char = text[index]
		if char != '\\':
			out.append(char)
			index += 1
			continue
		if index + 1 >= len(text) or text[index + 1] not in _FIELD_UNESCAPES:
			raise ValueError(f'bad escape in {text!r}')
		out.append(_FIELD_UNESCAPES[text[index + 1]])
		index += 2
	return ''.join(out)

def decode_state(state):
	"""Parse an entry state string.

	Returns None for '-' (absent), {'type': '?'} for '?' (unknown), otherwise a dict:
	files f:size:mtime_ns:mode:uid:gid:hash:xattr, directories d:mode:uid:gid:xattr,
	symlinks l:target (target escaped). Raises ValueError on anything else.
	"""
	if state == '-':
		return None
	if state == '?':
		return {'type': '?'}
	kind, sep, rest = state.partition(':')
	if not sep:
		raise ValueError(f'bad entry state {state!r}')
	if kind == 'l':
		return {'type': 'l', 'target': _unescape_field(rest)}
	fields = rest.split(':')
	if kind == 'f' and len(fields) == 7:
		size, mtime_ns, mode, uid, gid, file_hash, xattr = fields
		return {'type': 'f', 'size': int(size), 'mtime_ns': int(mtime_ns), 'mode': int(mode, 8),
			'uid': int(uid), 'gid': int(gid), 'hash': file_hash, 'xattr': xattr}
	if kind == 'd' and len(fields) == 4:
		mode, uid, gid, xattr = fields
		return {'type': 'd', 'mode': int(mode, 8), 'uid': int(uid), 'gid': int(gid), 'xattr': xattr}
	raise ValueError(f'bad entry state {state!r}')

def encode_state(info):
	"""Inverse of decode_state()."""
	if info is None:
		return '-'
	kind = info['type']
	if kind == '?':
		return '?'
	if kind == 'l':
		return 'l:' + _escape_field(info['target'])
	if kind == 'd':
		return f"d:{info['mode']:o}:{info['uid']}:{info['gid']}:{info['xattr']}"
	if kind == 'f':
		return (f"f:{info['size']}:{info['mtime_ns']}:{info['mode']:o}:{info['uid']}:{info['gid']}:"
			f"{info['hash']}:{info['xattr']}")
	raise ValueError(f'bad entry state type {kind!r}')

_TZ_STEP_NS = 15 * 60 * 10**9
_TZ_MAX_NS = 24 * 3600 * 10**9

def _tz_shift_close(a_ns, b_ns):
	"""True when a_ns and b_ns are a whole number of 15-minute steps apart (at most 24 hours), within the usual
	timestamp tolerance: one FAT timestamp read under two timezone settings (sys_tz, tz=, time_offset=)."""
	shift = int(round((a_ns - b_ns) / _TZ_STEP_NS)) * _TZ_STEP_NS
	return 0 < abs(shift) <= _TZ_MAX_NS and _mtimes_agree(a_ns, b_ns + shift)

def states_equal(a, b, compare_hash=True, metadata=True, full_hash=False, tz_tolerant=False):
	"""True when two entry states describe the same entry.

	Directories ignore mtime; files allow the timestamp-precision tolerance; an xattr
	digest of '-' on either side is not compared; '?' never equals anything.
	With compare_hash False neither the content hash nor the xattr digest is compared
	(both come from the same hasher). With metadata False mode, owner and xattrs are not
	compared: on FAT, exFAT and NTFS they come from mount options, not the filesystem.
	With tz_tolerant (a FAT section or scope: FAT stores local times) a whole number of 15-minute steps (at most 24
	hours) between file mtimes is tolerated when the hashes match and cover the whole file: a FAT time read under
	another timezone setting. The hashes cover the whole file when its size is at most HASH_SIZE, when they carry the
	full-hash mark (_FULL_HASH_MARK), or with full_hash; past HASH_SIZE only the file's tail is hashed, and a head
	edit would pass. NTFS stores UTC times and exFAT records the offset: no tolerance there.
	"""
	if a == '?' or b == '?':
		return False
	if a == b:
		return True
	info_a, info_b = decode_state(a), decode_state(b)
	if info_a is None or info_b is None or info_a['type'] != info_b['type']:
		return False
	if info_a['type'] == 'l':
		return info_a['target'] == info_b['target']
	if metadata and (info_a['mode'], info_a['uid'], info_a['gid']) != (info_b['mode'], info_b['uid'], info_b['gid']):
		return False
	if metadata and compare_hash and info_a['xattr'] != '-' and info_b['xattr'] != '-' and info_a['xattr'] != info_b['xattr']:
		return False
	if info_a['type'] == 'd':
		return True
	if info_a['size'] != info_b['size']:
		return False
	if compare_hash and info_a['hash'] != info_b['hash']:
		return False
	if _mtimes_agree(info_a['mtime_ns'], info_b['mtime_ns']):
		return True
	# FAT stores local times: read under another timezone setting the same file shows a whole-step offset. Only
	# FAT comparisons whose hashes confirmed the whole content accept it.
	return (tz_tolerant and compare_hash and info_a['hash'] not in ('', '-')
		and (full_hash or info_a['hash'].startswith(_FULL_HASH_MARK) or info_a['size'] <= HASH_SIZE)
		and _tz_shift_close(info_a['mtime_ns'], info_b['mtime_ns']))

def _hash_algo_name():
	return 'xxh64' if xxhash_available else 'blake2b'

def _new_hasher():
	return xxhash.xxh64() if xxhash_available else hashlib.blake2b()

_PORTABLE_SYSTEM_XATTRS = frozenset(('system.posix_acl_access', 'system.posix_acl_default', 'security.capability'))

def _is_portable_xattr(name):
	"""True for xattrs a copy or a patch image can carry: user.*, POSIX ACLs and file capabilities.

	Filesystem-synthesized names (system.nfs4_acl, lustre.*, trusted.*, security.selinux, ...)
	are not portable and are ignored.
	"""
	return name.startswith('user.') or name in _PORTABLE_SYSTEM_XATTRS

def _xattr_digest(path):
	"""Hex digest of the portable xattrs of path (see _is_portable_xattr; symlinks not followed).

	Returns None when the platform or the filesystem does not support xattrs.
	"""
	if not hasattr(os, 'listxattr'):
		return None
	try:
		names = os.listxattr(path, follow_symlinks=False)
	except OSError as e:
		if e.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
			return None
		raise
	hasher = _new_hasher()
	for name in sorted((name for name in names if _is_portable_xattr(name)), key=os.fsencode):
		try:
			value = os.getxattr(path, name, follow_symlinks=False)
		except OSError as e:
			if e.errno in (errno.ENODATA, errno.EPERM, errno.EACCES, errno.ENOTSUP):
				continue
			raise
		hasher.update(os.fsencode(name) + b'\0' + struct.pack('>Q', len(value)) + value)
	return hasher.hexdigest()

def get_entry_state(path, full_hash=False, read_xattrs=True, mark_full=False, stats=None):
	"""State string of one path (see decode_state). Missing paths, including ones below a
	regular file, are '-'; any other read failure is recorded with eprint and returns '?' (unknown).
	An existing entry that is not a regular file, directory or symlink (FIFO, socket, device)
	is '?' too, without an error: hpcp ignores special files, but they are not absent.
	mark_full: hash a regular file in full and mark the hash (_FULL_HASH_MARK), unless full_hash already hashes every file in full.
	stats: a dict; a regular file's _scan_stat values, from the lstat taken before the file is read, are stored under path.
	"""
	path = path.rstrip(os.sep) or os.sep
	try:
		st = os.lstat(path)
		if stat.S_ISLNK(st.st_mode):
			return encode_state({'type': 'l', 'target': os.readlink(path)})
		xattr = (_xattr_digest(path) if read_xattrs else None) or '-'
		if stat.S_ISDIR(st.st_mode):
			return encode_state({'type': 'd', 'mode': st.st_mode, 'uid': st.st_uid, 'gid': st.st_gid, 'xattr': xattr})
		if stat.S_ISREG(st.st_mode):
			if stats is not None:
				stats[path] = _scan_stat(st)
			# hash_file is lru_cached and would return a stale hash for a same-size rewrite.
			file_hash = hash_file.__wrapped__(path, st.st_size, full_hash or mark_full) or '-'
			if mark_full and not full_hash and file_hash != '-':
				file_hash = _FULL_HASH_MARK + file_hash
			return encode_state({'type': 'f', 'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'mode': st.st_mode,
				'uid': st.st_uid, 'gid': st.st_gid, 'hash': file_hash, 'xattr': xattr})
	except (FileNotFoundError, NotADirectoryError):
		return '-'
	except OSError as e:
		eprint(f'Entry state error: cannot read {path}: {e}')
		return '?'
	return '?'

def _remove_stale_xattrs(src, dest):
	"""Remove portable xattrs (user.*, POSIX ACLs, capabilities) that dest has and src does not.

	shutil.copystat() and cp -a copy and overwrite xattrs but never remove one. dest: a path (never followed), or
	a descriptor (a rewrite in place). Non-portable, filesystem-internal names are never removed. Skipped under
	CONTENT_ONLY; unsupported, forbidden or already-gone removals are ignored.
	"""
	if CONTENT_ONLY or not hasattr(os, 'listxattr'):
		return
	how = {} if isinstance(dest, int) else {'follow_symlinks': False}  # Python refuses follow_symlinks with a descriptor
	try:
		src_names = set(os.listxattr(src, follow_symlinks=False))
		dest_names = os.listxattr(dest, **how)
	except OSError:
		return
	for name in dest_names:
		if name in src_names or not _is_portable_xattr(name):
			continue
		try:
			os.removexattr(dest, name, **how)
		except OSError as e:
			if e.errno not in (errno.ENOTSUP, errno.EPERM, errno.ENODATA, errno.EACCES, errno.EINVAL, errno.ENOENT):
				raise

_OP_ORDER = {'D': 0, 'A': 1, 'M': 2}

def sort_change_rows(rows):
	"""Deterministic order: D, then A, then M; natural path order within each op."""
	return sorted(rows, key=lambda row: (_OP_ORDER[row.op], _natural_key(row.path)))

def _check_key(key):
	"""Reject keys that could escape the patch root."""
	if key == './':
		return
	if not key or key.startswith('/') or '\0' in key:
		raise ValueError(f'unsafe path {key!r}')
	if any(part in ('', '.', '..') for part in key.rstrip('/').split('/')):
		raise ValueError(f'unsafe path {key!r}')

def _open_names(path, mode):
	"""A list of file names (changes.tsv, the manifest) opened as text that holds each name's own bytes (os.fsencode's: the
	filesystem encoding and its error handler), under any locale, so a name read back is the name written and the file it names
	is found. UTF-8 held other bytes than the file's under Latin-1, EUC-JP or KOI8-R, and an apply read them as names that are
	no file's. disk.tsv holds tool output (labels, GPT names, which multiCMD decodes as UTF-8) and stays UTF-8."""
	return open(path, mode, encoding=sys.getfilesystemencoding(), errors=sys.getfilesystemencodeerrors(), newline='\n')

def _wrote_utf8_names(meta):
	"""Whether the hpcp that made a patch (its manifest's hpcp_version) wrote its change list as UTF-8 whatever the locale:
	every one before 9.87. A manifest without a version it can read is taken as written now."""
	try:
		return tuple(int(part) for part in str(meta.get('hpcp_version', '')).split('.')[:2]) < (9, 87)
	except ValueError:
		return False

def _utf8_name_line(line):
	"""A line of a change list hpcp wrote before 9.87 (UTF-8 whatever the locale), given as its bytes read by this locale
	(_open_names): read as 9.86 read it (UTF-8, then spelled by this locale: under Latin-1, the name 9.86 applied), and where
	this locale cannot spell that (LANG unset, where 9.86 stopped with a UnicodeEncodeError), as it stands: the line's own
	bytes."""
	try:
		# through the name's bytes and back: the spelling a listing gives that name here
		return os.fsdecode(os.fsencode(os.fsencode(line).decode('utf-8', 'surrogateescape')))
	except UnicodeError:
		return line

def write_changes(target, rows):
	"""Write rows to target in changes.tsv format (also the -dfl output format)."""
	with _open_names(target, 'w') as f:
		f.write(f'{CHANGES_HEADER}\t{CHANGES_VERSION}\n')
		f.write('#op\tpath\told\tnew\n')
		for row in sort_change_rows(rows):
			f.write(f'{row.op}\t{_escape_field(row.path)}\t{row.old}\t{row.new}\n')

def read_changes(source, utf8_names=False):
	"""Read a changes.tsv file into ChangeRows. Raises RuntimeError('Patch format error: ...'). utf8_names: written by an hpcp
	before 9.87 (_wrote_utf8_names), whose names are read as it read them (_utf8_name_line)."""
	rows = []
	with _open_names(source, 'r') as f:
		header = f.readline().rstrip('\n').split('\t')
		if len(header) != 2 or header[0] != CHANGES_HEADER:
			raise RuntimeError(f'Patch format error: {source} is not an hpcp change list')
		try:
			change_version = int(header[1])
		except ValueError:
			raise RuntimeError(f'Patch format error: {source} has a bad change list version {header[1]!r}')
		if change_version > CHANGES_VERSION:
			raise RuntimeError(f'Patch format error: {source} has change list version {change_version}, newer than supported {CHANGES_VERSION}')
		for line_number, line in enumerate(f, start=2):
			line = line.rstrip('\n')
			if not line or line.startswith('#'):
				continue
			if utf8_names:
				line = _utf8_name_line(line)
			fields = line.split('\t')
			if len(fields) != 4 or fields[0] not in _OP_ORDER:
				raise RuntimeError(f'Patch format error: {source} line {line_number} is malformed')
			try:
				path = _unescape_field(fields[1])
				_check_key(path)
				decode_state(fields[2])
				decode_state(fields[3])
			except ValueError as e:
				raise RuntimeError(f'Patch format error: {source} line {line_number}: {e}')
			rows.append(ChangeRow(fields[0], path, fields[2], fields[3]))
	return rows

_SECTION_NAME_RE = re.compile(r'part[1-9][0-9]*')
_SHA256_RE = re.compile(r'[0-9a-f]{64}')
_PART_NUMBER_RE = re.compile(r'[1-9][0-9]*')
_ASCII_DIGITS_RE = re.compile(r'[0-9]+')

def write_manifest(target, meta):
	"""Write meta as key/value TSV.

	'sections' is a list of (name, payload_dir, changes_file) and 'raws' a list of
	(name, file, old_sha256, new_sha256, size); both are written as repeatable lines.
	"""
	with _open_names(target, 'w') as f:
		for key, value in meta.items():
			if key not in ('sections', 'raws'):
				f.write(f'{key}\t{_escape_field(str(value))}\n')
		for name, payload_dir, changes_file in meta.get('sections', []):
			f.write(f'section\t{_escape_field(name)}\t{_escape_field(payload_dir)}\t{_escape_field(changes_file)}\n')
		for raw in meta.get('raws', []):
			f.write('raw\t' + '\t'.join(_escape_field(str(field)) for field in raw) + '\n')

def _check_raw_entry(fields):
	"""Validate one manifest raw line (name, file, old, new, size); returns it with size as int."""
	name, raw_file, old_sha, new_sha, size = fields
	bare = raw_file[len('raw/'):] if raw_file.startswith('raw/') else ''
	if not (name == 'mbr' or _SECTION_NAME_RE.fullmatch(name)):
		raise ValueError(f'bad raw area name {name!r}')
	if not bare or '/' in bare or bare in ('.', '..'):
		raise ValueError(f'bad raw payload file {raw_file!r}')
	if not (_SHA256_RE.fullmatch(old_sha) and _SHA256_RE.fullmatch(new_sha)):
		raise ValueError('bad raw area hash')
	if not _ASCII_DIGITS_RE.fullmatch(size):
		raise ValueError(f'bad raw area size {size!r}')
	return (name, raw_file, old_sha, new_sha, int(size))

def read_manifest(source):
	"""Read a manifest. Values are strings; 'sections' and 'raws' are lists of tuples.

	Unknown keys are kept and ignored by callers. Raw entries and the disk table are only
	valid in a layout disk manifest (format_version 2).
	"""
	meta = {'sections': [], 'raws': []}
	seen_keys = set()
	with _open_names(source, 'r') as f:
		for line in f:
			line = line.rstrip('\n')
			if not line or line.startswith('#'):
				continue
			try:
				fields = [_unescape_field(field) for field in line.split('\t')]
			except ValueError as e:
				raise RuntimeError(f'Patch format error: {source}: {e}')
			if fields[0] == 'section' and len(fields) == 4:
				name, payload_dir, changes_file = fields[1:]
				try:
					if not name or '/' in name:
						raise ValueError('bad section name')
					if payload_dir != '.':
						_check_key(payload_dir)
					if not changes_file or '/' in changes_file or changes_file in ('.', '..'):
						raise ValueError('bad changes file')
				except ValueError:
					raise RuntimeError(f'Patch format error: {source} has an unsafe section line {line!r}')
				meta['sections'].append(tuple(fields[1:]))
			elif fields[0] == 'raw' and len(fields) == 6:
				try:
					meta['raws'].append(_check_raw_entry(fields[1:]))
				except ValueError as e:
					raise RuntimeError(f'Patch format error: {source} has an unsafe raw line {line!r}: {e}')
			elif fields[0] not in ('section', 'raw') and len(fields) == 2:
				key, value = fields
				if key in ('sections', 'raws'):
					raise RuntimeError(f'Patch format error: {source} uses reserved key {key}')
				if key in seen_keys:
					raise RuntimeError(f'Patch format error: {source} has a duplicate key {key!r}')
				seen_keys.add(key)
				meta[key] = value
			else:
				raise RuntimeError(f'Patch format error: {source} has a malformed line {line!r}')
	if meta.get('format') != 'hpcp-diff':
		raise RuntimeError(f'Patch format error: {source} is not an hpcp-diff manifest')
	try:
		format_version = int(meta.get('format_version', ''))
	except ValueError:
		raise RuntimeError(f'Patch format error: {source} has a bad format_version')
	if format_version > DIFF_FORMAT_VERSION:
		raise RuntimeError(f'Patch format error: {source} uses format version {format_version}, newer than supported {DIFF_FORMAT_VERSION}')
	if meta.get('layout') == 'disk':
		if format_version < 2:
			raise RuntimeError(f'Patch format error: {source} has layout disk but format version {format_version}')
		for name, payload_dir, changes_file in meta['sections']:
			if not _SECTION_NAME_RE.fullmatch(name) or payload_dir != name or changes_file != name + '.changes.tsv':
				raise RuntimeError(f'Patch format error: {source} has a bad disk section {name!r}')
		for kind, names in (('section', [section[0] for section in meta['sections']]), ('raw area', [raw[0] for raw in meta['raws']])):
			if len(set(names)) != len(names):
				raise RuntimeError(f'Patch format error: {source} names a {kind} twice')
		disk_file = meta.get('disk', '')
		if not disk_file or '/' in disk_file or disk_file in ('.', '..'):
			raise RuntimeError(f'Patch format error: {source} has no valid disk table entry')
	elif meta['raws'] or 'disk' in meta:
		raise RuntimeError(f'Patch format error: {source} has raw areas or a disk table outside layout disk')
	return meta

DISK_HEADER = '#hpcp-disk'
DISK_VERSION = 1
DiskRow = namedtuple('DiskRow', 'kind number field old new')
_DISK_FIELDS = ('guid', 'sector_size')
_PART_FIELDS = ('kind', 'type_guid', 'unique_guid', 'name', 'attrs', 'size', 'fs_type', 'fs_uuid', 'fs_label', 'fs_params')

def write_disk_tsv(target, rows):
	"""Write disk.tsv: escaped kind / number / field / old / new rows."""
	with open(target, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		f.write(f'{DISK_HEADER}\t{DISK_VERSION}\n')
		f.write('#kind\tnumber\tfield\told\tnew\n')
		for row in rows:
			f.write('\t'.join(_escape_field(str(value)) for value in row) + '\n')

def read_disk_tsv(source):
	"""Read disk.tsv into DiskRows. Raises RuntimeError('Patch format error: ...')."""
	rows = []
	# Tool output (labels, GPT names) that multiCMD decoded as UTF-8, written as UTF-8: read the same way, not as names.
	with open(source, 'r', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		header = f.readline().rstrip('\n').split('\t')
		if len(header) != 2 or header[0] != DISK_HEADER or not _ASCII_DIGITS_RE.fullmatch(header[1]):
			raise RuntimeError(f'Patch format error: {source} is not an hpcp disk table')
		if int(header[1]) > DISK_VERSION:
			raise RuntimeError(f'Patch format error: {source} has disk table version {header[1]}, newer than supported {DISK_VERSION}')
		for line_number, line in enumerate(f, start=2):
			line = line.rstrip('\n')
			if not line or line.startswith('#'):
				continue
			try:
				fields = [_unescape_field(field) for field in line.split('\t')]
			except ValueError as e:
				raise RuntimeError(f'Patch format error: {source} line {line_number}: {e}')
			if len(fields) != 5:
				raise RuntimeError(f'Patch format error: {source} line {line_number} is malformed')
			kind, number, field = fields[:3]
			ok = ((kind == 'disk' and number == '-' and field in _DISK_FIELDS) or
				(kind == 'part' and _PART_NUMBER_RE.fullmatch(number) and field in _PART_FIELDS))
			if not ok:
				raise RuntimeError(f'Patch format error: {source} line {line_number} has an unknown row {kind} {number} {field}')
			rows.append(DiskRow(*fields))
	return rows

def _logically_excluded(rel, logical_root, exclude):
	"""True when rel (relative to a listed root) or one of its ancestors matches exclude
	once placed under logical_root."""
	if rel in ('', '.'):
		return False
	parts = rel.split(os.sep)
	for depth in range(1, len(parts) + 1):
		if is_excluded(os.path.join(logical_root, *parts[:depth]), exclude):
			return True
	return False

def _listing_error_path(error):
	"""The path a recorded 'Listing error' names, or None (a crashed listing task names none)."""
	prefix, overlong = 'Listing error: ', ': path longer than 4096 characters, not listed'
	if not error.startswith(prefix):
		return None
	body = error[len(prefix):]
	if body.endswith(overlong):
		return body[:-len(overlong)]
	index = body.rfind(': [Errno ')
	return body[:index] if index > 0 else None

def _listing_error_excluded(error, root, logical_root, exclude):
	"""True when error is a 'Listing error' for a path below root that exclude leaves out (logically, see
	_logically_excluded)."""
	path = _listing_error_path(error)
	if path is None or not (path == root or path.startswith(root.rstrip(os.sep) + os.sep)):
		return False
	return _logically_excluded(os.path.relpath(path, root), logical_root, exclude)

def build_entry_map(root, prefix='', parents=None, exclude=None, logical_root=None,
		max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""Map patch keys to absolute paths for one side of a diff.

	Keys are prefix + the path relative to root ('/'-separated, directories end with '/');
	root itself is keyed prefix, or './' when prefix is empty. parents adds extra
	{key: directory} entries (store-layout ancestors) when they exist. With logical_root,
	exclude is matched against logical_root + relative path instead of the physical path,
	so a base tree honours patterns written for the new tree. A root that is a symlink to a
	directory is listed from its target (keys stay relative, exclude is still matched
	against the given path). A missing root, or a symlink to anything else, yields only
	the existing parents.
	"""
	entries = {}
	for key, path in (parents or {}).items():
		if os.path.isdir(path) and not os.path.islink(path):
			entries[key] = path
	root = os.path.abspath(root)
	if os.path.islink(root) and os.path.isdir(root):
		if logical_root is None:
			logical_root = root
		root = os.path.realpath(root)
	if not os.path.isdir(root) or os.path.islink(root):
		return entries
	post_filter = bool(exclude) and logical_root is not None
	error_start = len(ERRORS)
	files, links, _, folders = get_file_list(root, exclude=None if post_filter else exclude, max_workers=max_workers,
		drop_cache=True, parallel_file_listing=parallel_file_listing)
	# The listing goes on past a directory it cannot read; a map without that directory's entries would show them as
	# deleted (or, on the base side, as added), so a patch or diff list made from it would be wrong.
	failures = [error for error in ERRORS[error_start:] if _is_failure(error)]
	if post_filter:
		# Listed without the excludes (matched below against logical paths): what they leave out does not count.
		failures = [error for error in failures if not _listing_error_excluded(error, root, logical_root, exclude)]
	if failures:
		raise RuntimeError(f'Listing error: {root} could not be listed completely ({len(failures)} error(s)); '
			'not comparing a partial tree: fix the paths above (or exclude them with -e) and re-run')
	entries[prefix or './'] = root
	for paths, is_dir in ((files, False), (links, False), (folders, True)):
		for path in paths:
			rel = os.path.relpath(path, root)
			if rel == '.':
				continue
			if post_filter and _logically_excluded(rel, logical_root, exclude):
				continue
			entries[prefix + rel.replace(os.sep, '/') + ('/' if is_dir else '')] = path
	return entries

def _source_entry_map(src, exclude=None, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""build_entry_map for one source of a compare, keyed where the copy lands it. A source from a list is keyed by its
	landing, with the directories that path implies; a file or a link in a list is an entry of its own."""
	if not isinstance(src, _ListedSource):
		return build_entry_map(src, _trim_prefix(src), exclude=exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing)
	landing = src.landing.replace(os.sep, '/')
	entries = {}
	for source, parent in _implied_parents(src):
		if os.path.isdir(source) and not os.path.islink(source):
			entries[parent.replace(os.sep, '/') + '/'] = os.path.abspath(source)
	if os.path.isdir(src) and not os.path.islink(src):
		entries.update(build_entry_map(src, landing + '/' if landing else '', exclude=exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
	elif not (exclude and is_excluded(src, exclude)):
		entries[landing] = os.path.abspath(src)
	return entries

def _store_layout_parents(new_dir, store_root):
	"""Ancestors of new_dir except '/', as ({key: live directory}, {key: directory inside the store})."""
	new_parents, base_parents = {}, {}
	parent = os.path.dirname(os.path.abspath(new_dir))
	while parent != os.path.dirname(parent):
		rel = os.path.relpath(parent, os.sep)
		key = rel.replace(os.sep, '/') + '/'
		new_parents[key] = parent
		base_parents[key] = os.path.join(store_root, rel)
		parent = os.path.dirname(parent)
	return new_parents, base_parents

def build_store_maps(new_dir, store_root, exclude=None, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""Entry maps for a directory against a -di image store mounted at store_root ('store' layout).

	-di keeps files at their absolute source path, so keys are relative to '/' and the base
	side is limited to new_dir's own subtree (plus its parent directories) inside the store.
	"""
	new_dir = os.path.abspath(new_dir)
	rel = os.path.relpath(new_dir, os.sep)
	prefix = '' if rel == '.' else rel.replace(os.sep, '/') + '/'
	new_parents, base_parents = _store_layout_parents(new_dir, store_root)
	new_map = build_entry_map(new_dir, prefix, new_parents, exclude, None, max_workers, parallel_file_listing)
	base_root = store_root if rel == '.' else os.path.join(store_root, rel)
	base_map = build_entry_map(base_root, prefix, base_parents, exclude, new_dir, max_workers, parallel_file_listing)
	return new_map, base_map

def build_root_maps(new_root, base_root, exclude=None, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""Entry maps for two trees compared root to root ('root' layout)."""
	new_map = build_entry_map(new_root, '', None, exclude, None, max_workers, parallel_file_listing)
	base_map = build_entry_map(base_root, '', None, exclude, os.path.abspath(new_root), max_workers, parallel_file_listing)
	return new_map, base_map

def build_rooted_maps(new_root, base_root, exclude=None, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False):
	"""Entry maps for two mounted filesystems compared root to root ('root' layout: -si images, -dd -diff sections),
	exclude matched against '/' + the path inside each filesystem (_rooted_excludes). Each side is listed with its own
	rooted patterns, so an excluded directory is skipped, not walked, on both sides."""
	exclude = frozenset(exclude or ())
	new_map = build_entry_map(new_root, '', None, _rooted_excludes(exclude, new_root), None, max_workers, parallel_file_listing)
	base_map = build_entry_map(base_root, '', None, _rooted_excludes(exclude, base_root), None, max_workers, parallel_file_listing)
	return new_map, base_map

def _states_chunk(paths, full_hash, read_xattrs, hash_size, marked=frozenset(), stat_paths=None):
	"""Worker body for _states_for_paths: states, the errors recorded while computing them, and the _scan_stat values
	of the regular files among stat_paths (None: no stats; {} comes back).

	hash_size is the caller's HASH_SIZE; a worker that did not fork from the caller
	(spawn / forkserver) would otherwise hash with the module default.
	"""
	global HASH_SIZE
	HASH_SIZE = hash_size
	error_start = len(ERRORS)
	stats = {} if stat_paths is not None else None
	states = {path: get_entry_state(path, full_hash, read_xattrs, path in marked, stats if stats is not None and path in stat_paths else None)
		for path in paths}
	errors = ERRORS[error_start:]
	del ERRORS[error_start:]
	return states, errors, stats or {}

def _states_for_paths(paths, full_hash=False, read_xattrs=True, max_workers=1, marked=frozenset(), stats=None, stat_paths=()):
	"""{path: state} for every path; parallel chunks on a ProcessPoolExecutor for large sets. Paths in marked are
	hashed in full with the mark (see get_entry_state). stats: a dict filled with the _scan_stat values of the regular
	files among stat_paths (only those: the dict grows with them, not with the number of paths)."""
	paths = list(paths)
	if max_workers <= 1 or len(paths) < 256:
		return {path: get_entry_state(path, full_hash, read_xattrs, path in marked, stats if path in stat_paths else None)
			for path in paths}
	chunk_size = max(64, len(paths) // (max_workers * 4) + 1)
	states = {}
	with _pool(max_workers) as executor:
		futures = set()
		for index in range(0, len(paths), chunk_size):
			chunk = paths[index:index + chunk_size]
			futures.add(_submit_task(executor, _states_chunk, chunk, full_hash, read_xattrs, HASH_SIZE,
				frozenset(path for path in chunk if path in marked),
				frozenset(path for path in chunk if path in stat_paths) if stats is not None else None))
		while futures:
			for future in _wait(futures, executor):
				futures.discard(future)
				result, _, crash = _settled(future)
				if crash is not None:
					raise RuntimeError(f"General Exception: the entries' states could not be read: {crash}")
				chunk_states, chunk_errors, chunk_stats = result
				states.update(chunk_states)
				ERRORS.extend(chunk_errors)
				if stats is not None:
					stats.update(chunk_stats)
	return states

def compute_changes(new_map, base_map, full_hash=False, max_workers=multiprocessing.cpu_count(), read_xattrs=True, metadata=True,
		fat_paths=frozenset(), scan=None):
	"""Rows that turn base_map's tree into new_map's tree, sorted D / A / M.

	With metadata False (see states_equal) a mode, owner or xattr difference alone is no change.
	fat_paths: paths (values of either map) on FAT. A key whose path on either side is one is compared like a FAT
	section: by content only and with the timezone tolerance, its files on both sides hashed in full and marked
	(_FULL_HASH_MARK; unless full_hash already hashes every file in full), so the tolerance holds past HASH_SIZE.
	scan: a dict, filled for the payload re-check (_rows_to_recheck): 'started_ns', the wall clock before any state is read, and
	'stats', {key: _scan_stat values} of the A / M file rows' new-side files, from the lstat taken before each was read.
	"""
	keys = set(new_map) | set(base_map)
	fat_keys = {key for key in keys if new_map.get(key) in fat_paths or base_map.get(key) in fat_paths} if fat_paths else set()
	marked = set() if full_hash else {tree[key] for tree in (new_map, base_map) for key in fat_keys if key in tree}
	stats = None
	if scan is not None:
		scan['started_ns'] = _wall_clock_ns()
		stats = {}
	states = _states_for_paths(set(new_map.values()) | set(base_map.values()), full_hash=full_hash,
		read_xattrs=read_xattrs and metadata, max_workers=max_workers, marked=marked, stats=stats,
		stat_paths=set(new_map.values()) if scan is not None else ())
	rows = []
	for key in keys:
		new_state = states[new_map[key]] if key in new_map else '-'
		old_state = states[base_map[key]] if key in base_map else '-'
		# A read failure (already recorded) or a special file: a guessed row could delete or add real files.
		if new_state == '?' or old_state == '?':
			continue
		if new_state == '-' and old_state == '-':
			continue
		if old_state == '-':
			rows.append(ChangeRow('A', key, '-', new_state))
		elif new_state == '-':
			rows.append(ChangeRow('D', key, old_state, '-'))
		elif old_state[0] != new_state[0]:
			rows.append(ChangeRow('D', key, old_state, '-'))
			rows.append(ChangeRow('A', key, '-', new_state))
		elif not states_equal(old_state, new_state, metadata=metadata and key not in fat_keys, full_hash=full_hash,
				tz_tolerant=key in fat_keys):
			rows.append(ChangeRow('M', key, old_state, new_state))
	if scan is not None:
		# Only the rows the payload check can look at: the dict does not grow with the unchanged files.
		scan['stats'] = {row.path: stats[new_map[row.path]] for row in rows
			if row.op in ('A', 'M') and row.new.startswith('f:') and new_map[row.path] in stats}
	return sort_change_rows(rows)

def _printable_path(path):
	"""path for a terminal: undecodable bytes (surrogate-escaped) shown as backslash escapes."""
	return path.encode('utf-8', 'surrogateescape').decode('utf-8', 'backslashreplace')

def report_changes(rows, diff_file_list=None, tar_diff_file_list=False):
	"""Write diff_file_list (if given), then print the A / M / D lists and counts.

	The file is changes.tsv format, or with tar_diff_file_list a plain newline list of
	added / modified paths for `tar -T`; the root './' and modified directories are left
	out there, since tar would archive their whole subtree. The file is written first so a
	name the terminal cannot show never costs the list.
	"""
	if diff_file_list:
		if tar_diff_file_list:
			with _open_names(diff_file_list, 'w') as f:
				for row in rows:
					if (row.op == 'A' and row.path != './') or (row.op == 'M' and not row.path.endswith('/')):
						f.write(row.path + '\n')
		else:
			write_changes(diff_file_list, rows)
	labels = (('A', 'Added (in src, not in dest)'), ('M', 'Modified'), ('D', 'Deleted (in dest, not in src)'))
	print('-'*80)
	for op, label in labels:
		print(f"{label}:")
		for row in rows:
			if row.op == op:
				print(_printable_path(row.path))
		print('-'*80)
	for op, label in labels:
		print(f"{label} count: {sum(1 for row in rows if row.op == op)}")
	if diff_file_list:
		print(f"Diff file list written to {diff_file_list}")

def _hash_hex_re():
	return re.compile(r'[0-9a-f]{16}' if xxhash_available else r'[0-9a-f]{128}')

# What -fl writes after a file's ':': an xxh64 or a blake2b digest (or nothing, with -hs 0).
_LIST_DIGEST_RE = re.compile(r'[0-9a-f]{16}|[0-9a-f]{128}')

def _colon_lines_name_live_links(lines, live_keys):
	"""For a list whose every line has a ':' but no digest and no bare 'path:' (a -hs 0 list of links only, or plain
	names that hold a ':'): True when more lines read as 'live path:target' than as a live path themselves."""
	as_links = sum(1 for line in lines if line not in live_keys
		and any(char == ':' and line[:index] in live_keys for index, char in enumerate(line)))
	return as_links > sum(1 for line in lines if line in live_keys)

def _parse_file_list_lines(lines, live_keys=()):
	"""Parse -fl style lines into {key: value}.

	value is the hash for files, the link target for symlinks, None when the list holds
	neither. A line ending in '/' is a directory. A line ending in ':<digest>' is a file, whatever
	algorithm made the digest, and in a hashed list so is 'path:' (made with -hs 0); only a digest of
	this system's algorithm is kept as the value, another one or none is None (not compared).
	A list is hashed when a line has a digest, or when a line is a bare 'path:' and every file and link
	line has a ':' (a -hs 0 list; an unhashed list has plain paths, which may all hold a ':' when the
	source root's name does); without either, the live tree decides (_colon_lines_name_live_links). In a hashed list the remaining lines are symlinks
	'path:target': the longest live key followed by ':' is the path, else the text before the first ':'.
	"""
	hex_re = _hash_hex_re()
	lines = [line for line in lines if line]
	entry_lines = [line for line in lines if not line.endswith('/')]
	live_keys = set(live_keys)
	hashed = (any(':' in line and _LIST_DIGEST_RE.fullmatch(line.rpartition(':')[2]) for line in entry_lines)
		or (bool(entry_lines) and all(':' in line for line in entry_lines)
			and (any(line.endswith(':') for line in entry_lines) or _colon_lines_name_live_links(entry_lines, live_keys))))
	entries = {}
	for line in lines:
		if line.endswith('/'):
			entries[line] = None
			continue
		path, sep, tail = line.rpartition(':')
		if sep and (_LIST_DIGEST_RE.fullmatch(tail) or (hashed and not tail)):
			entries[path] = tail if hex_re.fullmatch(tail) else None
			continue
		if hashed and ':' in line:
			key = None
			for index, char in enumerate(line):
				if char == ':' and line[:index] in live_keys:
					key = line[:index]
			if key is None:
				key = line.partition(':')[0]
			entries[key] = line[len(key) + 1:]
			continue
		entries[line] = None
	return entries

def _value_from_state(state):
	"""What a -fl list records for an entry: file hash, link target, or None."""
	info = decode_state(state)
	if info is None or info['type'] in ('d', '?'):
		return None
	if info['type'] == 'l':
		return info['target']
	return None if info['hash'] == '-' else info['hash']

def _rows_from_list_entries(new_entries, base_entries, new_states=None):
	"""Rows from -fl style entries; base states are unknown ('?')."""
	new_states = new_states or {}
	rows = []
	for key in set(new_entries) | set(base_entries):
		if key not in base_entries:
			rows.append(ChangeRow('A', key, '-', new_states.get(key, '?')))
		elif key not in new_entries:
			rows.append(ChangeRow('D', key, '?', '-'))
		else:
			new_value, base_value = new_entries[key], base_entries[key]
			if new_value is not None and base_value is not None and new_value != base_value:
				rows.append(ChangeRow('M', key, '?', new_states.get(key, '?')))
	return sort_change_rows(rows)

def _trim_prefix(path):
	"""Key prefix matching trim_paths(): a path without a trailing separator contributes its folder name."""
	if path.endswith(os.sep):
		return ''
	name = os.path.basename(os.path.normpath(path))
	return '' if name in ('', '.') else name + '/'

def _is_reserved_key(key):
	return key in (DIFF_DIR_NAME, DIFF_DIR_NAME + '/') or key.startswith(DIFF_DIR_NAME + '/')

def _key_path(root, key):
	"""Absolute path of key under root ('./' is root itself)."""
	parts = [part for part in key.rstrip('/').split('/') if part not in ('', '.')]
	return os.path.normpath(os.path.join(root, *parts)) if parts else os.path.normpath(root)

def _key_depth(key):
	return 0 if key == './' else key.rstrip('/').count('/') + 1

def _parent_keys(key):
	"""Directory keys above key, shallowest first: 'a/b/c' -> ['a/', 'a/b/']."""
	parts = key.rstrip('/').split('/')
	return ['/'.join(parts[:index]) + '/' for index in range(1, len(parts))]

def _remove_entry(path):
	"""Remove a file, symlink or whole directory tree; a missing path is fine."""
	try:
		if os.path.isdir(path) and not os.path.islink(path):
			_rmtree(path)
		else:
			os.unlink(path)
	except FileNotFoundError:
		pass

def _place_symlink(target, dest):
	if os.path.lexists(dest):
		_remove_entry(dest)
	_makedirs(os.path.dirname(dest), exist_ok=True)
	os.symlink(target, dest)

def _payload_bytes(rows):
	return sum(decode_state(row.new)['size'] for row in rows if row.op in ('A', 'M') and row.new.startswith('f:'))

def make_patch_meta(rows, new_name, base_name, layout, full_hash=False):
	"""Manifest fields for a new patch; 'volume' is filled in per volume by write_patch_metadata()."""
	return {
		'format': 'hpcp-diff',
		'format_version': (_FULL_HASH_FORMAT_VERSION if any(_full_marked(row.old) or _full_marked(row.new) for row in rows)
			else _SINGLE_FS_FORMAT_VERSION),
		'patch_id': uuid.uuid4().hex,
		'volume': '1/1',
		'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
		'hpcp_version': version,
		'new': new_name,
		'base': base_name,
		'hash_algo': _hash_algo_name(),
		'hash_size': HASH_SIZE,
		'full_hash': int(bool(full_hash)),
		'xattrs': int(hasattr(os, 'listxattr')),
		'layout': layout,
		'added': sum(1 for row in rows if row.op == 'A'),
		'modified': sum(1 for row in rows if row.op == 'M'),
		'deleted': sum(1 for row in rows if row.op == 'D'),
		'payload_bytes': _payload_bytes(rows),
		'sections': [('root', '.', 'changes.tsv')],
	}

def _payload_allocation(rows):
	"""Bytes a patch volume needs for rows: change list, allocation (not apparent size) and an inode each."""
	size = 2 * sum(len(row.path) + len(row.old) + len(row.new) + 8 for row in rows)
	for row in rows:
		if row.op in ('A', 'M'):
			# Allocation, not apparent size, plus the inode each entry needs (512 bytes on xfs).
			size += 512
			if row.new.startswith('f:'):
				size += _align_up(max(decode_state(row.new)['size'], 1))
			else:
				size += _IMAGE_BLOCK
	return size

def _claim_volume_names(output, count):
	"""Create every planned volume file exclusively (O_EXCL, O_NOFOLLOW); on a clash remove what this call created."""
	claimed = []
	for name in [_split_dest_image_name(output, i) for i in range(count)]:
		try:
			os.close(os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
		except FileExistsError:
			for done in claimed:
				try:
					os.unlink(done)
				except OSError:
					pass
			raise RuntimeError(f'Diff image error: output {name} already exists')
		claimed.append(name)
	return claimed

def _source_resized(new_base, row):
	"""True when row's source under new_base is gone or no longer a regular file of the recorded size."""
	try:
		info = os.lstat(_key_path(new_base, row.path))
	except OSError:
		return True
	return not stat.S_ISREG(info.st_mode) or info.st_size != decode_state(row.new)['size']

# Filesystems where a write to a file moves its ctime, which nothing can set back: an unchanged ctime then shows the
# content did not change, outside the racy window (_racy_window_ns). The one exception is a store through a shared
# writable mapping to a page that is already dirty: it leaves ctime unchanged until writeback cleans the page. The
# window therefore covers twice the writeback horizon, and tmpfs (never written back) is not trusted. Everything else
# (FAT, exFAT, NTFS, FUSE, network filesystems) is hashed again. The window is a heuristic: under writeback backlog a
# page can stay dirty longer than twice the horizon, and a store into it then goes unseen.
_TRUSTED_CTIME_FS = frozenset(('ext2', 'ext3', 'ext4', 'xfs', 'btrfs', 'f2fs'))

def _read_vm_centisecs(name):
	"""int of /proc/sys/vm/<name>, None when it cannot be read or parsed."""
	try:
		with open('/proc/sys/vm/' + name) as f:
			return int(f.read().strip())
	except (OSError, ValueError):
		return None

def _racy_window_ns():
	"""How close to the scan's start (or after it) a scan-time ctime must be for its copy to be hashed again, in ns:
	twice the writeback horizon (vm.dirty_expire_centisecs + vm.dirty_writeback_centisecs), at least 2 s; 120 s when
	either cannot be read (no /proc); None when periodic writeback is off (dirty_writeback_centisecs 0): no ctime is
	trusted then."""
	expire, writeback = _read_vm_centisecs('dirty_expire_centisecs'), _read_vm_centisecs('dirty_writeback_centisecs')
	if expire is None or writeback is None or expire < 0 or writeback < 0:
		return 120 * 1000000000
	if writeback == 0:
		return None
	return max(2 * 1000000000, 2 * (expire + writeback) * 10000000)

def _wall_clock_ns():
	"""Wall-clock time in ns (time.time_ns needs Python 3.7)."""
	return int(time.time() * 1000000000)

def _scan_stat(st):
	"""The lstat values the payload re-check compares: device, inode, size, mtime and ctime (ns)."""
	return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

def _holding_mount(points, real_directory):
	"""The deepest mount holding real_directory: its own entry in points ({mount point without a trailing separator
	('' for '/'): mountinfo entry}) or the nearest parent's, looked up directory by directory up to '/', so the cost
	grows with the depth of the path, not with the number of mounts. None when no mount holds it."""
	key = real_directory.rstrip(os.sep)
	while True:
		if key in points:
			return points[key]
		if not key:
			return None
		key = key.rpartition(os.sep)[0]

def _rows_to_recheck(rows, new_base, scan):
	"""The rows (A / M file rows with a payload copy) whose copy _check_payload hashes again.

	On a filesystem whose superblock is read-only (mountinfo super options; it cannot change) only a row whose lstat now
	differs from the scan's (a row with no scan stat is left). On any other, a row whose source may have changed since
	the scan read it: its filesystem is not in _TRUSTED_CTIME_FS, the scan (compute_changes' scan dict) has no stat for
	it, its scan-time ctime lies within _racy_window_ns of the scan's start or later (any ctime when the window is
	None), or its lstat now differs from the scan's. A row's filesystem is the visible mount whose mount point is its source file itself (a bind of one file), else the deepest visible
	mount holding the real path of its source's parent directory; not the one holding the spelled path: a key can
	be spelled through a symlink (store layout)."""
	entries = _mountinfo_entries()
	hidden = _hidden_mounts(entries)
	scan = scan or {}
	stats, started = scan.get('stats', {}), scan.get('started_ns')
	window = _racy_window_ns()
	points = {entry['mount_point'].rstrip(os.sep): entry for index, entry in enumerate(entries) if index not in hidden}
	holders = {}
	def holder(source):
		# A mount on the file itself (a bind of one file) holds it; else the deepest mount holding its directory.
		directory, name = os.path.split(source)
		if directory not in holders:
			real_directory = os.path.realpath(directory).rstrip(os.sep)
			found = _holding_mount(points, real_directory)
			holders[directory] = (found, real_directory)
		found, real_directory = holders[directory]
		return points.get(real_directory + os.sep + name, found)
	recheck = []
	for row in rows:
		source = _key_path(new_base, row.path)
		entry = holder(source)
		before = stats.get(row.path)
		if entry is not None and 'ro' in entry.get('super_options', ()):
			# The window and the filesystem type do not matter on a read-only superblock; a moved stat still does (a mount
			# replaced at the same path, a directory swapped for a symlink into it, a remount read-only after a write).
			if before is None:
				continue
		elif (entry is None or entry['fstype'] not in _TRUSTED_CTIME_FS or before is None or started is None or window is None
				or before[4] >= started - window):
			recheck.append(row)
			continue
		try:
			now = _scan_stat(os.lstat(source))
		except OSError:
			now = None
		if now != before:
			recheck.append(row)
	return recheck

def _check_payload(rows, volume_roots, full_hash=False, max_workers=1, new_base=None, scan=None):
	"""Raise RuntimeError('Diff image error: ...') unless every A / M file row's payload is on a volume
	and still matches its row; every failing row is checked and reported in one error.

	copy_file_list_parallel records its failures in ERRORS and goes on (it skips full destinations too): a row
	with no copy of the recorded size on any volume is an incomplete payload, unless its source
	under new_base no longer has that size either (a live file that grew or went away). A live
	source can also change between the change scan and the copy at the same size; that copy
	would fail every apply's verify, so its hash (computed as the row's was, with HASH_SIZE and
	full_hash) and mtime are compared with the row. With full_hash (-fh) every copy is hashed again; otherwise
	only when its source may have changed since the scan read it (_rows_to_recheck; every copy when new_base is None).
	"""
	incomplete, changed, copies = [], [], {}
	for row in rows:
		if row.op not in ('A', 'M') or not row.new.startswith('f:'):
			continue
		# The copy an apply would take.
		copy = _payload_source(volume_roots, row)
		if copy is not None:
			copies[row] = copy
		elif new_base is not None and _source_resized(new_base, row):
			changed.append(row)
		else:
			incomplete.append(row.path)
	# With -fh every copy is hashed again, whatever its source (as up to 9.70): a copy that came out wrong with the
	# right size is caught here, before the patch exists, not by every apply's verify after it was written.
	# Copies of files on FAT (the full-hash mark) are always hashed again: they were hashed in full, and are the ones a changed source hurts most.
	recheck = set(copies) if full_hash or new_base is None else (
		{row for row in copies if _full_marked(row.new)} | set(_rows_to_recheck([row for row in copies if not _full_marked(row.new)], new_base, scan)))
	copies = {row: copy for row, copy in copies.items() if row in recheck}
	marked = {copy for row, copy in copies.items() if _full_marked(row.new)}
	states = _states_for_paths(set(copies.values()), full_hash=full_hash, read_xattrs=False, max_workers=max_workers, marked=marked) if copies else {}
	for row, copy in copies.items():
		# Hash and mtime, as an apply's verify compares them (mode and owner are not checked here).
		if not states_equal(states[copy], row.new, compare_hash=decode_state(row.new)['hash'] != '-', metadata=False, full_hash=full_hash):
			changed.append(row)
	problems = []
	if incomplete:
		problems.append(f'payload incomplete for {len(incomplete)} file(s), e.g. {incomplete[0]}')
	if changed:
		paths = [row.path for row in sort_change_rows(changed)]
		shown = ', '.join(paths[:10]) + (f' and {len(paths) - 10} more' if len(paths) > 10 else '')
		problems.append(f'{len(paths)} file(s) changed while the patch was built (e.g. {shown}); '
			'exclude them (-e) or generate from an offline copy')
	if problems:
		raise RuntimeError('Diff image error: ' + '; '.join(problems) + '; patch metadata not written')

def write_patch_metadata(volume_roots, rows, meta):
	"""Write .hpcp_diff/changes.tsv on every volume first, then each volume's manifest (volume k/n).

	All change lists are written before any manifest, so a manifest on any volume implies
	every volume already has its change list.
	"""
	total = len(volume_roots)
	for volume_root in volume_roots:
		diff_dir = os.path.join(volume_root, DIFF_DIR_NAME)
		_makedirs(diff_dir, exist_ok=True)
		write_changes(os.path.join(diff_dir, 'changes.tsv'), rows)
	for index, volume_root in enumerate(volume_roots, start=1):
		diff_dir = os.path.join(volume_root, DIFF_DIR_NAME)
		volume_meta = dict(meta)
		volume_meta['volume'] = f'{index}/{total}'
		write_manifest(os.path.join(diff_dir, 'manifest'), volume_meta)

def _build_payload(rows, new_base, volume_roots, max_workers=1, verbose=False, estimated_size=0, full_hash=False, scan=None):
	"""Copy the A / M payload of rows from new_base onto the volume roots and check it landed
	unchanged (_check_payload; full_hash as the rows were hashed).

	Every map key is relpath(path, new_base), so new_base is '/' for store layout and the
	new tree's root for root layout. scan: compute_changes' scan dict for these rows (see _rows_to_recheck).
	"""
	for row in rows:
		if _is_reserved_key(row.path):
			raise RuntimeError(f'Diff image error: {row.path} uses the reserved name {DIFF_DIR_NAME}')
	payload = [row for row in rows if row.op in ('A', 'M')]
	dir_keys = set()
	for row in payload:
		dir_keys.update(_parent_keys(row.path))
		if row.new.startswith('d:'):
			dir_keys.add(row.path)
	for key in sorted(dir_keys, key=_key_depth):
		for volume_root in volume_roots:
			_makedirs(_key_path(volume_root, key), exist_ok=True)
	files = [_key_path(new_base, row.path) for row in payload if row.new.startswith('f:')]
	if files:
		copy_file_list_parallel(files, [], new_base, list(volume_roots), max(1, max_workers), verbose=verbose,
			estimated_size=estimated_size)
	for row in payload:
		if row.new.startswith('l:'):
			_place_symlink(decode_state(row.new)['target'], _key_path(volume_roots[0], row.path))
	# Directories last and deepest first, so later writes cannot disturb their mtimes.
	for key in sorted(dir_keys, key=_key_depth, reverse=True):
		source = _key_path(new_base, key)
		if os.path.islink(source) and os.path.isdir(source):
			# A symlinked root is listed from its target (build_entry_map); take its metadata there too.
			source = os.path.realpath(source)
		sync_directory_metadata(source, [_key_path(volume_root, key) for volume_root in volume_roots])
	_check_payload(payload, volume_roots, full_hash, max_workers, new_base, scan)

def build_patch_tree(rows, new_base, volume_roots, meta, max_workers=1, verbose=False, scan=None):
	"""Copy the A / M payload of rows from new_base onto the volume roots, then write .hpcp_diff/.
	scan: compute_changes' scan dict for rows (see _rows_to_recheck)."""
	_build_payload(rows, new_base, volume_roots, max_workers, verbose, meta.get('payload_bytes', 0),
		str(meta.get('full_hash', 0)) == '1', scan=scan)
	write_patch_metadata(volume_roots, rows, meta)

PatchCheck = namedtuple('PatchCheck', 'row action found reason')

def _fold_name(name):
	"""name as a case-insensitive filesystem (FAT, exFAT, HFS+) compares it: each character upper-cased when that
	gives exactly one character, otherwise kept, so 'ß' stays apart from 'SS' as FAT keeps them."""
	return ''.join(char.upper() if len(char.upper()) == 1 else char for char in name)

def _under_removed(key, removed, target_root='', case_insensitive=False):
	"""True when key or one of its ancestors is a name some D row will remove.

	removed maps each removed name (folded when case_insensitive) to [(exact name, target path)]. A name that
	matches only after folding counts when the filesystem agrees: the key's own prefix is missing on the
	target, or is the very entry the D row removes (a fold may equate names the filesystem keeps apart).
	"""
	parts = key.rstrip('/').split('/')
	for index in range(1, len(parts) + 1):
		prefix = '/'.join(parts[:index])
		for name, path in removed.get(_fold_name(prefix) if case_insensitive else prefix, ()):
			if name == prefix:
				return True
			own = _key_path(target_root, prefix)
			if not os.path.lexists(own) or _same_entry(own, [path]):
				return True
	return False

def _spelled_on_disk(target_root, name, listings):
	"""name (a key without its trailing '/') as the target spells it; None when a component is missing. A component
	the directory lists exactly is used as is; otherwise it is matched with _fold_name. listings caches
	{directory: (set of names, {folded name: name})}."""
	directory, spelled = os.path.normpath(target_root), []
	for part in name.split('/'):
		if directory not in listings:
			try:
				names = os.listdir(directory)
			except OSError:
				names = []
			listings[directory] = (set(names), {_fold_name(entry): entry for entry in names})
		names, folded = listings[directory]
		entry = part if part in names else folded.get(_fold_name(part))
		if entry is None:
			return None
		spelled.append(entry)
		directory = os.path.join(directory, entry)
	return '/'.join(spelled)

def _unlisted_children(key, path, deleted_keys, case_insensitive=False):
	"""Children of directory path whose keys are not in deleted_keys; None when path cannot be listed.

	case_insensitive: deleted_keys are folded (_fold_name) and so is each child's key (a target that looks
	names up case-insensitively may spell an entry as it did before a case-only rename).
	"""
	prefix = '' if key == './' else key
	try:
		names = os.listdir(path)
	except OSError:
		return None
	unlisted = []
	for name in names:
		child = os.path.join(path, name)
		child_key = prefix + name + ('/' if os.path.isdir(child) and not os.path.islink(child) else '')
		if (_fold_name(child_key) if case_insensitive else child_key) not in deleted_keys:
			unlisted.append(child_key)
	return unlisted

def check_patch_target(rows, target_root, compare_hash=True, full_hash=False, max_workers=1, metadata=True, case_insensitive=False, tz_tolerant=False):
	"""Classify every row against target_root as 'apply', 'skip' or 'conflict'.

	D: target == old -> apply; absent, or already the replacing A row's state -> skip.
	A: absent -> apply; == new -> skip. M: == old -> apply; == new -> skip.
	Anything else is a conflict, as is a D directory holding entries the patch does not
	delete or one that cannot be listed.
	A and M rows are judged as the target will be after the applicable D rows: a name a D
	row removes (with or without the trailing '/'), and everything below it, counts as absent.
	metadata False compares content only (see states_equal). case_insensitive (a FAT, exFAT or HFS+
	target, which looks names up case-insensitively) compares names with _fold_name: a D directory's
	entries, the name a D row removes, and the A row that replaces a D row. A case-only rename (D on one
	spelling, A on another) is then a delete and a write; once applied (the target already spells the A
	row's name) both rows are skipped.
	Rows whose states carry the full-hash mark (_FULL_HASH_MARK) are compared with the target's file hashed in full and marked
	(only with compare_hash: no hash is compared otherwise).
	tz_tolerant: a FAT section (see states_equal).
	"""
	fold = _fold_name if case_insensitive else (lambda name: name)
	paths = {row.path: _key_path(target_root, row.path) for row in rows}
	# Without hashes to compare (another algorithm) nothing is hashed in full.
	marked = {paths[row.path] for row in rows if _full_marked(row.old) or _full_marked(row.new)} if compare_hash else set()
	states = _states_for_paths(set(paths.values()), full_hash=full_hash, read_xattrs=metadata, max_workers=max_workers, marked=marked)
	same = functools.partial(states_equal, compare_hash=compare_hash, metadata=metadata, full_hash=full_hash, tz_tolerant=tz_tolerant)
	added_by_name = {fold(row.path.rstrip('/')): row for row in rows if row.op == 'A'}
	deleted_keys = {fold(row.path) for row in rows if row.op == 'D'}
	listings = {}
	checks = {}
	removed_names = {}
	for row in rows:
		if row.op != 'D':
			continue
		name = row.path.rstrip('/')
		found = states[paths[row.path]]
		replacement = added_by_name.get(fold(name))
		if replacement is not None and replacement.path.rstrip('/') != name:
			# A case-only rename: the target's own spelling shows whether it already happened. Only the
			# components the two rows spell differently count: the node may spell the others its own way.
			spelled = _spelled_on_disk(target_root, name, listings)
			added = replacement.path.rstrip('/')
			if spelled is None or any(on_disk != new for on_disk, old, new in zip(spelled.split('/'), name.split('/'), added.split('/'))
					if old != new):
				replacement = None
		if found == '-':
			checks[row] = PatchCheck(row, 'skip', found, 'already absent')
			continue
		if replacement is not None and same(found, replacement.new):
			checks[row] = PatchCheck(row, 'skip', found, 'already replaced')
			continue
		removed_names.setdefault(fold(name), []).append((name, paths[row.path]))
		if not same(found, row.old):
			checks[row] = PatchCheck(row, 'conflict', found, 'changed on target')
			continue
		unlisted = _unlisted_children(row.path, paths[row.path], deleted_keys, case_insensitive) if found.startswith('d:') else []
		if unlisted is None:
			checks[row] = PatchCheck(row, 'conflict', found, 'cannot list directory')
		elif unlisted:
			checks[row] = PatchCheck(row, 'conflict', found, 'directory holds entries the patch does not delete')
		else:
			checks[row] = PatchCheck(row, 'apply', found, '')
	for row in rows:
		if row.op == 'D':
			continue
		found = '-' if _under_removed(row.path, removed_names, target_root, case_insensitive) else states[paths[row.path]]
		if row.op == 'A':
			if found == '-':
				checks[row] = PatchCheck(row, 'apply', found, '')
			elif same(found, row.new):
				checks[row] = PatchCheck(row, 'skip', found, 'already in new state')
			else:
				checks[row] = PatchCheck(row, 'conflict', found, 'already exists on target')
		elif same(found, row.old):
			checks[row] = PatchCheck(row, 'apply', found, '')
		elif same(found, row.new):
			checks[row] = PatchCheck(row, 'skip', found, 'already in new state')
		else:
			checks[row] = PatchCheck(row, 'conflict', found, 'missing on target' if found == '-' else 'changed on target')
	return [checks[row] for row in rows]

def _validate_patch_rows(rows):
	"""Reject rows whose op, states and key shape disagree, or that nest under a symlink the patch creates."""
	symlinks = set()
	for row in rows:
		old, new = row.old, row.new
		where = f'{row.op} {row.path!r}'
		if _is_reserved_key(row.path):
			raise RuntimeError(f'Patch format error: row {where} uses the reserved name {DIFF_DIR_NAME}')
		unset = ('-', '?')
		if row.op == 'A':
			ok = old == '-' and new not in unset
		elif row.op == 'D':
			ok = new == '-' and old not in unset
		elif row.op == 'M':
			ok = old not in unset and new not in unset and old[0] == new[0]
		else:
			ok = False
		if not ok:
			raise RuntimeError(f'Patch format error: row {where} has inconsistent states ({old} -> {new})')
		if row.path == './' and not (row.op == 'M' and old.startswith('d:') and new.startswith('d:')):
			raise RuntimeError(f'Patch format error: row {where} may only modify the root directory metadata')
		for state in (old, new):
			if state != '-' and row.path.endswith('/') != state.startswith('d:'):
				raise RuntimeError(f'Patch format error: row {where} has a state of the wrong type for its key')
		if row.op in ('A', 'M') and new.startswith('l:'):
			symlinks.add(row.path)
	for row in rows:
		if row.op == 'D':
			continue
		for parent in _parent_keys(row.path):
			if parent.rstrip('/') in symlinks:
				raise RuntimeError(f'Patch format error: row {row.op} {row.path!r} lies under symlink {parent.rstrip("/")!r} created by the same patch')

def _load_patch(volume_roots, volume_names=None):
	"""Read and cross-check every volume's manifest.

	Returns (meta of volume 1, [(section name, payload dir, rows)], volume roots ordered by
	volume number). volume_names (image paths, parallel to volume_roots) only improve error messages.
	"""
	names = list(volume_names) if volume_names else [str(root) for root in volume_roots]
	metas = []
	for volume_root, name in zip(volume_roots, names):
		manifest = os.path.join(volume_root, DIFF_DIR_NAME, 'manifest')
		if not os.path.isfile(manifest):
			raise RuntimeError(f'Incomplete patch: {name} has no {DIFF_DIR_NAME}/manifest')
		metas.append(read_manifest(manifest))
	for name, meta in zip(names, metas):
		if not meta.get('patch_id'):
			raise RuntimeError(f'Incomplete patch: {name} has no patch_id')
	patch_ids = sorted({meta['patch_id'] for meta in metas})
	if len(patch_ids) != 1:
		raise RuntimeError(f'Incomplete patch: volumes belong to different patches ({", ".join(patch_ids)})')
	by_number = {}
	total = None
	for volume_root, name, meta in zip(volume_roots, names, metas):
		number, _, count = str(meta.get('volume', '')).partition('/')
		try:
			number, count = int(number), int(count)
			if count < 1 or not 1 <= number <= count:
				raise ValueError
		except ValueError:
			raise RuntimeError(f'Patch format error: {name} has a bad volume field {meta.get("volume")!r}')
		if total is None:
			total = count
		elif count != total:
			raise RuntimeError(f'Incomplete patch: volumes disagree on the volume count ({total} vs {count})')
		by_number[number] = (volume_root, name, meta)
	missing = [number for number in range(1, total + 1) if number not in by_number]
	if missing:
		if 1 in by_number:
			expected = ', '.join(_split_dest_image_name(by_number[1][1], number - 1) for number in missing)
		else:
			expected = ', '.join(f'volume {number}' for number in missing)
		raise RuntimeError(f'Incomplete patch: missing volume(s) {expected} of {total}')
	ordered = [by_number[number] for number in range(1, total + 1)]
	meta = ordered[0][2]
	sections = []
	for name, payload_dir, changes_file in meta['sections']:
		rows = read_changes(os.path.join(ordered[0][0], DIFF_DIR_NAME, changes_file), _wrote_utf8_names(meta))
		_validate_patch_rows(rows)
		sections.append((name, payload_dir, rows))
	return meta, sections, [entry[0] for entry in ordered]

def load_patch_volumes(volume_roots, volume_names=None):
	"""Single-filesystem patch: (meta, rows, volume roots ordered by volume number)."""
	meta, sections, ordered = _load_patch(volume_roots, volume_names)
	if meta.get('layout') == 'disk' or meta['sections'] != [('root', '.', 'changes.tsv')]:
		raise RuntimeError(f"Patch format error: unsupported sections {meta['sections']}; disk patches apply to a whole disk (-siff PATCH -d DISK)")
	return meta, sections[0][2], ordered

def _section_volume_roots(volume_roots, payload_dir):
	"""Where a section's payload lives on each volume.

	A volume whose section directory is a symlink (or sits behind one) is left out, so the
	payload containment root can never move outside the patch volume.
	"""
	if payload_dir == '.':
		return list(volume_roots)
	return [os.path.join(volume_root, payload_dir) for volume_root in volume_roots
		if os.path.realpath(os.path.join(volume_root, payload_dir)) == os.path.join(os.path.realpath(volume_root), payload_dir)]

def _conflict_lines(checks, prefix=''):
	"""(detail, short) text per conflicting check, as printed in the conflict list and the forcing warnings."""
	lines = []
	for check in checks:
		expected = '-' if check.row.op == 'A' else check.row.old
		lines.append((f"{prefix}{check.row.op} {check.row.path!r}: {check.reason} (expected {expected}, found {check.found})",
			f"{prefix}{check.row.op} {check.row.path}: {check.reason}"))
	return lines

def _resolve_conflict_lines(lines, force):
	"""Print conflicts, then prompt (default abort), abort without a tty, or force with warnings."""
	print(f"{len(lines)} path(s) on the target differ from the state this patch expects:")
	for detail, _ in lines:
		# The states can hold link targets with undecodable bytes; stdout may be strict UTF-8.
		print(f"  {_printable_path(detail)}")
	if not force:
		if not sys.stdin.isatty():
			raise RuntimeError(f'Patch conflict: {len(lines)} conflicting path(s) and stdin is not a tty; nothing applied (use -fa to force)')
		answer = multiCMD.input_with_timeout_and_countdown(60, 'Apply anyway? [y/N]')
		if (answer or '').strip().lower() not in ('y', 'yes'):
			raise RuntimeError(f'Patch conflict: {len(lines)} conflicting path(s); nothing applied')
	for _, short in lines:
		print(f"Warning: forcing {short}", file=sys.stderr)

def _inside_root(root, path):
	"""True when path's parent directory resolves inside root (no symlinked escape)."""
	root_real = os.path.realpath(root)
	parent_real = os.path.realpath(os.path.dirname(path))
	return parent_real == root_real or parent_real.startswith(root_real.rstrip(os.sep) + os.sep)

def _payload_source(volume_roots, row):
	"""Path of row's payload in the first volume holding a safe copy, else None.

	Safe means inside its volume root (no symlinked parent), a regular non-symlink file
	of the recorded size.
	"""
	size = decode_state(row.new)['size']
	for volume_root in volume_roots:
		source = _key_path(volume_root, row.path)
		try:
			info = os.lstat(source)
		except OSError:
			continue
		if stat.S_ISREG(info.st_mode) and info.st_size == size and _inside_root(volume_root, source):
			return source
	return None

_TEMP_PREFIX = '.hpcp_tmp_'

def _temp_sibling(dest):
	"""A fresh temporary name in dest's directory (no basename in it, so NAME_MAX cannot overflow)."""
	return os.path.join(os.path.dirname(dest), _TEMP_PREFIX + uuid.uuid4().hex[:16])

def _unsafe_parent(target_root, dest):
	"""Why dest must not be written right now, or None.

	Writes follow the rule deletions follow (_inside_root): dest's parent must be a directory
	whose real path lies inside target_root. A symlinked ancestor that resolves inside the
	target is followed; one that leads outside it is refused.
	"""
	if not _inside_root(target_root, dest):
		return f'resolves outside {target_root}'
	parent = os.path.dirname(dest)
	if not os.path.isdir(parent):
		return f'{parent} is not a directory'
	return None

def _names_removed_first(rows, target_root):
	"""Names (keys without the trailing '/') that rows remove before anything is written below them:
	a D row's, or a directory row's over a non-directory (a symlink, say) on the target."""
	replaced = {row.path.rstrip('/') for row in rows if row.op == 'D'}
	for row in rows:
		if row.op != 'D' and row.new.startswith('d:') and row.path != './':
			path = _key_path(target_root, row.path)
			if os.path.lexists(path) and (os.path.islink(path) or not os.path.isdir(path)):
				replaced.add(row.path.rstrip('/'))
	return replaced

def _rows_escaping_target(rows, target_root):
	"""Rows whose parent exists now (a dangling symlink counts) and resolves outside target_root.

	D rows are judged against the target as it is. An A or M row below a name the patch
	first removes (a D row, or a directory row over a non-directory) is left to the
	write-time check, like a row whose parent does not exist yet: by then that name is a
	new directory, not the symlink it is now.
	"""
	replaced = _names_removed_first(rows, target_root)
	escaping = []
	for row in rows:
		if row.path == './':
			continue
		if row.op != 'D' and any(key.rstrip('/') in replaced for key in _parent_keys(row.path)):
			continue
		path = _key_path(target_root, row.path)
		# lexists: a dangling symlink parent that points outside is refused too.
		if os.path.lexists(os.path.dirname(path)) and not _inside_root(target_root, path):
			escaping.append(row)
	return escaping

def _discard_temp(path, base=None):
	"""Remove a temporary entry that was not renamed into place; a missing one is fine. base: the door's result for its
	directory (_cp_beside), which it is removed from."""
	try:
		if base is None:
			os.unlink(path)
		else:
			os.unlink(*_at(base, os.path.basename(path)), **_fd_of(base))
	except FileNotFoundError:
		pass
	except OSError as e:
		print(f"Warning: cannot remove temporary file {path}: {e}", file=sys.stderr)

def _fat_ioctl_numbers(machine=None):
	"""(FAT_IOCTL_GET_ATTRIBUTES, FAT_IOCTL_SET_ATTRIBUTES) for machine (os.uname()'s by default, read at import:
	platform.machine() runs `uname -p` there on Python 3.6 to 3.8, and the import runs no subprocess):
	_IOR('r', 0x10, __u32) and _IOW('r', 0x11, __u32). alpha, mips, powerpc and sparc put _IOC_READ (2) and
	_IOC_WRITE (4) at bit 29, and parisc swaps the generic read and write bits: on all of them the two numbers come
	out the other way round."""
	if machine is None:
		machine = os.uname().machine if hasattr(os, 'uname') else platform.machine()
	machine = machine.lower()
	if machine.startswith(('alpha', 'mips', 'ppc', 'powerpc', 'sparc', 'parisc')):
		return 0x40047210, 0x80047211
	return 0x80047210, 0x40047211

_FAT_IOCTL_GET_ATTRIBUTES, _FAT_IOCTL_SET_ATTRIBUTES = _fat_ioctl_numbers()
# Read-only, hidden, system and archive: the attribute bits a regular file carries.
_FAT_FILE_ATTRIBUTES = 0x27

def _fat_attributes(path):
	"""The FAT attribute bits of an existing file on vfat / msdos, else None (missing file, not a regular
	file, another filesystem, or a platform without fcntl)."""
	try:
		import fcntl
		if not stat.S_ISREG(os.lstat(path).st_mode):
			return None
		fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
	except (ImportError, OSError):
		return None
	try:
		return struct.unpack('I', fcntl.ioctl(fd, _FAT_IOCTL_GET_ATTRIBUTES, b'\0' * 4))[0] & _FAT_FILE_ATTRIBUTES
	except OSError:
		return None
	finally:
		os.close(fd)

def _set_fat_attributes(path, attributes, key):
	"""Set path's FAT attribute bits; a failure is a warning, never an error."""
	try:
		import fcntl
		if not stat.S_ISREG(os.lstat(path).st_mode):
			return
		fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
		try:
			fcntl.ioctl(fd, _FAT_IOCTL_SET_ATTRIBUTES, struct.pack('I', attributes))
		finally:
			os.close(fd)
	except (ImportError, OSError) as e:
		print(f'Warning: cannot keep the FAT attributes of {key}: {e}', file=sys.stderr)

def _finish_replace(job, target_root, wait, metadata=True):
	"""Wait for one temporary copy, check it, and rename it over its destination.

	The temporary file is created in the live directory, so it inherits that directory's
	default ACL as an access ACL, which cp -af keeps when the payload has none. Portable
	xattrs the payload lacks are removed from it before the rename (cp -af already set the
	payload's mode, and with it the ACL mask, so the group bits stay right). On a content-only target
	(metadata False) an existing FAT file's attribute bits are kept. Any failure is reported as 'Copy failed' and leaves the destination untouched; the
	temporary file never outlives this call. The temporary file is written to disk here, in the parent's loop, one at a time:
	measured against the workers fsyncing their own files, which came out slower on EL7 (ext4 and XFS, two workers).
	"""
	row, source, dest, tmp = job
	try:
		wait()
		size = decode_state(row.new)['size']
		info = os.lstat(tmp)
		if not stat.S_ISREG(info.st_mode) or info.st_size != size:
			raise RuntimeError(f'copied {info.st_size} bytes, expected {size}')
		if metadata:
			_remove_stale_xattrs(source, tmp)
		reason = _unsafe_parent(target_root, dest)
		if reason:
			raise RuntimeError(reason)
		# A FAT file keeps its read-only / hidden / system bits: the replacement would otherwise get the defaults.
		attributes = None if metadata else _fat_attributes(dest)
		_fsync_file(tmp)  # renamed unsynced, a power cut can leave dest empty (_fsync_file)
		os.replace(tmp, dest)
		if attributes is not None and attributes != _fat_attributes(dest):
			_set_fat_attributes(dest, attributes, row.path)
	except Exception as e:
		eprint(f'Copy failed: {row.path}: {e}')
	finally:
		_discard_temp(tmp)

def _copy_content(src_path, dest_paths, full_hash=False, verbose=False):
	"""copy_file for a target without POSIX metadata (FAT, exFAT, NTFS): the bytes and the
	times only; mode and owner come from the target's mount options. Raises on failure."""
	st = os.stat(src_path)
	for dest in dest_paths:
		shutil.copyfile(src_path, dest)
		os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns))

def _replace_files(files, target_root, max_workers=1, verbose=False, full_hash=False, metadata=True):
	"""Copy each (row, source, dest) to a temporary sibling of dest, then rename it over dest.

	copy_file (cp -af) preserves mode, owner, times and xattrs; _finish_replace strips what
	the temporary file inherited from its directory (a default ACL). With metadata False only
	the content and times are copied (_copy_content). Copies run on a process
	pool, or serially when max_workers <= 1. A running process keeps the old inode, an
	interruption leaves dest fully old or fully new, and hard links to the old file are broken.
	"""
	copy = copy_file if metadata else _copy_content
	jobs = []
	for row, source, dest in files:
		reason = _unsafe_parent(target_root, dest)
		if reason:
			eprint(f'Copy failed: {row.path}: {reason}; skipped')
			continue
		jobs.append((row, source, dest, _temp_sibling(dest)))
	if not jobs:
		return
	print(f"Copying {len(jobs)} file(s) with {max(1, max_workers)} worker(s)")
	if max_workers <= 1 or len(jobs) == 1:
		for job in jobs:
			row, source, dest, tmp = job
			_finish_replace(job, target_root,
				functools.partial(_call_raising_recorded_failures, copy, source, [tmp], full_hash, verbose), metadata)
		return
	futures = {}
	try:
		with _pool(max_workers) as executor:
			for row, source, dest, tmp in jobs:
				futures[_submit_task(executor, copy, source, [tmp], full_hash, verbose)] = (row, source, dest, tmp)
			while futures:
				for future in _wait(futures, executor):
					_finish_replace(futures.pop(future), target_root, functools.partial(_raise_task_failures, future), metadata)
	finally:
		# A stop or a failure: the pool's exit has waited for its workers, so a temporary copy one of them made and this loop
		# did not rename is discarded here (as _finish_replace discards its own).
		for row, source, dest, tmp in futures.values():
			_discard_temp(tmp)

def _replace_with_symlink(row, link_target, dest, target_root):
	"""Create the symlink at a temporary sibling of dest and rename it over dest."""
	tmp = None
	try:
		reason = _unsafe_parent(target_root, dest)
		if reason:
			raise RuntimeError(reason)
		tmp = _temp_sibling(dest)
		os.symlink(link_target, tmp)
		reason = _unsafe_parent(target_root, dest)
		if reason:
			raise RuntimeError(reason)
		os.replace(tmp, dest)
	except (OSError, RuntimeError) as e:
		eprint(f'Failed to create symbolic link: {row.path}: {e}')
	finally:
		if tmp:
			_discard_temp(tmp)

def _holds_type(path, state):
	"""True when path exists (its last component not followed) with the entry type of state: 'd:', 'f:' or 'l:'."""
	try:
		mode = os.lstat(path).st_mode
	except OSError:
		return False
	return {'d': stat.S_ISDIR, 'f': stat.S_ISREG, 'l': stat.S_ISLNK}[state[0]](mode)

def _entry_identity(path):
	"""(st_dev, st_ino, file type) of path, its last component not followed; None when it is missing."""
	try:
		info = os.lstat(path)
	except OSError:
		return None
	return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))

def _identities(target_root, rows):
	"""{row: _entry_identity of its path} for the A / M rows among rows."""
	return {row: _entry_identity(_key_path(target_root, row.path)) for row in rows if row.op != 'D'}

def _entry_snapshots(target_root, rows, metadata=True, max_workers=1):
	"""{row: (_entry_identity of its path, its state for a directory row, else None)} for the A / M rows among rows.
	A directory is written in place, keeping its inode, so only its state (read without hashing; xattrs when
	metadata) shows a change."""
	directories = {row: _key_path(target_root, row.path) for row in rows if row.op != 'D' and row.new.startswith('d:')}
	states = _states_for_paths(set(directories.values()), read_xattrs=metadata, max_workers=max_workers) if directories else {}
	return {row: (identity, states.get(directories.get(row))) for row, identity in _identities(target_root, rows).items()}

def _same_entry(path, others):
	"""True when path and one of others are one entry: same st_dev and st_ino (last components not followed)."""
	try:
		info = os.lstat(path)
	except OSError:
		return False
	for other in others:
		if other == path:
			continue
		try:
			found = os.lstat(other)
		except OSError:
			continue
		if (found.st_dev, found.st_ino) == (info.st_dev, info.st_ino):
			return True
	return False

def _mounts_below(path, mount_points):
	"""The mount points at or below path's real path, relative to it ('.' for path itself), sorted."""
	real = os.path.realpath(path)
	inside = real.rstrip(os.sep) + os.sep
	return sorted(os.path.relpath(point, real) for point in mount_points if point == real or point.startswith(inside))

# Whether the platform has the descriptor-relative calls the delete walk needs (as shutil.rmtree checks).
_DIR_FD_WALK = (os.open in os.supports_dir_fd and os.listdir in os.supports_fd and os.stat in os.supports_dir_fd
	and os.stat in os.supports_follow_symlinks and os.unlink in os.supports_dir_fd and os.rmdir in os.supports_dir_fd and hasattr(os, 'O_NOFOLLOW'))

def _descend(fd, parts, flags):
	"""Open each of parts in turn, the first below descriptor fd, each with flags (O_NOFOLLOW | O_DIRECTORY): a component
	that is a symlink or no directory raises OSError (ELOOP, ENOTDIR). Takes fd over (closed); returns the last one."""
	try:
		for part in parts:
			next_fd = os.open(part, flags, dir_fd=fd)
			os.close(fd)
			fd = next_fd
	except BaseException:
		os.close(fd)
		raise
	return fd

def _open_below(root_real, rel, flags):
	"""A descriptor of root_real/rel reached one component at a time from a descriptor of root_real (_descend)."""
	fd = os.open(root_real, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0))
	return _descend(fd, [] if rel in ('', '.') else rel.split(os.sep), flags)

def _remove_tree_around_mounts(path, mount_points=None, target_root=None):
	"""Remove the directory tree at path as shutil.rmtree would, without entering a mount.

	An entry that is a mount point in mountinfo (any entry: bind mounts and hidden mounts too), or a directory whose
	st_dev differs from path's (a mount made after mountinfo was read, or a btrfs subvolume), is left in place with
	everything below it, and so are the directories above it; path itself too when it is a mount point in mountinfo or
	its st_dev differs from its parent's. Symlinks are unlinked, never
	followed, and that holds for path itself: only its parent is resolved, so a path that is a symlink (or not a
	directory) removes nothing and returns an error. The walk is iterative and keeps going after an error. Returns (the entries left in place as mount
	points, relative to path; the errors met, as text).

	The walk is _remove_tree_fd's, by directory descriptors as shutil.rmtree does: every directory is opened with
	O_NOFOLLOW relative to its parent (path itself included) and checked against the lstat taken before descending, and
	entries are read and removed relative to the descriptor, so a directory swapped for a symlink during the walk is
	refused, never followed (this runs as root on trees other users may write to). The apply runs only on Linux, which
	has every descriptor function this needs.

	target_root: the tree the removal must stay in (the apply's target; '/' when None). The top's parent is resolved once and must lie inside it; it is then reached from a descriptor of target_root one component at a time with O_NOFOLLOW, and the top is examined, entered and removed relative to its parent's descriptor. A directory above the top swapped for a symlink after the apply checked the path therefore refuses the removal instead of acting outside the target (this runs as root on trees other users may write to).
	"""
	if mount_points is None:
		mount_points = {entry['mount_point'] for entry in _mountinfo_entries()}
	# The parent is resolved once, the last component never: a symlink there must not be followed to its target.
	stripped = path.rstrip(os.sep) or os.sep
	name = os.path.basename(stripped)
	if name in ('', os.curdir, os.pardir):
		return [], ['.: refusing to remove . or .. or a filesystem root']
	root_real = os.path.realpath(target_root or os.sep)
	parent_real = os.path.realpath(os.path.dirname(stripped))
	top = os.path.join(parent_real, name)
	if parent_real != root_real and not parent_real.startswith(root_real.rstrip(os.sep) + os.sep):
		return [], [f'..: resolves outside {root_real}; not removed']
	try:
		parent_fd = _open_below(root_real, os.path.relpath(parent_real, root_real), _BELOW_FLAGS)
	except FileNotFoundError:
		return [], []
	except OSError as e:
		return [], [f'..: {e.strerror} (a directory above it changed while it was being removed); not removed']
	try:
		try:
			top_info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
		except FileNotFoundError:
			return [], []
		except OSError as e:
			return [], [f'.: {e.strerror}']
		if not stat.S_ISDIR(top_info.st_mode):
			return [], ['.: not a directory (a symlink?)']
		top_dev = top_info.st_dev
		if top in mount_points:
			return ['.'], []
		# A mount made on path itself after mountinfo was read (or a btrfs subvolume) is on another device than its parent.
		if os.fstat(parent_fd).st_dev != top_dev:
			return ['.'], []
		_, left, failures = _remove_tree_fd(top, top_info, remove_files=True, parent_fd=parent_fd, mounts=(mount_points, top_dev))
		return (sorted(os.path.relpath(path, top) for path in left),
			[f'{os.path.relpath(path, top)}: {e.strerror}' for path, e in failures])
	finally:
		os.close(parent_fd)

def _apply_rows(rows, volume_roots, target_root, max_workers=1, verbose=False, full_hash=False, forced_rows=(), metadata=True,
		skipped=()):
	"""Carry out checked rows: deletions, then directories, files, symlinks, directory metadata.

	Containment of every target path is re-checked immediately before each write, and the
	target root itself is never removed. A directory is removed with rmdir, so an entry that
	appeared after the check survives (and verify reports the row); only a row in
	forced_rows (a conflict the user forced) may remove a whole tree, and that removal never enters a mount (see
	_remove_tree_around_mounts): a tree holding a mount point is left in place and fails verify. Files and symlinks are
	built at a temporary sibling and renamed over the target (see _replace_files), so an
	existing file or symlink is never removed before its replacement exists. Only an entry
	a rename cannot replace is removed first: a directory where a file or symlink goes, or
	a file or symlink where a directory goes.
	skipped: the A / M rows check_patch_target found already done. When the removals took anything
	away, each one whose target is now missing or of another type (a delete that reached it through a
	name differing only by case, or through a symlinked directory) is written again like any row.
	Returns (those restored rows, the keys whose entries the removals took away).
	With metadata False (a FAT / exFAT / NTFS target) only content and file times are
	written: no mode, owner, xattrs or directory metadata (new directories get the target's
	default mode).
	"""
	root_norm = os.path.normpath(target_root)
	forced_rows = set(forced_rows)
	removed = []

	def guarded(row, removing=False):
		"""Target path for row, or None (with an error) when it must not be touched."""
		path = _key_path(target_root, row.path)
		if removing and os.path.normpath(path) == root_norm:
			eprint(f'Patch format error: {row.path} would remove the target root; skipped')
			return None
		if row.path != './' and not _inside_root(target_root, path):
			eprint(f'Patch format error: {row.path} resolves outside {target_root}; skipped')
			return None
		return path

	def is_real_dir(path):
		return os.path.isdir(path) and not os.path.islink(path)

	def remove(row):
		"""Unlink row's target, rmdir it, or (forced rows only) remove its whole tree, never entering a mount."""
		path = guarded(row, removing=True)
		if path is None:
			return
		directory = is_real_dir(path)
		try:
			if directory and row in forced_rows:
				# Recorded before the walk: one that stops partway may already have removed entries, so the
				# repair step must still check the skipped rows.
				removed.append(row.path)
				kept, errors = _remove_tree_around_mounts(path, target_root=target_root)
				for error in errors:
					eprint(f'Remove directory failed: {row.path}: {error}')
				if kept:
					eprint(f"Remove directory failed: {row.path}: {', '.join(kept[:3])}{' ...' if len(kept) > 3 else ''} "
						'is a mount point or another filesystem (a btrfs subvolume, say); left in place with the directories above it')
				return
			elif directory:
				os.rmdir(path)
			else:
				os.unlink(path)
			removed.append(row.path)
		except FileNotFoundError:
			pass
		except OSError as e:
			eprint(f"Remove {'directory' if directory else 'file'} failed: {row.path}: {e}")

	def remove_other_type(row):
		"""Remove what a rename cannot replace: a directory where a file or symlink goes, or the reverse."""
		if row.path == './':
			return
		path = guarded(row)
		if path is not None and os.path.lexists(path) and row.new.startswith('d:') != is_real_dir(path):
			remove(row)

	def sync_dir(row):
		"""Copy volume 1's directory metadata onto the target directory, with containment checks."""
		path = guarded(row)
		if path is None:
			return
		if row.path != './' and not is_real_dir(path):
			eprint(f'Patch format error: {row.path} is not a real directory on the target; metadata skipped')
			return
		source = _key_path(volume_roots[0], row.path)
		if row.path != './' and (not is_real_dir(source) or not _inside_root(volume_roots[0], source)):
			eprint(f'Patch format error: directory metadata source for {row.path} is not a real directory in volume 1; skipped')
			return
		try:
			sync_directory_metadata(source, [path])
		except OSError as e:
			eprint(f'Copy failed: {row.path}: {e}')

	deletes = [row for row in rows if row.op == 'D']
	payload = [row for row in rows if row.op in ('A', 'M')]
	# Non-directories first, then directories deepest first.
	for row in sorted(deletes, key=lambda r: (r.path.endswith('/'), -_key_depth(r.path))):
		remove(row)
	for row in payload:
		remove_other_type(row)
	restored = []
	if removed:
		restored = [row for row in skipped if row.op != 'D' and row.path != './'
			and not _holds_type(_key_path(target_root, row.path), row.new)]
		for row in restored:
			remove_other_type(row)
		if restored:
			print(f"Restoring {len(restored)} entr{'y' if len(restored) == 1 else 'ies'} skipped as already done that a removal "
				f"took away, e.g. {_printable_path(restored[0].path)}")
		payload = payload + restored
	dir_rows = []
	for row in sorted((r for r in payload if r.new.startswith('d:')), key=lambda r: _key_depth(r.path)):
		path = guarded(row)
		if path is None:
			continue
		try:
			# 0700 until sync_dir sets the payload's metadata; a content-only target keeps its default mode.
			_makedirs(path, mode=0o700 if metadata else 0o777, exist_ok=True)
		except OSError as e:
			eprint(f'Copy failed: {row.path}: {e}')
			continue
		dir_rows.append(row)
	files = []
	for row in payload:
		if not row.new.startswith('f:'):
			continue
		path = guarded(row)
		if path is None:
			continue
		source = _payload_source(volume_roots, row)
		if source is None:
			eprint(f'Incomplete patch: payload for {row.path} is missing or unsafe in every volume')
			continue
		files.append((row, source, path))
	_replace_files(files, target_root, max_workers, verbose, full_hash, metadata)
	for row in payload:
		if not row.new.startswith('l:') or row.path == './':
			continue
		path = guarded(row, removing=True)
		if path is not None:
			_replace_with_symlink(row, decode_state(row.new)['target'], path, target_root)
	if metadata:
		for row in sorted(dir_rows, key=lambda r: _key_depth(r.path), reverse=True):
			sync_dir(row)
	return restored, removed

def _verify_rows(rows, target_root, compare_hash=True, full_hash=False, max_workers=1, metadata=True, written_rows=(),
		passed=None, tz_tolerant=False):
	"""Re-read every applied path and compare with its final expected state (content only with
	metadata False); returns the failure count.

	A D row whose path still exists passes when an A / M row's name differs from it only by case
	(_fold_name) and both paths are one entry (same st_dev / st_ino): a case-only rename on a
	case-insensitive target, whether or not hpcp recognised the filesystem as one. written_rows: more
	A / M rows that count for this, not verified themselves. passed: a list that receives the rows that pass.
	Marked rows are read hashed in full with compare_hash (see check_patch_target); tz_tolerant: a FAT section.
	"""
	final = {}
	for row in sort_change_rows(rows):
		# D sorts before A, so for a D + A pair on one name the A row decides the final state.
		final[row.path.rstrip('/')] = row
	paths = {name: _key_path(target_root, row.path) for name, row in final.items()}
	marked = {paths[name] for name, row in final.items() if _full_marked(row.old) or _full_marked(row.new)} if compare_hash else set()
	states = _states_for_paths(set(paths.values()), full_hash=full_hash, read_xattrs=metadata, max_workers=max_workers, marked=marked)
	written = {}
	for name, row in final.items():
		if row.op != 'D':
			written.setdefault(_fold_name(name), []).append(paths[name])
	for row in written_rows:
		if row.op != 'D':
			written.setdefault(_fold_name(row.path.rstrip('/')), []).append(_key_path(target_root, row.path))
	failed = 0
	for name, row in final.items():
		expected = '-' if row.op == 'D' else row.new
		found = states[paths[name]]
		ok = found == '-' if expected == '-' else states_equal(found, expected, compare_hash, metadata, full_hash, tz_tolerant)
		if not ok and expected == '-' and _same_entry(paths[name], written.get(_fold_name(name), ())):
			ok = True
		if not ok:
			failed += 1
			eprint(f'Patch verify error: {row.op} {row.path}: expected {expected}, found {found}')
		elif passed is not None:
			passed.append(row)
	return failed

# metadata False: a content-only section (see states_equal). case_insensitive: the target looks names
# up case-insensitively (FAT, exFAT, HFS+ but not HFSX; see _has_case_insensitive_names). skipped_identities:
# {row: _entry_identity} of the A / M rows planned as skipped, recorded while planning (see _apply_section).
# tz_tolerant: a FAT section (see states_equal).
SectionPlan = namedtuple('SectionPlan', 'name volume_roots target_root rows checks metadata case_insensitive skipped_identities tz_tolerant')

@contextlib.contextmanager
def _patch_hash_settings(meta):
	"""Use the patch's hash size while checking and verifying; yields (compare_hash, full_hash)."""
	global HASH_SIZE
	compare_hash = meta.get('hash_algo') == _hash_algo_name()
	if not compare_hash:
		print(f"Warning: patch was hashed with {meta.get('hash_algo')}, this system uses {_hash_algo_name()}; hashes and xattr digests are not compared", file=sys.stderr)
	full_hash = meta.get('full_hash') == '1'
	try:
		patch_hash_size = int(meta.get('hash_size', HASH_SIZE))
	except ValueError:
		raise RuntimeError(f"Patch format error: bad hash_size {meta.get('hash_size')!r}")
	saved_hash_size = HASH_SIZE
	HASH_SIZE = patch_hash_size
	try:
		yield compare_hash, full_hash
	finally:
		HASH_SIZE = saved_hash_size

def _plan_section(name, rows, volume_roots, target_root, compare_hash, full_hash, max_workers, metadata=True, case_insensitive=False, tz_tolerant=False):
	"""Check one section's rows against target_root; nothing changes. The entries of the rows found already done are
	recorded, so a later write that reaches one of them (through a symlinked directory, or a name differing only by
	case) is seen."""
	checks = check_patch_target(rows, target_root, compare_hash, full_hash, max_workers, metadata, case_insensitive,
		tz_tolerant=tz_tolerant)
	skipped = [check.row for check in checks if check.action == 'skip' and check.row.op != 'D']
	return SectionPlan(name, list(volume_roots), target_root, rows, checks, metadata, case_insensitive,
		_identities(target_root, skipped), tz_tolerant)

def _section_conflicts(plan):
	return [check for check in plan.checks if check.action == 'conflict']

def _section_todo(plan):
	return [check.row for check in plan.checks if check.action != 'skip']

def _resolve_section_conflicts(plans, force, extra_lines=()):
	"""One conflict list and one prompt for every section (and extra lines such as raw areas).

	Before them, a note for each conflicting row whose forced removal of a directory would stop at a mount below it
	(see _remove_tree_around_mounts), so the user sees it before confirming.
	"""
	lines = []
	mount_points = None
	for plan in plans:
		conflicts = _section_conflicts(plan)
		where = '' if plan.name == 'root' else f'[{plan.name}] '
		lines += _conflict_lines(conflicts, where)
		for check in conflicts:
			row = check.row
			if not check.found.startswith('d:') or (row.op != 'D' and row.new.startswith('d:')):
				continue
			if mount_points is None:
				mount_points = {entry['mount_point'] for entry in _mountinfo_entries()}
			below = _mounts_below(_key_path(plan.target_root, row.path), mount_points)
			if below:
				first = os.path.normpath(os.path.join(row.path.rstrip('/'), below[0]))
				more = f' (and {len(below) - 1} more)' if len(below) > 1 else ''
				print(_printable_path(f'Note: forced removal of {where}{row.path} stops at the mount at {first}{more}; that row will fail verify'))
	lines += list(extra_lines)
	if lines:
		_resolve_conflict_lines(lines, force)

def _preflight_section(plan):
	"""Refusals -fa cannot override, raised before anything changes."""
	todo = _section_todo(plan)
	if plan.rows and not plan.volume_roots:
		# The section's payload directory is a symlink on every volume (_section_volume_roots): a damaged or crafted patch.
		# Refused even when every row is already done: a later restore (_restore_across_sections) needs the payload too.
		raise RuntimeError(f'Incomplete patch: no usable payload directory for section {plan.name} (a symlink on every volume); nothing applied')
	# Not forceable: the deletes would run and the writes be refused, leaving the patch half applied.
	escaping = _rows_escaping_target(todo, plan.target_root)
	if escaping:
		for row in escaping:
			print(f"  {row.op} {row.path!r}: resolves outside {plan.target_root} through a symlink")
		raise RuntimeError(f'Patch conflict: {len(escaping)} path(s) resolve outside the target through symlinks, e.g. {escaping[0].path}; nothing applied')
	unsafe = [row for row in todo if row.op in ('A', 'M') and row.new.startswith('f:') and _payload_source(plan.volume_roots, row) is None]
	if unsafe:
		raise RuntimeError(f'Incomplete patch: payload missing or unsafe for {len(unsafe)} file(s), e.g. {unsafe[0].path}; nothing applied')

def _apply_section(plan, max_workers, verbose, full_hash, compare_hash, outcome=None):
	"""Apply a checked section and verify it (with the rows the repair step restored); returns its summary.

	When the section writes anything, verify also covers each skipped A / M row its own writes changed: the row's
	_entry_snapshots (its entry, and a directory's state, since a directory changed in place keeps its inode) differ
	from a snapshot taken right before the writes (another row's write reached it through a symlinked directory, or
	through a name differing only by case on a case-insensitive filesystem, recognised or not). Such a contradictory
	patch then fails loudly. What an earlier section did to a skipped row (all sections are planned before any runs) is
	not this section's failure: the row keeps planning's entry (plan.skipped_identities), and _restore_across_sections
	and _verify_across_sections deal with it once every section ran, as on a single section.
	outcome: a dict that receives, for _restore_across_sections and _verify_across_sections: 'removed' (True when a
	removal took anything away), 'restored' (the rows the repair step wrote again), 'verified' (the rows verified),
	'failures' (the A / M rows that failed the verify), 'deletes' (the D rows that passed) and 'identities' (each A / M
	row that passed: its entry's _entry_identity right after the verify; each skipped row not verified: planning's).
	"""
	conflicts = _section_conflicts(plan)
	todo = _section_todo(plan)
	skipped = [check.row for check in plan.checks if check.action == 'skip' and check.row.op != 'D']
	# Taken right before the writes: a section's verify sees only what its own writes did.
	before = _entry_snapshots(plan.target_root, skipped, plan.metadata, max_workers) if todo else {}
	restored, removed = _apply_rows(todo, plan.volume_roots, plan.target_root, max_workers, verbose, full_hash,
		forced_rows=[check.row for check in conflicts], metadata=plan.metadata, skipped=skipped)
	restored_rows = set(restored)
	changed = []
	if todo:
		unrestored = [row for row in skipped if row not in restored_rows]
		after = _entry_snapshots(plan.target_root, unrestored, plan.metadata, max_workers)
		changed = [row for row in unrestored if after[row] != before[row]]
	verified, passed = todo + restored + changed, []
	failed = _verify_rows(verified, plan.target_root, compare_hash, full_hash, max_workers, plan.metadata, passed=passed,
		tz_tolerant=plan.tz_tolerant)
	if outcome is not None:
		# Skipped rows not verified here keep planning's entry, so the final pass across sections sees an earlier
		# section's change as well as a later one's.
		verified_rows, passed_rows = set(verified), set(passed)
		identities = {row: plan.skipped_identities.get(row) for row in skipped if row not in verified_rows}
		identities.update(_identities(plan.target_root, passed))
		outcome.update(removed=bool(removed), restored=restored, verified=verified, deletes=[row for row in passed if row.op == 'D'],
			identities=identities, failures={row for row in verified if row.op != 'D' and row not in passed_rows})
	return {
		'deleted': sum(1 for row in todo if row.op == 'D'),
		'added': sum(1 for row in todo if row.op == 'A'),
		'modified': sum(1 for row in todo if row.op == 'M'),
		'skipped': len(plan.checks) - len(todo) - len(restored),
		'restored': len(restored),
		'forced': len(conflicts),
		'failed': failed,
	}

def _restore_across_sections(ran, max_workers, verbose, full_hash, compare_hash):
	"""The repair step across sections, once every section ran (or the loop stopped after a failed one).

	A section's own repair step (_apply_rows) sees only its own removals, but a delete in one section can reach,
	through a symlinked directory, an entry another section skipped as already done (a mount below the target, a
	partition shown inside another). ran: [(plan, summary, outcome)] for the sections applied, outcome as
	_apply_section fills it. When any of them removed anything, every one's skipped A / M rows are lstat'ed again; a
	missing one is written again with its own section's root, payload and metadata setting, through the guarded
	writes _apply_rows uses (the directories of every section first), and verified. Nothing is removed: an entry of
	another type in a skipped row's place was put there by another row, so it is left and fails that row's verify. A
	row that failed its section's verify (outcome 'failures': already counted, a restore of its own among them) is not
	tried again. The summaries are updated ('restored', 'skipped', 'failed'), and the outcomes ('verified',
	'identities') for _verify_across_sections.
	"""
	if not any(outcome.get('removed') for _, _, outcome in ran):
		return
	lost = []
	for plan, summary, outcome in ran:
		tried = outcome.get('failures', set())
		rows = [check.row for check in plan.checks if check.action == 'skip' and check.row.op != 'D' and check.row.path != './'
			and check.row not in tried and not _holds_type(_key_path(plan.target_root, check.row.path), check.row.new)]
		if rows:
			lost.append((plan, summary, outcome, rows, []))
	missing = [(plan, row) for plan, _, _, rows, _ in lost for row in rows if not os.path.lexists(_key_path(plan.target_root, row.path))]
	if missing:
		plan, row = missing[0]
		example = row.path if plan.name == 'root' else f'[{plan.name}] {row.path}'
		print(f"Restoring {len(missing)} entr{'y' if len(missing) == 1 else 'ies'} skipped as already done that a removal "
			f"elsewhere in the patch took away, e.g. {_printable_path(example)}")
	for directories in (True, False):
		for plan, _, _, rows, written in lost:
			# Still missing: a restore of another section's row may have just written the same entry.
			batch = [row for row in rows if row.new.startswith('d:') == directories
				and not os.path.lexists(_key_path(plan.target_root, row.path))]
			if batch:
				_apply_rows(batch, plan.volume_roots, plan.target_root, max_workers, verbose, full_hash, metadata=plan.metadata)
				written.extend(batch)
	for plan, summary, outcome, rows, written in lost:
		summary['restored'] = summary.get('restored', 0) + len(written)
		summary['skipped'] -= len(written)
		passed = []
		summary['failed'] += _verify_rows(rows, plan.target_root, compare_hash, full_hash, max_workers, plan.metadata, passed=passed,
		tz_tolerant=plan.tz_tolerant)
		outcome['verified'] = list(outcome.get('verified', ())) + rows
		# A row that passed its section's verify and fails here must not be checked (and counted) again at the end.
		identities = outcome.setdefault('identities', {})
		for row in rows:
			identities.pop(row, None)
		identities.update(_identities(plan.target_root, passed))

def _verify_across_sections(ran, max_workers, full_hash, compare_hash):
	"""The final verify once more than one section ran, after _restore_across_sections.

	A section is verified before the next one runs, so a later section (or a restore) that reaches one of its
	entries through a symlinked directory would go unseen; nor does a section's verify cover what an earlier section
	did to a row it skipped (see _apply_section). On a single section the same contradictory patch fails at verify.
	ran as for _restore_across_sections. Every D row that passed its verify is verified again (an lstat; the
	case-only rename exception of _verify_rows still applies, with the section's verified A / M rows). Every
	directory row that passed, or that its section skipped without verifying, is verified again (no hashing): a
	directory is written in place, keeping its inode. Every other A / M row that passed, or that its section skipped
	without verifying, is lstat'ed and verified again only when its entry (_entry_identity, recorded right after its
	verify, or while planning) changed or is gone. A row that failed its section's verify is not verified again.
	Failures are reported as verify errors and added to the section's summary.
	"""
	if len(ran) < 2:
		return
	for plan, summary, outcome in ran:
		rows = list(outcome.get('deletes', ()))
		rows += [row for row, identity in outcome.get('identities', {}).items()
			if row.new.startswith('d:') or _entry_identity(_key_path(plan.target_root, row.path)) != identity]
		if rows:
			summary['failed'] += _verify_rows(rows, plan.target_root, compare_hash, full_hash, max_workers, plan.metadata,
				written_rows=outcome.get('verified', ()), tz_tolerant=plan.tz_tolerant)

def _trigger_automount(mount_point):
	"""Enter mount_point so that its automount (autofs) mounts what it stands for: a lookup through it (mount_point/.),
	since a plain stat of the mount point itself does not trigger one. A failure is left to the caller's new look at
	mountinfo."""
	try:
		os.stat(os.path.join(mount_point, '.'))
	except OSError:
		pass

def _autofs_is_direct(entries, real, prefix):
	"""True when the autofs mount standing at real/prefix (the target's own, the deepest one holding real, when prefix is
	'') is a direct map: mountinfo's super options carry 'direct'. An indirect map ('indirect') or an offset ('offset') is
	mounted only through its keys, never by a lookup of the mount point."""
	point = os.path.join(real, prefix) if prefix else real
	found = None
	for entry in entries:
		inside = entry['mount_point'].rstrip(os.sep) + os.sep
		if entry['fstype'] == 'autofs' and (entry['mount_point'] == point if prefix else (point.rstrip(os.sep) + os.sep).startswith(inside)):
			if found is None or len(entry['mount_point']) >= len(found['mount_point']):
				found = entry
	return found is not None and 'direct' in found.get('super_options', ())

def _trigger_automounts(target_root, rows, entries):
	"""mountinfo entries after triggering each visible direct automount (autofs, super option 'direct') at or below
	target_root that rows write into (rows _rows_by_scope groups under it, the target's own scope included); entries as
	given when there is none. Other automounts (/net, NFS home directories) are left alone, and so are indirect maps
	(a lookup of their mount point mounts nothing; apply_patch_tree refuses a patch writing into one)."""
	scopes = _mount_scopes(target_root, entries)
	grouped = _rows_by_scope(rows, scopes)
	real = os.path.realpath(target_root)
	points = [os.path.join(real, prefix) if prefix else real
		for prefix, fs_type, _ in scopes if fs_type == 'autofs' and prefix in grouped and _autofs_is_direct(entries, real, prefix)]
	for point in points:
		_trigger_automount(point)
	return _mountinfo_entries() if points else entries

def apply_patch_tree(volume_roots, target_root, force=False, max_workers=1, verbose=False, volume_names=None, store_note=False):
	"""Apply a patch whose volumes are mounted (or unpacked) at volume_roots onto directory target_root.

	Returns {'deleted', 'added', 'modified', 'skipped', 'restored', 'forced', 'failed'}. Raises RuntimeError
	for a malformed or incomplete patch, or when conflicts are not overridden.
	Another process modifying the target concurrently during an apply is out of scope
	(residual race between the containment checks and the writes).
	Files and symlinks are replaced atomically (a temporary sibling renamed over the target),
	so running processes keep the old file and every target ends either old or new.
	Paths are followed through symlinked directories that resolve inside target_root, for
	deletions and writes alike; a patch that would delete or write through a symlink leading
	outside it is refused before anything changes, even when forced (and before any conflict prompt).
	Only conflicts the user forced may have a whole directory tree removed.
	An interrupted apply leaves new directories 0700 and root-owned.
	A hard kill mid-apply can leave new directories with default metadata (it is applied
	only after every write); the re-run then reports those A rows as conflicts and needs -fa.
	Rows are planned per mount (the target's own, and each filesystem mounted below it, from
	/proc/self/mountinfo): on FAT, exFAT or NTFS by content only (on FAT with the timezone tolerance, see states_equal), on FAT, exFAT or HFS+ (not HFSX) with
	names matched case-insensitively (see check_patch_target). A write skipped as already done that a
	delete then takes away (a case-insensitive filesystem hpcp does not recognise, a symlinked directory
	alias, also from one mount into another) is written again and counted as restored (see _apply_rows
	and _restore_across_sections). With more than one mount patched, what one mount's rows or a
	restore changed in another is verified again at the end (see _verify_across_sections), so a
	contradictory patch fails as it would on a single filesystem.
	A direct automount (autofs) the patch writes into is triggered before the scopes are planned (see _trigger_automounts); one that cannot be mounted, and an indirect map, are refused before anything changes.
	"""
	meta, rows, volume_roots = load_patch_volumes(volume_roots, volume_names)
	if store_note and meta.get('layout') == 'store':
		example = rows[0].path if rows else ''
		print(f"Note: this patch uses image-store layout; paths such as {example!r} are resolved under {target_root}")
	# An automounted filesystem the patch writes into (an ESP systemd mounts on demand) is mounted first, so it is
	# planned as what it holds; one that stays an automount is refused before anything changes.
	entries = _trigger_automounts(target_root, rows, _mountinfo_entries())
	scopes = _mount_scopes(target_root, entries)
	grouped = _rows_by_scope(rows, scopes)
	real = os.path.realpath(target_root)
	for prefix, fs_type, _ in scopes:
		if fs_type == 'autofs' and prefix in grouped:
			point = os.path.join(real, prefix) if prefix else real
			if _autofs_is_direct(entries, real, prefix):
				raise RuntimeError(f'Patch error: {point} is an automount (autofs) that could not be mounted; mount it and re-run')
			raise RuntimeError(f"Patch error: {point} is an indirect automount map (autofs); hpcp does not write into one: leave it out of the patch "
				f"(-e '{'/' + prefix if prefix else ''}/*' when generating) or mount what it needs and re-run")
	with _patch_hash_settings(meta) as (compare_hash, full_hash):
		plans = []
		for prefix, fs_type, source in scopes:
			if prefix and prefix not in grouped:
				continue
			fs_type = _scope_fs_type(fs_type, source)
			plans.append(_plan_section(prefix + '/' if prefix else 'root', grouped.get(prefix, []), volume_roots, target_root,
				compare_hash, full_hash, max_workers, _has_posix_metadata(fs_type), _has_case_insensitive_names(fs_type, source),
				tz_tolerant=_is_fat(fs_type)))
		for plan in plans:
			_preflight_section(plan)
		_resolve_section_conflicts(plans, force)
		ran, stopped = [], None
		for plan in plans:
			outcome = {}
			ran.append((plan, _apply_section(plan, max_workers, verbose, full_hash, compare_hash, outcome), outcome))
			if ran[-1][1]['failed']:
				stopped = plan.name
				break
		_restore_across_sections(ran, max_workers, verbose, full_hash, compare_hash)
		_verify_across_sections(ran, max_workers, full_hash, compare_hash)
	summaries = [summary for _, summary, _ in ran]
	summary = {key: sum(part[key] for part in summaries) for key in summaries[0]}
	print('Patch summary: ' + ', '.join(f'{key} {value}' for key, value in summary.items()))
	if stopped is not None and len(summaries) < len(plans):
		eprint(f'Patch verify error: stopping: scope {stopped} failed; the filesystems mounted below were not patched; '
			'fix the cause and re-run')
	return summary

def _probe_fs_type(device):
	"""Filesystem type of device from `blkid -p` ('' when unknown). A failed probe leaves nothing in ERRORS."""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-s', 'TYPE', '-o', 'value', device], quiet=True, extra_env=_C_LOCALE)
	except Exception:
		output = []
	finally:
		del ERRORS[error_count:]
	return next((line.strip() for line in output if line.strip()), '')

def _ro_mount_options(device):
	"""Read-only mount options that also work for cloned (duplicate-UUID) xfs and dirty ext3/4 journals."""
	return {'xfs': _XFS_RO_MOUNT_OPTIONS, 'ext3': 'ro,noload', 'ext4': 'ro,noload'}.get(_probe_fs_type(device), 'ro')

def _rw_mount_command(device, mount_point):
	"""The read-write mount command for device: a clone of a mounted xfs (a golden image, an image of a live node's
	disk) carries the same UUID, which xfs refuses to mount twice without nouuid."""
	if _probe_fs_type(device) == 'xfs':
		return ['mount', '-o', 'nouuid', device, mount_point]
	return ['mount', device, mount_point]

def _dd_source_mount_command(device, mount_point, entries=None):
	"""The mount command for a -dd source partition, which is only read. One not mounted yet is mounted read-only, with
	nouuid for a cloned xfs; unlike _ro_mount_options this keeps log / journal recovery, so a dirty source is copied in
	its recovered state. One already mounted (a live system disk) shares its live filesystem, and the kernel refuses a
	mount that would change that filesystem's read-only state: it is mounted plainly when read-write (as before 9.73)."""
	entries = _mountinfo_entries() if entries is None else entries
	try:
		devnum = _device_number(device)
	except OSError:
		devnum = None
	live = [entry for entry in entries if _entry_matches(entry, devnum, device)]
	if live and not any('ro' in entry['super_options'] for entry in live):
		return ['mount', device, mount_point]
	if _probe_fs_type(device) == 'xfs':
		return ['mount', '-o', 'ro,nouuid', device, mount_point]
	return ['mount', '-o', 'ro', device, mount_point]

def _mount_single_fs_image(image, read_only, mount_points, loop_devices):
	"""Attach image, mount its largest partition (or the whole partitionless image), return the mount point + os.sep."""
	image = os.path.abspath(os.path.expanduser(image))
	if not os.path.isfile(image):
		raise RuntimeError(f'Diff image error: image {image} does not exist')
	if os.geteuid() != 0:
		print(f"Warning: mounting {image} usually needs root privileges", file=sys.stderr)
	loop_device = create_loop_device(image, read_only=read_only)
	loop_devices.append(loop_device)
	partitions = get_partitions(loop_device)
	if len(partitions) > 1:
		print(f"Warning: {image} has {len(partitions)} partitions; using the largest one. Use -dd for whole-disk images.", file=sys.stderr)
	target = get_largest_partition(loop_device) if partitions else loop_device
	mount_point = tempfile.mkdtemp()
	mount_points.append(mount_point)
	if read_only:
		# lsblk reads udev's cache, which a fresh loop device may not have yet; blkid -p reads the device.
		options = _ro_mount_options(target)
		command = ['mount', '-o', options, target, mount_point]
	else:
		command = _rw_mount_command(target, mount_point)
	run_command_in_multicmd_with_path_check(command, strict=False)
	if not os.path.ismount(mount_point):
		raise RuntimeError(f'Diff image error: cannot mount {target} from {image}')
	return mount_point + os.sep

def _drop_lost_found(entry_map):
	"""entry_map without lost+found/ and everything under it (mkfs.ext4 puts one on every image root)."""
	return {key: path for key, path in entry_map.items() if not key.startswith('lost+found/')}

def _inside_excludes(paths, roots):
	"""Rooted exclude patterns (matched against '/' + a path inside a listed tree; see _rooted_excludes), one for each
	of paths that lies strictly inside one of roots: its path relative to that root, compared by real path. Used for
	this run's mount points and inputs, so both listings leave them out. A root itself is never excluded."""
	patterns = set()
	for path in paths:
		path_real = os.path.realpath(path)
		for root in roots:
			root_real = os.path.realpath(root)
			if path_real.startswith(root_real.rstrip(os.sep) + os.sep):
				patterns.add(glob.escape(os.sep + os.path.relpath(path_real, root_real)))
	return patterns

def _rooted_excludes(patterns, root):
	"""patterns written for '/' + a path inside the tree at root, as patterns for the real paths under root.

	root's own path, glob-escaped, goes in front of each, so a real path matches exactly when its '/'-rooted path does:
	'/var/log/*' under /tmp/tmpab12 becomes '/tmp/tmpab12/var/log/*', and '*/tmp/*' matches a tmp/ inside the tree,
	never the tree's own (temporary) mount path. Nothing changes for root '/'."""
	prefix = glob.escape(os.path.abspath(root).rstrip(os.sep))
	return frozenset(prefix + pattern for pattern in patterns)

def _refuse_root_exclude(exclude, root=None):
	"""Refuse a user exclude pattern that leaves the compared tree's root out: the whole tree would drop out of one
	listing (a directory source's walker tests its root) and not the other, so the patch could delete everything.

	root None: a mounted filesystem (-si image, -dd -diff section), whose patterns are matched against '/' + the path
	inside it; else a directory source, whose patterns are matched against its real paths."""
	for pattern in sorted(exclude or ()):
		if root is None:
			hit, what = fnmatch.fnmatch(os.sep, pattern), 'the filesystem'
		else:
			hit, what = is_excluded(os.path.abspath(root), [pattern]), os.path.abspath(root)
		if hit:
			raise RuntimeError(f"Invalid arguments: exclude pattern {pattern!r} matches the root of {what}: it would leave the whole tree out of the comparison; remove it")

def process_get_diff_image(src_paths, src_image, dest_paths, dest_image, output, mount_points, loop_devices,
		exclude=None, full_hash=False, dest_image_size=0, max_workers=multiprocessing.cpu_count(),
		parallel_file_listing=False, verbose=False):
	"""Write a patch image that turns the base (-d DIR / -di IMG) into the new tree (src_path DIR / -si IMG).

	This run's own mount points are left out of both listings. When an image is compared root
	to root, lost+found/ at the root (and everything under it) is left out of both listings.
	Returns the patch path, or None when the trees do not differ (no image is created).
	"""
	if src_image:
		images = expand_paths(src_image)
		if len(images) != 1:
			raise RuntimeError(f'Invalid arguments: -diff needs exactly one -si image, got {images}')
		_refuse_root_exclude(exclude)
		new_root = _mount_single_fs_image(images[0], True, mount_points, loop_devices)
		new_name, new_is_dir = os.path.basename(images[0]), False
	else:
		paths = expand_paths(src_paths)
		if len(paths) != 1 or not os.path.isdir(paths[0]):
			raise RuntimeError(f'Invalid arguments: -diff needs exactly one existing source directory, got {src_paths}')
		new_root = os.path.abspath(paths[0])
		_refuse_root_exclude(exclude, new_root)
		new_name, new_is_dir = os.path.basename(os.path.normpath(new_root)) or 'root', True
	if dest_image:
		base_root = _mount_single_fs_image(dest_image, True, mount_points, loop_devices)
		base_name, base_is_image = os.path.basename(dest_image), True
	else:
		bases = expand_paths(dest_paths or [])
		if len(bases) != 1 or not os.path.isdir(bases[0]):
			raise RuntimeError(f'Invalid arguments: -diff needs exactly one existing base directory (-d), got {dest_paths}')
		base_root = os.path.abspath(bases[0])
		base_name, base_is_image = os.path.basename(os.path.normpath(base_root)) or 'root', False
	layout = 'store' if new_is_dir and base_is_image else 'root'
	# Store layout lists only the new tree's own subtree inside the store (see build_store_maps).
	base_listed = os.path.join(base_root, os.path.relpath(os.path.abspath(new_root), os.sep)) if layout == 'store' else base_root
	# This run's mount points, and its inputs inside a directory being compared (the -si / -di image files, the base
	# and new directories), stay out of both listings: `-diff / -di /root/golden-v1.img` must not ship the image.
	internal = _inside_excludes(mount_points, (new_root, base_listed))
	directories = [root for root, is_dir in ((new_root, new_is_dir), (base_root, not base_is_image)) if is_dir]
	images_in = [os.path.expanduser(path) for path in (images[0] if src_image else None, dest_image) if path]
	internal |= _inside_excludes(images_in + directories, directories)
	if src_image:
		# A mounted source: patterns are matched against '/' + the path inside each filesystem.
		new_map, base_map = build_rooted_maps(new_root, base_root, frozenset(exclude or ()) | internal, max_workers, parallel_file_listing)
	else:
		# A directory source: patterns are matched against its real paths, as in copy mode.
		exclude = frozenset(exclude or ()) | _rooted_excludes(internal, new_root)
		if layout == 'store':
			new_map, base_map = build_store_maps(new_root, base_root, exclude, max_workers, parallel_file_listing)
		else:
			new_map, base_map = build_root_maps(new_root, base_root, exclude, max_workers, parallel_file_listing)
	new_base = os.sep if layout == 'store' else new_root
	if layout == 'root' and (src_image or dest_image):
		# On both sides: `-si golden.img -d /` must not delete or replace the live root's own lost+found/.
		new_map = _drop_lost_found(new_map)
		base_map = _drop_lost_found(base_map)
	if any(_is_reserved_key(key) for key in new_map):
		raise RuntimeError(f'Diff image error: {new_root} contains {DIFF_DIR_NAME}, which hpcp reserves for patch metadata')
	if not output or output == 'auto':
		output = f'DIFF_{new_name}_TO_{base_name}_{int(time.time())}.img'
	output = os.path.abspath(os.path.expanduser(output))
	if os.path.lexists(output):
		raise RuntimeError(f'Diff image error: output {output} already exists')
	# Modes and owners on FAT, exFAT and NTFS come from mount options: compare content only when either side is one.
	entries = _mountinfo_entries()
	metadata = _compare_metadata((new_root, base_root), entries)
	# Files on FAT on either side (the root itself, or a mount below it such as /boot/efi) are hashed in full.
	fat_paths = _paths_on_fat(new_map.values(), new_root, entries) | _paths_on_fat(base_map.values(), base_root, entries)
	scan = {}
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata, fat_paths=fat_paths, scan=scan)
	# Fail here, before any image exists, on rows every apply would reject.
	_validate_patch_rows(rows)
	meta = make_patch_meta(rows, new_name, base_name, layout, full_hash)
	print(f"Changes: {meta['added']} added, {meta['modified']} modified, {meta['deleted']} deleted; payload {format_bytes(meta['payload_bytes'])}B")
	if not rows:
		print(f"No differences between {new_root} and {base_root}; no patch image created.")
		return None
	init_size = _payload_allocation(rows)
	if os.geteuid() != 0:
		print("Warning: creating a patch image usually needs root privileges (loop devices, mkfs, mount)", file=sys.stderr)
	target_mount_point = tempfile.mkdtemp()
	mount_points.append(target_mount_point)
	count, _ = _plan_dest_images(init_size, dest_image_size)
	_claim_volume_names(output, count)
	volume_roots = create_image(output, target_mount_point, loop_devices, [], mount_points, max_workers=max_workers,
		dest_image_size=dest_image_size, init_size=init_size)
	build_patch_tree(rows, new_base, volume_roots, meta, max_workers=max_workers, verbose=verbose, scan=scan)
	extra = f" (+{len(volume_roots) - 1} more volume(s))" if len(volume_roots) > 1 else ''
	print(f"Patch image written to {output}{extra}")
	return output

def _patch_layout(volume_roots):
	"""layout of the first readable manifest among the volumes ('root' when none can be read)."""
	for volume_root in volume_roots:
		try:
			return read_manifest(os.path.join(volume_root, DIFF_DIR_NAME, 'manifest')).get('layout', 'root')
		except (OSError, RuntimeError):
			continue
	return 'root'

def process_load_diff_image(load_diff_image, src_paths, dest_paths, dest_image, mount_points, loop_devices,
		force=False, max_workers=multiprocessing.cpu_count(), verbose=False, dd=False):
	"""Mount every -siff volume and apply the patch: a disk patch to -d DISK (or one path), a
	single-filesystem patch to -di IMG, -d DIR, or a single positional path."""
	images = expand_paths(load_diff_image)
	if not images:
		raise RuntimeError(f'Incomplete patch: no patch image found for {load_diff_image}')
	volume_roots = [_mount_single_fs_image(image, True, mount_points, loop_devices) for image in images]
	volume_names = [os.path.abspath(image) for image in images]
	if _patch_layout(volume_roots) == 'disk':
		if dest_image:
			raise RuntimeError('Invalid arguments: disk patches take the target disk with -d (or a single path), not -di')
		targets = expand_paths(dest_paths) if dest_paths else expand_paths(src_paths or [])
		if len(targets) != 1 or os.path.isdir(targets[0]):
			raise RuntimeError(f'Invalid arguments: a disk patch needs exactly one target disk (block device or disk image file), got {dest_paths or src_paths}')
		return apply_disk_patch(volume_roots, targets[0], mount_points, loop_devices, force=force, max_workers=max_workers,
			verbose=verbose, volume_names=volume_names)
	if dd:
		raise RuntimeError('Invalid arguments: -dd with -siff needs a disk patch; this is a single-filesystem patch')
	if dest_image:
		target_root = _mount_single_fs_image(dest_image, False, mount_points, loop_devices)
		store_note = False
	else:
		targets = expand_paths(dest_paths) if dest_paths else expand_paths(src_paths or [])
		if len(targets) != 1 or not os.path.isdir(targets[0]):
			raise RuntimeError(f'Invalid arguments: -siff needs exactly one existing target directory, got {dest_paths or src_paths}')
		target_root = os.path.abspath(targets[0])
		store_note = target_root != os.path.abspath(os.sep)
	try:
		return apply_patch_tree(volume_roots, target_root, force=force, max_workers=max_workers, verbose=verbose,
			volume_names=volume_names, store_note=store_note)
	finally:
		# A directory on a filesystem someone else mounted is written to the page cache only (see apply_disk_patch).
		_syncfs(target_root)

def _validate_diff_args(get_diff_image, load_diff_image, force_apply, src_path, src_image, dest_paths, dest_image,
		dd=False, remove=False, remove_extra=False, target_file_list=None, compare_file_list=False,
		diff_file_list=None, tar_diff_file_list=False, content_only=False):
	"""Check -diff / -siff / -fa combinations before anything is mounted.

	Returns (get_diff_image, src_path). A -diff value that is an existing directory is the
	source path that argparse swallowed (`hpcp -diff /new -d /base`); it is moved back.
	"""
	src_path = list(src_path or [])
	dest_paths = list(dest_paths or [])
	if force_apply and not load_diff_image:
		raise RuntimeError('Invalid arguments: -fa / --force_apply only applies together with -siff')
	if get_diff_image and load_diff_image:
		raise RuntimeError('Invalid arguments: -diff and -siff cannot be used together')
	mode = '-siff' if load_diff_image else '-diff'
	clashes = [flag for flag, enabled in (('-rm/-rf', remove), ('-rme', remove_extra), ('-fl', target_file_list),
		('-cfl', compare_file_list), ('-dfl', diff_file_list), ('-tdfl', tar_diff_file_list), ('-co', content_only)) if enabled]
	if clashes:
		raise RuntimeError(f'Invalid arguments: {mode} cannot be combined with {", ".join(clashes)}')
	if load_diff_image:
		if src_image:
			raise RuntimeError('Invalid arguments: -siff takes no -si image; give the patch volumes with -siff')
		targets = (1 if dest_image else 0) + len(dest_paths)
		if targets and src_path:
			raise RuntimeError(f'Invalid arguments: -siff takes no source paths when -d / -di is given, got {src_path}')
		if not targets:
			targets = len(src_path)
		if targets != 1:
			raise RuntimeError(f'Invalid arguments: -siff needs exactly one target (-d DIR, -di IMG or one path), got {targets}')
		return get_diff_image, src_path
	if dd:
		if src_image or dest_image:
			raise RuntimeError('Invalid arguments: -dd -diff takes the new disk as a path and the old disk with -d (no -si / -di)')
		if get_diff_image not in (None, 'auto') and os.path.lexists(os.path.expanduser(get_diff_image)):
			# The output must not exist, so an existing path is the new disk argparse took as the patch name.
			print(f"Note: -diff value {get_diff_image} exists; using it as the new disk and generating the patch name")
			src_path = [get_diff_image] + src_path
			get_diff_image = 'auto'
		if len(src_path) != 1:
			raise RuntimeError(f'Invalid arguments: -dd -diff needs exactly one new disk (a path), got {len(src_path)}')
		if len(dest_paths) != 1:
			raise RuntimeError(f'Invalid arguments: -dd -diff needs exactly one old disk (-d), got {len(dest_paths)}')
		return get_diff_image, src_path
	if get_diff_image not in (None, 'auto') and os.path.isdir(os.path.expanduser(get_diff_image)):
		print(f"Note: -diff value {get_diff_image} is an existing directory; using it as the source and generating the patch name")
		src_path = [get_diff_image] + src_path
		get_diff_image = 'auto'
	sources = len(src_path) + len(src_image or [])
	if sources != 1:
		raise RuntimeError(f'Invalid arguments: -diff needs exactly one source (a directory or one -si image), got {sources}')
	bases = (1 if dest_image else 0) + len(dest_paths)
	if bases != 1:
		raise RuntimeError(f'Invalid arguments: -diff needs exactly one base (-d DIR or -di IMG), got {bases}')
	return get_diff_image, src_path

#%% ---- Disk Patch ----
_ESP_TYPE_GUID = 'C12A7328-F81F-11D2-BA4B-00A0C93EC93B'

def _sysfs_block_name(device):
	"""device's name in /sys/class/block: its real path below /dev with each '/' spelled '!', as the
	kernel does (/dev/cciss/c0d0 is cciss!c0d0); the base name of a node outside /dev."""
	real = os.path.realpath(device)
	if real.startswith('/dev/'):
		return real[len('/dev/'):].replace('/', '!')
	return os.path.basename(real)

def _is_kernel_partition(device, sys_block_path='/sys/class/block'):
	"""True when the kernel keeps device as a partition: its sysfs directory holds a 'partition' file."""
	return os.path.exists(os.path.join(sys_block_path, _sysfs_block_name(device), 'partition'))

def _partition_devices(disk, sys_block_path='/sys/class/block'):
	"""{partition number: device path} for disk, using the kernel's partition number, not the name suffix.

	Only the disk's own partitions count: the real child directories of its sysfs node that hold
	a 'partition' file. Other descendants (a partitioned md array built on its partitions) do not.
	A '!' in a sysfs name is a '/' in /dev (cciss!c0d0p1 is /dev/cciss/c0d0p1).
	"""
	result = {}
	disk_dir = os.path.join(sys_block_path, _sysfs_block_name(disk))
	try:
		children = os.listdir(disk_dir)
	except OSError:
		return result
	for child in children:
		child_dir = os.path.join(disk_dir, child)
		if os.path.islink(child_dir):
			continue
		try:
			with open(os.path.join(child_dir, 'partition')) as f:
				result[int(f.read().strip())] = _dev_path(child)
		except (OSError, ValueError):
			continue
	return result

def _udev_settle():
	"""Wait until udev has handled its queued events, up to 10 s (quietly; nothing when udevadm is missing).
	A partition re-read settles in milliseconds; the cap keeps one stuck udev event ( a hung multipath path ) from
	holding every partition listing for udevadm's default 120 s."""
	if check_path('udevadm'):
		error_count = len(ERRORS)
		try:
			run_command_in_multicmd_with_path_check(['udevadm', 'settle', '--timeout=10'], quiet=True)
		finally:
			del ERRORS[error_count:]

def _settle_partitions(disk, numbers, timeout=10):
	"""{number: device} once udev is idle and every partition in numbers has its device node.

	udev re-reads a whole disk's partition table after the disk is closed from writing (a new
	loop device, an MBR or GPT write, an sgdisk read), removing and re-adding the partition nodes meanwhile.
	Raises RuntimeError('Disk layout mismatch: ...') when a node is still missing after timeout seconds.
	"""
	_udev_settle()
	deadline = time.time() + timeout
	while True:
		devices = _partition_devices(disk)
		missing = [number for number in numbers if number not in devices or not os.path.exists(devices[number])]
		if not missing:
			return devices
		if time.time() >= deadline:
			raise RuntimeError(f'Disk layout mismatch: the kernel shows no device for partition(s) {missing} of {disk}; re-read its partition table (partprobe) first')
		time.sleep(0.1)

def _stable_fs_params(value, key=''):
	"""value without what differs between two copies of one filesystem: device paths (xfs_info's
	meta-data=/dev/...) and ext runtime-state features (needs_recovery, orphan_present, ...)."""
	if isinstance(value, dict):
		return {name: _stable_fs_params(item, name) for name, item in value.items()
			if not (isinstance(item, str) and item.startswith('/dev/'))}
	if key == 'features' and isinstance(value, list):
		return [feature for feature in value if feature not in _EXT_RUNTIME_FEATURES]
	return value

def _fs_params_text(params):
	"""Stable text form of probe_fs_params() output for disk.tsv (see _stable_fs_params)."""
	params = _stable_fs_params(params or {})
	return ';'.join(f'{key}={params[key]}' for key in sorted(params))

_UDEV_HEX_ESCAPE_RE = re.compile(rb'\\x([0-9a-fA-F]{2})')
# multiCMD's backslashreplace escape of an output byte that is not UTF-8: always 0x80 or above, in lowercase. Only
# these \x80-\xff escapes are decoded on a raw `-o value` line: a label's own literal '\x41' text stays.
_BACKSLASHREPLACE_RE = re.compile(rb'\\x([89a-f][0-9a-f])')
# blkid -o udev keeps an *_ENC value in a 265-byte buffer: a longer one comes out cut, at times mid-escape (a\x2).
# Counted in UTF-8 bytes, as the buffer is: blkid keeps UTF-8 as it is, so a multibyte label is cut in fewer characters.
_UDEV_ENC_CUT_BYTES = 261
_UDEV_PARTIAL_ESCAPE_RE = re.compile(r'\\(x[0-9a-fA-F]?)?$')
# The filesystems whose label can pass 127 bytes and whose own tool reads it (see _probe_identity).
_LABEL_READERS = {'btrfs': ['btrfs', 'filesystem', 'label'], 'ntfs': ['ntfslabel']}

def _udev_decode(value):
	"""A blkid `-o udev` *_ENC value as text, with the original bytes: blkid writes unsafe bytes (a space, a
	backslash, bytes that are not UTF-8) as \\xNN and the rest as UTF-8, and multiCMD decodes undecodable output
	bytes to the same escape. Bytes that are not UTF-8 are kept with surrogateescape."""
	data = value.encode('utf-8', 'surrogateescape')
	data = _UDEV_HEX_ESCAPE_RE.sub(lambda match: bytes([int(match.group(1), 16)]), data)
	return data.decode('utf-8', 'surrogateescape')

def _probe_identity(device):
	"""(fs_type, fs_uuid, fs_label) of device from a low-level `blkid -p -o udev` probe, read exactly ('' when
	nothing is found).

	-o udev gives encoded forms (ID_FS_UUID_ENC, ID_FS_LABEL_ENC; see _udev_decode). -o export prints bytes above
	127 in M- notation in every locale, so a label such as 'röot' would read back mangled. No blkid cache, so a
	label or UUID this run just changed reads back as it is on disk; exact keys, so UUID_SUB (btrfs) and
	LABEL_FATBOOT (FAT) never stand in for UUID and LABEL.
	An ID_FS_LABEL_ENC value of 261 bytes or more, or one ending in a partial escape, may have been cut (blkid keeps
	it in a 265-byte buffer; a long btrfs, NTFS, f2fs or udf label full of spaces needs more): the label is then
	read raw (_probe_raw_label).
	A btrfs or NTFS label comes from that filesystem's own tool (_LABEL_READERS): blkid 2.23.2 (EL7) cuts every label
	at 127 bytes and trims what is left, so no length shows the cut.
	"""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'udev', device], quiet=True, extra_env=_C_LOCALE)
	finally:
		# blkid exits 2 when it finds nothing (BIOS boot and other raw partitions).
		del ERRORS[error_count:]
	values = {}
	for line in output:
		key, sep, value = line.rstrip('\r\n').partition('=')
		if sep:
			values[key] = value
	uuid = values.get('ID_FS_UUID_ENC', values.get('ID_FS_UUID', ''))
	label = values.get('ID_FS_LABEL_ENC', '')
	fs_type = values.get('ID_FS_TYPE', '')
	exact_label = _probe_tool_label(fs_type, device) if label else None
	if not exact_label and (len(label.encode('utf-8', 'surrogateescape')) >= _UDEV_ENC_CUT_BYTES or _UDEV_PARTIAL_ESCAPE_RE.search(label)):
		exact_label = _probe_raw_label(device)
	return fs_type, _udev_decode(uuid), exact_label if exact_label else _udev_decode(label)

def _label_printed_raw(command):
	"""The label command prints raw on its first line (None when it prints nothing; a failure leaves nothing in ERRORS).

	multiCMD writes each output byte that is not UTF-8 as \\xNN (0x80 and above); those become the bytes again, kept
	with surrogateescape. A backslash in the label itself is printed as it is and stays: _udev_decode would read a
	label's own '\\x41' as 'A'.
	"""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(command, quiet=True, extra_env=_C_LOCALE)
	finally:
		del ERRORS[error_count:]
	if not output:
		return None
	data = output[0].rstrip('\r\n').encode('utf-8', 'surrogateescape')
	data = _BACKSLASHREPLACE_RE.sub(lambda match: bytes([int(match.group(1), 16)]), data)
	return data.decode('utf-8', 'surrogateescape')

def _probe_raw_label(device):
	"""device's label from `blkid -p -o value -s LABEL` (see _label_printed_raw)."""
	return _label_printed_raw(['blkid', '-p', '-o', 'value', '-s', 'LABEL', device])

def _probe_tool_label(fs_type, device):
	"""device's label from its filesystem's own tool (see _label_printed_raw); None for a filesystem with none in
	_LABEL_READERS, or without the tool."""
	reader = _LABEL_READERS.get(fs_type)
	return _label_printed_raw(reader + [device]) if reader and check_path(reader[0]) else None

def _partition_table_type(device):
	"""Partition table type of device from a low-level `blkid -p` probe ('gpt', 'dos', ...; '' when it finds
	none, which includes an ambivalent probe: see _no_partition_table)."""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'value', '-s', 'PTTYPE', device], quiet=True, extra_env=_C_LOCALE)
	finally:
		# blkid exits 2 when it finds nothing.
		del ERRORS[error_count:]
	return next((line.strip() for line in output if line.strip()), '')

def _no_partition_table(name):
	"""Why a disk on which _partition_table_type found nothing is refused. blkid also finds nothing when
	its probe is ambivalent: stale whole-disk RAID or filesystem signatures next to a GPT."""
	return (f'blkid found no partition table on {name} (if it is a GPT disk, stale whole-disk signatures may hide it: '
		f'check `wipefs -n {name}`)')

def read_disk_layout(device, name=None, pttype=None, fs_params=True):
	"""Fresh GPT + filesystem identity read of block device (name: the disk as messages call it, default device).

	Returns {'disk': {'guid', 'sector_size', 'size'}, 'parts': {number: {'type_guid',
	'unique_guid', 'name', 'attrs', 'size', 'fs_type', 'fs_uuid', 'fs_label', 'fs_params',
	'device'}}}. The partition caches are cleared first: loop device names are reused.
	pttype: the partition-table type the caller already probed (_partition_table_type), so it is not probed twice.
	The GPT is read with sgdisk alone, udev is then settled once (sgdisk's runs make it re-read the partition table),
	and each partition is probed once on its settled device (_probe_identity); fs_params False skips the mkfs-parameter
	probes ('fs_params' is then '': a verify read compares identity values only).
	Raises RuntimeError('Disk layout mismatch: ...') unless the disk has a GPT partition table:
	sgdisk reads an MBR disk as if it were GPT, and an identity change would convert it.
	"""
	name = name or device
	if pttype is None:
		pttype = _partition_table_type(device)
	if not pttype:
		raise RuntimeError(f'Disk layout mismatch: {_no_partition_table(name)}; disk patches support GPT disks only')
	if pttype != 'gpt':
		raise RuntimeError(f'Disk layout mismatch: {name} has no GPT partition table (found {pttype}); disk patches support GPT disks only')
	get_partition_infos.cache_clear()
	get_partition_details.cache_clear()
	# A node udev is still re-adding would be probed as empty, so udev is settled after the GPT is read, then every
	# node waited for: every sgdisk run, --print included, makes udev re-read the partition table (its watch sees a
	# close after a write-open), removing and re-adding the nodes a few ms after sgdisk exits.
	infos = get_partition_infos(device, identity=False)
	devices = _settle_partitions(device, [int(key) for key, info in infos.items() if 'disk_identifier' not in info])
	disk, parts = {}, {}
	for key, info in infos.items():
		if 'disk_identifier' in info:
			disk = {'guid': info['disk_identifier'], 'sector_size': info['sector_size'], 'size': info['size']}
			continue
		number = int(key)
		fs_type, fs_uuid, fs_label = _probe_identity(devices[number])
		params = probe_fs_params(devices[number], fs_type) if fs_params and fs_type and MIRROR_FS_PARAMS else {}
		parts[number] = {
			'type_guid': info['partition_guid_code'], 'unique_guid': info['unique_partition_guid'],
			'name': info['partition_name'], 'attrs': info['partition_attrs'], 'size': info['size'],
			'fs_type': fs_type, 'fs_uuid': fs_uuid, 'fs_label': fs_label,
			'fs_params': _fs_params_text(params), 'device': devices[number],
		}
	return {'disk': disk, 'parts': parts}

def _default_kind(part):
	"""'swap' for swap, 'fs' when blkid found a filesystem, else 'raw' (generation may downgrade fs to raw)."""
	if part['fs_type'] == 'swap':
		return 'swap'
	return 'fs' if part['fs_type'] else 'raw'

def compare_disk_layouts(old, new, kinds=None):
	"""(problems, rows): why a patch from old to new is impossible, and the disk.tsv rows.

	Partition numbers, type GUIDs, filesystem types and the sector size must match; a raw
	area must keep its size. kinds maps partition number -> 'fs' / 'swap' / 'raw'
	(default: _default_kind of the new partition).
	"""
	problems = []
	kinds = kinds or {}
	if old['disk'].get('sector_size') != new['disk'].get('sector_size'):
		problems.append(f"sector size {old['disk'].get('sector_size')} vs {new['disk'].get('sector_size')}")
	if sorted(old['parts']) != sorted(new['parts']):
		problems.append(f"partition numbers {sorted(old['parts'])} vs {sorted(new['parts'])}")
		return problems, []
	rows = [DiskRow('disk', '-', 'guid', str(old['disk'].get('guid', '')), str(new['disk'].get('guid', ''))),
		DiskRow('disk', '-', 'sector_size', str(old['disk'].get('sector_size', '')), str(new['disk'].get('sector_size', '')))]
	for number in sorted(new['parts']):
		old_part, new_part = old['parts'][number], new['parts'][number]
		kind = kinds.get(number) or _default_kind(new_part)
		for field in ('type_guid', 'fs_type'):
			if old_part[field] != new_part[field]:
				problems.append(f'partition {number} {field} {old_part[field]!r} vs {new_part[field]!r}')
		if kind == 'raw' and old_part['size'] != new_part['size']:
			problems.append(f"partition {number} is a raw area and its size changed ({old_part['size']} vs {new_part['size']})")
		rows.append(DiskRow('part', str(number), 'kind', kind, kind))
		for field in _PART_FIELDS[1:]:
			rows.append(DiskRow('part', str(number), field, str(old_part[field]), str(new_part[field])))
	return problems, rows

def match_target_layout(disk_rows, target):
	"""Why target cannot take a patch with these disk.tsv rows (empty when it matches its base)."""
	problems = []
	parts = {}
	for row in disk_rows:
		if row.kind == 'disk' and row.field == 'sector_size' and str(target['disk'].get('sector_size')) != row.old:
			problems.append(f"sector size {target['disk'].get('sector_size')}, patch expects {row.old}")
		elif row.kind == 'part':
			parts.setdefault(int(row.number), {})[row.field] = row
	if sorted(parts) != sorted(target['parts']):
		problems.append(f"partition numbers {sorted(target['parts'])}, patch expects {sorted(parts)}")
		return problems
	for number, fields in sorted(parts.items()):
		missing = [field for field in _PART_FIELDS if field not in fields]
		if missing:
			raise RuntimeError(f"Patch format error: the disk table lacks {', '.join(missing)} for partition {number}")
		part = target['parts'][number]
		for field in ('type_guid', 'fs_type'):
			if str(part[field]) != fields[field].old:
				problems.append(f'partition {number} {field} is {part[field]!r}, patch expects {fields[field].old!r}')
		if fields['kind'].old == 'raw' and str(part['size']) != fields['size'].old:
			problems.append(f"partition {number} is a raw area of size {part['size']}, patch expects {fields['size'].old}")
	return problems

_MBR_BOOTCODE_SIZE = 440
_AREA_CHUNK = 4 << 20

_BLKSSZGET = 0x1268
_BLKFLSBUF = 0x1261

def _logical_block_size(fd):
	"""fd's logical block size (BLKSSZGET); 4096 when it is not a block device or unknown, which suits every size."""
	try:
		import fcntl
		size = struct.unpack('i', fcntl.ioctl(fd, _BLKSSZGET, b'\0' * 4))[0]
	except (ImportError, OSError):
		return 4096
	return size if size > 0 else 4096

def _hash_area_direct(path, offset, length):
	"""sha256 of length bytes of path from offset, read past the page cache (O_DIRECT); None when O_DIRECT is refused.

	O_DIRECT needs the buffer, the file offset and the read size aligned to the logical block size: the reads cover
	the enclosing blocks into a page-aligned buffer (mmap), and only the requested bytes are hashed. Raises OSError
	when path ends before offset + length.
	"""
	flag = getattr(os, 'O_DIRECT', 0)
	if not flag:
		return None
	import mmap
	try:
		fd = os.open(path, os.O_RDONLY | flag | getattr(os, 'O_CLOEXEC', 0))
	except OSError as e:
		if e.errno == errno.EINVAL:
			return None
		raise
	buffer = None
	try:
		block = _logical_block_size(fd)
		chunk = max(block, _AREA_CHUNK - _AREA_CHUNK % block)
		buffer = mmap.mmap(-1, chunk)
		end = offset + length
		position = offset - offset % block
		digest = hashlib.sha256()
		while position < end:
			os.lseek(fd, position, os.SEEK_SET)
			try:
				count = os.readv(fd, [buffer])
			except OSError as e:
				if e.errno == errno.EINVAL and position == offset - offset % block:
					return None
				raise
			if count <= 0:
				break
			first, last = max(offset - position, 0), min(count, end - position)
			if last > first:
				with memoryview(buffer) as view:
					digest.update(view[first:last])
			position += count
		if position < end:
			raise OSError(errno.EIO, f'{path} ends before byte {end}')
		return digest.hexdigest()
	finally:
		if buffer is not None:
			buffer.close()
		os.close(fd)

def _drop_cached_pages(path):
	"""Ask the kernel to forget path's cached pages: BLKFLSBUF for a block device, then posix_fadvise DONTNEED."""
	try:
		fd = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
	except OSError:
		return
	try:
		try:
			import fcntl
			if stat.S_ISBLK(os.fstat(fd).st_mode):
				fcntl.ioctl(fd, _BLKFLSBUF, 0)
		except (ImportError, OSError):
			pass
		if hasattr(os, 'posix_fadvise'):
			try:
				os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
			except OSError:
				pass
	finally:
		os.close(fd)

def _hash_area(path, offset=0, length=None, direct=False):
	"""sha256 hex digest of length bytes of path from offset (to the end when length is None).

	direct (with a length): read past the page cache (_hash_area_direct), so a read-back sees what the device holds;
	where O_DIRECT is refused, the cached pages are dropped first (_drop_cached_pages) and the area read normally.
	Raises OSError when path ends before offset + length.
	"""
	if direct and length is not None:
		found = _hash_area_direct(path, offset, length)
		if found is not None:
			return found
		_drop_cached_pages(path)
	digest = hashlib.sha256()
	remaining = length
	with open(path, 'rb') as f:
		f.seek(offset)
		while remaining is None or remaining > 0:
			chunk = f.read(_AREA_CHUNK if remaining is None else min(_AREA_CHUNK, remaining))
			if not chunk:
				break
			digest.update(chunk)
			if remaining is not None:
				remaining -= len(chunk)
	if remaining:
		raise OSError(errno.EIO, f'{path} ends before byte {offset + length}')
	return digest.hexdigest()

def _copy_area(source, dest, offset, length):
	"""Copy length bytes of source from offset into the new file dest (fsync'd); returns their sha256."""
	digest = hashlib.sha256()
	remaining = length
	with open(source, 'rb') as src, open(dest, 'wb') as out:
		src.seek(offset)
		while remaining > 0:
			chunk = src.read(min(_AREA_CHUNK, remaining))
			if not chunk:
				raise OSError(errno.EIO, f'{source} ends before byte {offset + length}')
			out.write(chunk)
			digest.update(chunk)
			remaining -= len(chunk)
		out.flush()
		os.fsync(out.fileno())
	return digest.hexdigest()

def _write_area(target, payload, offset, length):
	"""Write the first length bytes of payload into target at offset, then fsync.

	Bytes of target outside [offset, offset + length) are left as they are (the MBR keeps its
	disk signature, partition entries and boot signature).
	"""
	remaining = length
	with open(payload, 'rb') as src, open(target, 'r+b') as out:
		out.seek(offset)
		while remaining > 0:
			chunk = src.read(min(_AREA_CHUNK, remaining))
			if not chunk:
				raise OSError(errno.EIO, f'{payload} is shorter than {length} bytes')
			out.write(chunk)
			remaining -= len(chunk)
		out.flush()
		os.fsync(out.fileno())

def _raw_payload_candidates(volume_roots, raw_file, size):
	"""Paths of raw_file on the volumes that are regular files inside their volume with the recorded size."""
	for volume_root in volume_roots:
		path = os.path.join(volume_root, raw_file)
		if (_inside_root(volume_root, path) and os.path.isfile(path) and not os.path.islink(path)
				and os.path.getsize(path) == size):
			yield path

def _raw_payload_present(volume_roots, raw_file, size):
	"""True when some volume holds raw_file with the recorded size (checked before anything is hashed; its content
	is checked by _raw_payload)."""
	return next(_raw_payload_candidates(volume_roots, raw_file, size), None) is not None

def _raw_payload(volume_roots, raw_file, new_sha, size):
	"""Path of raw_file on a volume when it is a regular file inside that volume with the recorded
	size and sha256; None when no volume holds an intact copy."""
	return next((path for path in _raw_payload_candidates(volume_roots, raw_file, size) if _hash_area(path) == new_sha), None)

def _plan_raw(old_sha, new_sha, found_sha):
	"""'skip' when the area already holds the new content, 'apply' when it holds the old, else 'conflict'."""
	if found_sha == new_sha:
		return 'skip'
	if found_sha == old_sha:
		return 'apply'
	return 'conflict'


def _unescape_mount(text):
	"""Undo the octal escapes (\\040 for a space, …) the kernel uses in mountinfo and /proc/swaps."""
	return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), text)

def _mountinfo_entries(mountinfo_path='/proc/self/mountinfo'):
	"""Parsed mountinfo lines (an empty list when the file cannot be read)."""
	entries = []
	try:
		f = open(mountinfo_path, encoding='utf-8', errors='surrogateescape')
	except OSError:
		return entries
	with f:
		for line in f:
			fields = line.split()
			if '-' not in fields or len(fields) < 7:
				continue
			try:
				separator = fields.index('-')
				major, _, minor = fields[2].partition(':')
				entries.append({
					'mount_id': int(fields[0]), 'parent_id': int(fields[1]),
					'devnum': (int(major), int(minor)), 'root': _unescape_mount(fields[3]),
					'mount_point': _unescape_mount(fields[4]), 'options': fields[5].split(','),
					'fstype': fields[separator + 1] if len(fields) > separator + 1 else '',
					'source': _unescape_mount(fields[separator + 2]) if len(fields) > separator + 2 else '',
					'super_options': fields[separator + 3].split(',') if len(fields) > separator + 3 else [],
				})
			except (ValueError, IndexError):
				continue
	return entries

def _device_number(device):
	"""(major, minor) of a block device node."""
	st = os.stat(device)
	return (os.major(st.st_rdev), os.minor(st.st_rdev))

def _entry_matches(entry, devnum, device):
	"""A mountinfo entry belongs to device: same device number, or (btrfs and other filesystems with
	anonymous device numbers) the same source device path."""
	if entry['devnum'] == devnum:
		return True
	return bool(device) and entry['source'].startswith('/') and os.path.realpath(entry['source']) == os.path.realpath(device)

def _live_mount(devnum, entries, device=''):
	"""(mount point, read_only) of the mount that shows device's whole filesystem (root '/'), or (None, False)."""
	for entry in entries:
		if entry['root'] == '/' and _entry_matches(entry, devnum, device):
			return entry['mount_point'], 'ro' in entry['options']
	return None, False

def _is_mount_point(path):
	"""True when path is a mount point, including a bind of a filesystem that path's parent also shows.

	os.path.ismount compares st_dev with the parent, so it misses a bind mounted from the
	same filesystem the directory lives on; mountinfo does not.
	"""
	path = path.rstrip(os.sep) or os.sep
	if os.path.ismount(path):
		return True
	real = os.path.realpath(path)
	return any(entry['mount_point'] == real for entry in _mountinfo_entries())

def _hidden_mounts(entries):
	"""Indexes (into entries) of the mounts path lookup does not reach, so the files at their mount point are not theirs.

	With mountinfo's ids: M is reached when its parent is not in the table, or is M itself (the kernel lists a
	namespace's root mount that way), or M is on a parent cycle (impossible on a real system; counted as reached, the
	side that never hides what may be visible). Else M is reached when its parent is reached and, if M is on another
	mount point than its parent, nothing else is stacked on the parent at the parent's mount point and no other child
	of the parent sits on M's mount point or a parent path of it below the parent's (a mount made over M's place).
	M is visible when it is reached and nothing is stacked on it at its own mount point. Resolved without recursion
	and with the children of each parent indexed by mount point (linear in the entries). An entry without ids (a
	test double) is never hidden and hides nothing."""
	def point(entry):
		return entry['mount_point'].rstrip(os.sep) or os.sep
	by_id = {entry['mount_id']: entry for entry in entries if 'mount_id' in entry}
	at = {}
	for entry in by_id.values():
		at.setdefault(entry['parent_id'], {}).setdefault(point(entry), []).append(entry)
	def covered(entry, parent):
		"""True when a mount made over entry's place sits on parent."""
		mounts = at.get(parent['mount_id'], {})
		parent_point = point(parent)
		if any(other is not entry and other is not parent for other in mounts.get(parent_point, ())):
			return True
		current = point(entry)
		while len(current) > len(parent_point) and current != os.path.dirname(current):
			if any(other is not entry for other in mounts.get(current, ())):
				return True
			current = os.path.dirname(current)
		return False
	memo = {}
	def reached(entry):
		chain, position, current = [], {}, entry
		while current['mount_id'] not in memo:
			position[current['mount_id']] = len(chain)
			chain.append(current)
			parent = by_id.get(current['parent_id'])
			if parent is None:
				memo[current['mount_id']] = True
				chain.pop()
				break
			if parent['mount_id'] in position:
				# A cycle (a self-parent included): every member from the parent on is reached.
				for node in chain[position[parent['mount_id']]:]:
					memo[node['mount_id']] = True
				del chain[position[parent['mount_id']]:]
				break
			current = parent
		for node in reversed(chain):
			parent = by_id[node['parent_id']]
			memo[node['mount_id']] = memo[parent['mount_id']] and (point(node) == point(parent) or not covered(node, parent))
		return memo[entry['mount_id']]
	hidden = set()
	for index, entry in enumerate(entries):
		if 'mount_id' not in entry:
			continue
		if not reached(entry) or any(other is not entry for other in at.get(entry['mount_id'], {}).get(point(entry), ())):
			hidden.add(index)
	return hidden

def _mount_scopes(path, entries):
	"""[(key prefix, fstype, source)] for path: first ('', ...) for the visible mount holding path's real path (the
	deepest), then one per visible mount below it, deepest last. A mount path lookup does not reach, or that another is
	stacked on, is left out (see _hidden_mounts), as on the system. fstype and source '' when no mount holds path.
	fstype is mountinfo's; see _scope_fs_type."""
	hidden = _hidden_mounts(entries)
	visible = [entry for index, entry in enumerate(entries) if index not in hidden]
	real = os.path.realpath(path)
	real_inside = real.rstrip(os.sep) + os.sep
	holder, depth, below = None, -1, []
	for entry in visible:
		inside = entry['mount_point'].rstrip(os.sep) + os.sep
		if real_inside.startswith(inside):
			if len(inside) >= depth:
				holder, depth = entry, len(inside)
		elif inside.startswith(real_inside):
			below.append((os.path.relpath(entry['mount_point'], real), entry['fstype'], entry.get('source', '')))
	below.sort(key=lambda scope: (scope[0].count('/'), scope[0]))
	own = ('', holder['fstype'], holder.get('source', '')) if holder else ('', '', '')
	return [own] + below

def _scope_fs_type(fs_type, source):
	"""fs_type, a fuseblk one (ntfs-3g, exFAT through FUSE) replaced by what its device holds (_probe_fs_type) when
	that is known. Probes only a block device, and only when asked: it may wake or stall a slow disk."""
	if (fs_type == 'fuseblk' or fs_type.startswith('fuseblk.')) and source.startswith('/dev/'):
		return _probe_fs_type(source) or fs_type
	return fs_type

def _rows_by_scope(rows, scopes):
	"""{key prefix: rows}: each row under the deepest scope prefix it lies in ('' for the target's own). Each row
	looks up its own name and parent names, deepest first, so the cost does not grow with the number of mounts."""
	prefixes = {prefix for prefix, _, _ in scopes if prefix}
	grouped = {}
	for row in rows:
		name = row.path.rstrip('/')
		while name and name not in prefixes:
			name = name.rpartition('/')[0]
		grouped.setdefault(name, []).append(row)
	return grouped

def _paths_on_fat(paths, root, entries):
	"""The paths (absolute, under root) on a FAT filesystem: root's own mount, or one mounted below it (_mount_scopes;
	hidden mounts left out). Each path looks up its own parent names, deepest first, as _rows_by_scope does. Paths may
	lie under root as given or under its real path (build_entry_map lists a symlinked root from its target)."""
	fat = {prefix: _is_fat(fs_type) for prefix, fs_type, _ in _mount_scopes(root, entries)}
	if not any(fat.values()):
		return set()
	roots = {os.path.abspath(root), os.path.realpath(root)}
	insides = [(top, top.rstrip(os.sep) + os.sep) for top in roots]
	result = set()
	for path in paths:
		name = ''
		for top, inside in insides:
			if path != top and path.startswith(inside):
				name = path[len(inside):].replace(os.sep, '/')
				break
		while name and name not in fat:
			name = name.rpartition('/')[0]
		if fat.get(name, False):
			result.add(path)
	return result

def _compare_metadata(roots, entries):
	"""False when any of roots sits on a filesystem whose modes and owners come from mount options (FAT, exFAT,
	NTFS): a single-filesystem patch between such trees compares content only (see states_equal)."""
	return all(_has_posix_metadata(_scope_fs_type(*_mount_scopes(root, entries)[0][1:])) for root in roots)

def _swap_devices(swaps_path='/proc/swaps'):
	"""Real paths of the active swap areas."""
	result = set()
	try:
		with open(swaps_path, encoding='utf-8', errors='surrogateescape') as f:
			next(f, None)
			for line in f:
				fields = line.split()
				if fields:
					result.add(os.path.realpath(_unescape_mount(fields[0])))
	except OSError:
		pass
	return result

def _partition_in_use(device, devnum, entries, swaps, sys_block_path='/sys/class/block'):
	"""Why device is in use ('' when it is not): mounted, active swap, or held by LVM / md / dm."""
	if any(_entry_matches(entry, devnum, device) for entry in entries):
		return 'mounted'
	if os.path.realpath(device) in swaps:
		return 'active swap'
	try:
		if os.listdir(os.path.join(sys_block_path, _sysfs_block_name(device), 'holders')):
			return 'held by another device'
	except OSError:
		pass
	return ''


_FAT_TYPES = ('fat32', 'fat16', 'fat12', 'fat', 'vfat', 'msdos')

def _is_fat(fs_type):
	"""True for the FAT family: local-time timestamps (the timezone tolerance) and full hashes (_FULL_HASH_MARK)."""
	return fs_type in _FAT_TYPES
# Filesystems whose file modes and owners come from mount options (umask=, uid=), not the disk.
_NO_POSIX_METADATA_TYPES = frozenset(_FAT_TYPES + ('exfat', 'ntfs', 'ntfs3'))
# Filesystems whose Linux drivers look names up case-insensitively (ntfs3 and ntfs-3g use the exact case;
# hfsplus folds case unless it is HFSX, and keeps owner, group and mode on disk).
_CASE_INSENSITIVE_TYPES = frozenset(_FAT_TYPES + ('exfat', 'hfs', 'hfsplus'))
def _has_posix_metadata(fs_type):
	"""False for a filesystem whose sections are compared and applied by content only."""
	return fs_type not in _NO_POSIX_METADATA_TYPES

def _is_hfsx(device):
	"""True when device holds case-sensitive HFSX: its HFS+ volume header signature (2 bytes at offset 1024) is 'HX'.
	False for 'H+' or an unreadable header."""
	try:
		with open(device, 'rb') as f:
			f.seek(1024)
			return f.read(2) == b'HX'
	except OSError:
		return False

def _has_case_insensitive_names(fs_type, device=''):
	"""True for a filesystem whose Linux driver looks names up case-insensitively: the FAT family, exFAT, HFS and
	HFS+, except case-sensitive HFSX (see _is_hfsx). Only a /dev/ path is read (a mount source such as 'none' would
	be opened relative to the working directory); an unreadable header counts as case-insensitive."""
	if fs_type not in _CASE_INSENSITIVE_TYPES:
		return False
	return not (fs_type == 'hfsplus' and device.startswith('/dev/') and _is_hfsx(device))

_IDENTITY_FIELDS = ('unique_guid', 'name', 'attrs', 'fs_uuid', 'fs_label')
IdentityChange = namedtuple('IdentityChange', 'number field current new')
# In a command's tool place, not a program: _run_identity_commands calls _write_fat_serial for it.
_FAT_SERIAL_WRITER = 'hpcp-write-fat-serial'

def _write_fat_serial(device, uuid):
	"""Set the volume ID of the FAT filesystem on device to uuid (blkid's XXXX-XXXX), as `fatlabel -i` does: dosfstools
	3.0.20 (EL7) and 4.1 (EL8) have no -i, so hpcp writes it on every system. Failures are reported (eprint), and
	the identity check after the write phase counts them.

	The ID is four little-endian bytes after the extended boot signature (0x28 or 0x29): at 39 on FAT12 and FAT16, at
	67 on FAT32, which is the layout with no 16-bit FAT size at 22. FAT32 repeats its boot sector in the reserved
	sectors, at the sector number at 50 (sector size at 11, reserved sectors at 14), and fsck.fat reports a copy that
	differs. Without the signature, those bytes are not a volume ID (DOS 3.x boot code starts there).
	"""
	if not re.fullmatch(r'[0-9A-Fa-f]{4}-?[0-9A-Fa-f]{4}', uuid):
		eprint(f'Identity change error: {uuid!r} is not a FAT volume ID (XXXX-XXXX); {device} is left as it is')
		return
	serial = struct.pack('<I', int(uuid.replace('-', ''), 16))
	try:
		fd = os.open(device, os.O_RDWR | getattr(os, 'O_CLOEXEC', 0))
		try:
			boot = os.pread(fd, 512, 0)
			fat32 = struct.unpack_from('<H', boot, 22)[0] == 0
			offset = 67 if fat32 else 39
			if boot[offset - 1] not in (0x28, 0x29):
				eprint(f'Identity change error: the FAT boot sector of {device} has no volume ID field (no extended boot signature); it is left as it is')
				return
			offsets = [offset]
			if fat32:
				sector_size, reserved = struct.unpack_from('<H', boot, 11)[0], struct.unpack_from('<H', boot, 14)[0]
				backup = struct.unpack_from('<H', boot, 50)[0]
				if 0 < backup < reserved:
					offsets.append(backup * sector_size + offset)
			for place in offsets:
				os.pwrite(fd, serial, place)
			os.fsync(fd)
		finally:
			os.close(fd)
	except OSError as e:
		eprint(f'Identity change error: cannot write the FAT volume ID of {device}: {e.strerror}')

def _fs_identity_commands(fs_type, device, uuid=None, label=None):
	"""argv lists that set a filesystem's UUID and / or label on an unmounted device.

	uuid / label None leave the value alone; label '' clears it. Returns None when fs_type
	cannot have a requested value set without recreating it. dd mode's delayed UUID
	commands (write_partition_info) come from this table too; a FAT volume ID is never one of them
	(mkfs.vfat -i sets it there), and only _run_identity_commands knows _FAT_SERIAL_WRITER.
	"""
	commands = []
	if uuid:
		if fs_type in ('ext2', 'ext3', 'ext4'):
			# tune2fs -U on a metadata_csum filesystem refuses unless it was freshly checked.
			commands += [_FS_FIX_COMMANDS[fs_type] + [device], ['tune2fs', '-U', uuid, device]]
		elif fs_type == 'xfs':
			commands.append(['xfs_admin', '-U', uuid, device])
		elif fs_type == 'btrfs':
			commands.append(['btrfstune', '-f', '-U', uuid, device])
		elif fs_type in _FAT_TYPES:
			commands.append([_FAT_SERIAL_WRITER, device, uuid])
		elif fs_type == 'exfat':
			# exfatlabel -i takes the serial as 0xXXXXXXXX; blkid shows it as XXXX-XXXX ("invalid serial number").
			commands.append(['exfatlabel', '-i', device, '0x' + uuid.replace('-', '')])
		elif fs_type == 'udf':
			commands.append(['udflabel', '--uuid', uuid, device])
		elif fs_type == 'jfs':
			commands.append(['jfs_tune', '-U', uuid, device])
		elif fs_type == 'reiserfs':
			commands.append(['reiserfstune', '-u', uuid, device])
		elif fs_type == 'swap':
			commands.append(['swaplabel', '-U', uuid, device])
		else:
			return None
	if label is not None:
		if fs_type in ('ext2', 'ext3', 'ext4'):
			commands.append(['e2label', device, label])
		elif fs_type == 'xfs':
			commands.append(['xfs_admin', '-L', label or '--', device])
		elif fs_type == 'btrfs':
			commands.append(['btrfs', 'filesystem', 'label', device, label])
		elif fs_type in _FAT_TYPES:
			commands.append(['fatlabel', device, label])
		elif fs_type == 'exfat':
			commands.append(['exfatlabel', device, label])
		elif fs_type == 'ntfs':
			commands.append(['ntfslabel', device, label])
		elif fs_type == 'f2fs':
			commands.append(['f2fslabel', device, label])
		elif fs_type == 'udf':
			commands.append(['udflabel', device, label])
		elif fs_type == 'jfs':
			commands.append(['jfs_tune', '-L', label, device])
		elif fs_type == 'reiserfs':
			commands.append(['reiserfstune', '-l', label, device])
		elif fs_type == 'swap':
			commands.append(['swaplabel', '-L', label, device])
		else:
			return None
	return commands

def _gpt_identity_commands(disk, number=None, unique_guid=None, name=None, attrs=None, disk_guid=None):
	"""sgdisk argv lists for a partition's unique GUID / name / attributes and the disk GUID."""
	commands = []
	if disk_guid:
		commands.append(['sgdisk', f'--disk-guid={disk_guid}', disk])
	if number is not None:
		if unique_guid:
			commands.append(['sgdisk', f'--partition-guid={number}:{unique_guid}', disk])
		if name is not None:
			commands.append(['sgdisk', f'--change-name={number}:{name}', disk])
		if attrs:
			commands.append(['sgdisk', f'--attributes={number}:=:{attrs}', disk])
	return commands

def _plan_identity(disk_rows, layout, in_use):
	"""Identity values the patch changes on this target: (changes, refusals, warnings).

	Values already new are skipped (re-runs). A target value that differs from the patch
	base is a warning, not a conflict: node identities may legitimately differ. A change on
	a partition that is in use is a refusal. The disk GUID only rewrites the GPT headers and
	is changed even on a live disk. Raw areas carry their own filesystem identity.
	"""
	kinds = {row.number: row.new for row in disk_rows if row.kind == 'part' and row.field == 'kind'}
	changes, refusals, warnings = [], [], []
	for row in disk_rows:
		if row.kind == 'disk':
			if row.field != 'guid':
				continue
			number, current = None, str(layout['disk'].get('guid', ''))
		else:
			if row.field not in _IDENTITY_FIELDS:
				continue
			if kinds.get(row.number) == 'raw' and row.field in ('fs_uuid', 'fs_label'):
				continue
			number = int(row.number)
			current = str(layout['parts'][number][row.field])
		if row.old == row.new or current == row.new:
			continue
		where = 'disk' if number is None else f'partition {number}'
		if current != row.old:
			warnings.append(f'{where} {row.field} is {current!r}, the patch base had {row.old!r}; setting {row.new!r}')
		if number is not None and in_use.get(number):
			refusals.append(f'partition {number} is {in_use[number]} and its {row.field} changes')
		changes.append(IdentityChange(number, row.field, current, row.new))
	return changes, refusals, warnings

def _identity_command_list(changes, layout, disk_device):
	"""(commands, unsupported) for planned identity changes.

	Filesystem UUID / label commands come first (per partition, UUID before label), then GPT
	and disk GUID commands. unsupported lists (number, fields, text) that cannot be set.
	"""
	fs_fields, gpt_commands = {}, []
	for change in changes:
		if change.number is None:
			gpt_commands += _gpt_identity_commands(disk_device, disk_guid=change.new)
		elif change.field in ('fs_uuid', 'fs_label'):
			fs_fields.setdefault(change.number, {})[change.field] = change.new
		else:
			gpt_commands += _gpt_identity_commands(disk_device, number=change.number, **{change.field: change.new})
	commands, unsupported = [], []
	for number, fields in sorted(fs_fields.items()):
		part = layout['parts'][number]
		fs_commands = _fs_identity_commands(part['fs_type'], part['device'], uuid=fields.get('fs_uuid'), label=fields.get('fs_label'))
		if fs_commands is None:
			names = tuple(sorted(fields))
			unsupported.append((number, names, f"partition {number} ({part['fs_type'] or 'unknown'}) cannot have its {' / '.join(names)} set"))
		else:
			commands += fs_commands
	return commands + gpt_commands, unsupported


def _attach_disk(path, read_only, loop_devices):
	"""Block device for path: itself when it is one, else a loop device over the disk-image file."""
	path = os.path.abspath(os.path.expanduser(path))
	if pathlib.Path(path).resolve().is_block_device():
		return os.path.realpath(path)
	if not os.path.isfile(path):
		raise RuntimeError(f'Invalid arguments: {path} is not a block device or a disk image file')
	loop_device = create_loop_device(path, read_only=read_only)
	loop_devices.append(loop_device)
	return loop_device

def _release_mount(mount_point, mount_points):
	"""Unmount one of this run's temp mounts and remove its directory (clean_up handles anything left)."""
	path = mount_point.rstrip(os.sep)
	if _is_mount_point(path):
		run_command_in_multicmd_with_path_check(['umount', path], strict=False)
	if not _is_mount_point(path):
		try:
			os.rmdir(path)
		except OSError:
			pass
		if path in mount_points:
			mount_points.remove(path)

def _mount_partition(device, read_only, mount_points, entries):
	"""Mount point (with a trailing separator) showing exactly device's filesystem, or None.

	A partition that is already mounted is bind-mounted (non-recursively) from its live mount,
	so filesystems mounted below it stay out of view; otherwise it is mounted directly
	(read-only with _ro_mount_options). Nothing is recorded in ERRORS: a partition the kernel
	cannot mount is an expected answer, and the caller decides what it means.
	"""
	live, _ = _live_mount(_device_number(device), entries, device)
	mount_point = tempfile.mkdtemp()
	mount_points.append(mount_point)
	error_count = len(ERRORS)
	if live:
		run_command_in_multicmd_with_path_check(['mount', '--bind', live, mount_point], quiet=True)
		if _is_mount_point(mount_point):
			# Private, so mounts made later under the live filesystem do not propagate into this bind.
			run_command_in_multicmd_with_path_check(['mount', '--make-private', mount_point], quiet=True)
		if read_only and _is_mount_point(mount_point):
			# Best effort: generation only reads through this bind mount.
			run_command_in_multicmd_with_path_check(['mount', '-o', 'remount,bind,ro', mount_point], quiet=True)
	elif read_only:
		run_command_in_multicmd_with_path_check(['mount', '-o', _ro_mount_options(device), device, mount_point], quiet=True)
	else:
		run_command_in_multicmd_with_path_check(_rw_mount_command(device, mount_point), quiet=True)
	del ERRORS[error_count:]
	if not _is_mount_point(mount_point):
		_release_mount(mount_point, mount_points)
		return None
	return mount_point + os.sep

def _live_excludes(root, device, paths, entries):
	"""Rooted exclude patterns (see _rooted_excludes) for the paths (this run's mount points and input disk images)
	that a live mount of device shows, as they lie in what hpcp's own mount of device at root shows.

	A live mount M of the device may show the whole filesystem, a btrfs subvolume (Fedora mounts / from subvolume
	root) or a bound directory: its mountinfo root R says which. A path P below M is R + P's path below M in the
	filesystem's own terms, mapped through the visible mount that holds it (the deepest mount point above it, of any
	device; hpcp's own mount is not recursive, so a path another mount holds is not on this device); it is excluded as its path below the root R0 hpcp's own mount shows (mountinfo root of
	root's mount), when it lies there. Entries of other devices, hpcp's own mount and this run's other mounts (the
	paths themselves) are not live mounts. Empty when root is not a mount point in entries."""
	root_real = os.path.realpath(root)
	own = next((entry for entry in entries if entry['mount_point'] == root_real), None)
	if own is None or not device:
		return set()
	own_root = own['root']
	devnum = _device_number(device)
	reals = [os.path.realpath(path) for path in paths]
	skip = set(reals) | {root_real}
	hidden = _hidden_mounts(entries)
	visible = [entry for index, entry in enumerate(entries) if index not in hidden]
	patterns = set()
	for real in reals:
		# The visible mount that holds the path (the deepest; any device): hpcp's own mount is not recursive, so a path
		# held by another mount is not on this device, and one held by a bind over a directory is that bind's.
		holder, depth = None, -1
		for entry in visible:
			inside = entry['mount_point'].rstrip(os.sep) + os.sep
			if real.startswith(inside) and len(inside) >= depth:
				holder, depth = entry, len(inside)
		if holder is None or holder['mount_point'] in skip or not _entry_matches(holder, devnum, device):
			continue
		inside = os.path.normpath(os.path.join(holder['root'], os.path.relpath(real, holder['mount_point'])))
		if own_root != os.sep and not inside.startswith(own_root.rstrip(os.sep) + os.sep):
			continue
		patterns.add(glob.escape(os.sep + os.path.relpath(inside, own_root)))
	return patterns

def _partition_changes(new_root, old_root, mount_points, devices, exclude, full_hash, max_workers, parallel_file_listing,
		input_files=(), metadata=True, fat=False, scan=None):
	"""Change rows of one filesystem partition (root to root; lost+found, this run's mounts and
	its input disk-image files left out). exclude is matched against '/' + the path inside the partition's filesystem
	(build_rooted_maps). metadata False: content-only rows (see states_equal).
	fat: a FAT section, its files hashed in full (see compute_changes).
	devices: (new device, old device) of the partition, whose live mounts may show this run's paths (see _live_excludes); () when there are none.
	scan: compute_changes' scan dict (see _rows_to_recheck).
	Names are compared by exact case in every section: a case-only rename is a delete plus an add."""
	internal = _inside_excludes(mount_points, (new_root, old_root))
	# A live mount of the partition (the whole filesystem, a btrfs subvolume or a bound directory) holds the (empty)
	# directories of this run's other temp mount points, and may hold the disk images this run compares (the old
	# golden image on the new golden). mountinfo is read now: hpcp's own mounts of both sides exist.
	entries = _mountinfo_entries() if devices else []
	for root, device in zip((new_root, old_root), devices):
		internal |= _live_excludes(root, device, list(mount_points) + list(input_files), entries)
	new_map, base_map = build_rooted_maps(new_root, old_root, frozenset(exclude or ()) | internal, max_workers, parallel_file_listing)
	new_map, base_map = _drop_lost_found(new_map), _drop_lost_found(base_map)
	if any(_is_reserved_key(key) for key in new_map):
		raise RuntimeError(f'Diff image error: {new_root} contains {DIFF_DIR_NAME}, which hpcp reserves for patch metadata')
	fat_paths = frozenset(new_map.values()) | frozenset(base_map.values()) if fat else frozenset()
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers, metadata=metadata, fat_paths=fat_paths, scan=scan)
	_validate_patch_rows(rows)
	return rows

def _refuse_raw_in_use(number, disks, entries, swaps):
	"""Raise 'Disk in use' when partition number, to be compared as a raw area, is in use on any
	of disks ([(name, layout)]): a raw area that changes while it is hashed and copied fails late,
	after a full copy. Swap partitions are identity-only and never come here."""
	for name, layout in disks:
		device = layout['parts'][number]['device']
		reason = _partition_in_use(device, _device_number(device), entries, swaps)
		if reason:
			raise RuntimeError(f'Disk in use: partition {number} of {name} is {reason}; a raw area cannot be copied while it is in use; generate from an offline copy')

def _is_identity_change(row, kinds):
	"""A disk.tsv row whose value the patch would set on a target."""
	if row.old == row.new:
		return False
	if row.kind == 'disk':
		return row.field == 'guid'
	if row.field not in _IDENTITY_FIELDS:
		return False
	return not (kinds.get(int(row.number)) == 'raw' and row.field in ('fs_uuid', 'fs_label'))

def make_disk_patch_meta(sections, raws, new_name, base_name, full_hash=False):
	"""Manifest fields for a disk patch; sections = [(number, rows)], raws = [(name, file, old, new, size)]."""
	meta = make_patch_meta([row for _, rows in sections for row in rows], new_name, base_name, 'disk', full_hash)
	# Version 3 when a section's rows carry the full-hash mark (see make_patch_meta), else the disk version 2.
	if meta['format_version'] != _FULL_HASH_FORMAT_VERSION:
		meta['format_version'] = _DISK_FORMAT_VERSION
	meta['sections'] = [(f'part{number}', f'part{number}', f'part{number}.changes.tsv') for number, _ in sections]
	meta['raws'] = list(raws)
	meta['disk'] = 'disk.tsv'
	meta['raw_bytes'] = sum(raw[4] for raw in raws)
	return meta

def write_disk_patch_metadata(volume_roots, sections, disk_rows, meta):
	"""Write every change list and disk.tsv on every volume first, then each volume's manifest (volume k/n)."""
	total = len(volume_roots)
	for volume_root in volume_roots:
		diff_dir = os.path.join(volume_root, DIFF_DIR_NAME)
		_makedirs(diff_dir, exist_ok=True)
		for number, rows in sections:
			write_changes(os.path.join(diff_dir, f'part{number}.changes.tsv'), rows)
		write_disk_tsv(os.path.join(diff_dir, meta['disk']), disk_rows)
	for index, volume_root in enumerate(volume_roots, start=1):
		volume_meta = dict(meta)
		volume_meta['volume'] = f'{index}/{total}'
		write_manifest(os.path.join(volume_root, DIFF_DIR_NAME, 'manifest'), volume_meta)

def _store_raw(volume_roots, raw_file, device, size, new_sha):
	"""Copy a raw area (the first size bytes of device) into raw_file on a volume with room."""
	for volume_root in volume_roots:
		if shutil.disk_usage(volume_root).free > size + (1 << 20):
			dest = os.path.join(volume_root, raw_file)
			_makedirs(os.path.dirname(dest), exist_ok=True)
			if _copy_area(device, dest, 0, size) != new_sha:
				raise RuntimeError(f'Raw area error: {raw_file} changed while it was copied; patch metadata not written')
			return dest
	raise RuntimeError(f'Diff image error: no patch volume has room for {raw_file} ({format_bytes(size)}B); use a larger -dis')

def _check_raw_fits_a_volume(raw_sizes, init_size, dest_image_size):
	"""Raise 'Diff image error: ...' when the patch needs more than one volume (_plan_dest_images for init_size)
	and its largest raw area, which must fit whole in one volume, is larger than one volume's usable space."""
	if not raw_sizes:
		return
	count, volume_size = _plan_dest_images(init_size, dest_image_size)
	largest_raw = max(raw_sizes)
	# Same usable-space estimate as _plan_dest_images uses for split volumes.
	if count > 1 and largest_raw > int(volume_size * 0.90) - (16 << 20):
		needed = _align_up(int((largest_raw + (16 << 20)) / 0.90) + 1)
		# Whole MiB, rounded up, in a form the -dis parser reads exactly ('M' alone is 10^6 bytes).
		needed_mib = (needed + (1 << 20) - 1) >> 20
		raise RuntimeError(f'Diff image error: a raw area of {format_bytes(largest_raw)}B must fit in one patch volume; use -dis {needed_mib}MiB or larger')

def process_get_disk_diff_image(src_paths, dest_paths, output, mount_points, loop_devices, exclude=None,
		full_hash=False, dest_image_size=0, max_workers=multiprocessing.cpu_count(), parallel_file_listing=False, verbose=False):
	"""Write a disk patch image that turns a disk laid out like the old disk (-d) into the new disk.

	Both disks are GPT block devices or disk-image files and are only read. Returns the
	patch path, or None when the disks do not differ (no image is created).
	"""
	_refuse_root_exclude(exclude)
	sources, bases = expand_paths(src_paths or []), expand_paths(dest_paths or [])
	if len(sources) != 1 or len(bases) != 1:
		raise RuntimeError(f'Invalid arguments: -dd -diff needs exactly one new disk and one old disk (-d), got {src_paths} and {dest_paths}')
	new_name, base_name = os.path.basename(sources[0]), os.path.basename(bases[0])
	if not output or output == 'auto':
		output = f'DIFF_{new_name}_TO_{base_name}_{int(time.time())}.img'
	output = os.path.abspath(os.path.expanduser(output))
	if os.path.lexists(output):
		raise RuntimeError(f'Diff image error: output {output} already exists')
	new_disk = _attach_disk(sources[0], True, loop_devices)
	old_disk = _attach_disk(bases[0], True, loop_devices)
	new_layout, old_layout = read_disk_layout(new_disk, name=sources[0]), read_disk_layout(old_disk, name=bases[0])
	problems, _ = compare_disk_layouts(old_layout, new_layout)
	if problems:
		raise RuntimeError('Disk layout mismatch: ' + '; '.join(problems) + '; re-clone instead')
	# Disk-image files (not block devices) that may lie on a live partition being compared.
	input_files = [os.path.realpath(path) for path in (sources[0], bases[0]) if os.path.isfile(path)]
	entries = _mountinfo_entries()
	swaps = _swap_devices()
	disks = ((sources[0], new_layout), (bases[0], old_layout))
	for number in sorted(new_layout['parts']):
		if _default_kind(new_layout['parts'][number]) == 'raw':
			_refuse_raw_in_use(number, disks, entries, swaps)
	kinds, sections, raws = {}, [], []

	def add_raw(number):
		"""Hash partition number's area on both disks; a changed one becomes a raw entry."""
		new_part, old_part = new_layout['parts'][number], old_layout['parts'][number]
		if new_part['size'] != old_part['size']:
			raise RuntimeError(f"Disk layout mismatch: partition {number} is a raw area and its size changed ({old_part['size']} vs {new_part['size']}); re-clone instead")
		old_sha, new_sha = _hash_area(old_part['device'], 0, old_part['size']), _hash_area(new_part['device'], 0, new_part['size'])
		if old_sha != new_sha:
			raws.append((f'part{number}', new_part['device'], new_part['size'], old_sha, new_sha))

	numbers = sorted(new_layout['parts'])
	raw_numbers = [number for number in numbers if _default_kind(new_layout['parts'][number]) == 'raw']
	# Raw areas first, before any filesystem section is diffed (compare_disk_layouts already refused one whose size
	# changed, before anything was hashed): an area too large for -dis is refused before the sections' file diffs when
	# the raw areas alone force more than one volume (otherwise below, once the sections' payload is known).
	for number in raw_numbers:
		add_raw(number)
		kinds[number] = 'raw'
	_check_raw_fits_a_volume([raw[2] for raw in raws], sum(_align_up(raw[2]) for raw in raws) + 64 * 1024, dest_image_size)
	for number in numbers:
		if number in kinds:
			continue
		new_part, old_part = new_layout['parts'][number], old_layout['parts'][number]
		kind = _default_kind(new_part)
		if kind == 'fs':
			new_root = _mount_partition(new_part['device'], True, mount_points, entries)
			old_root = _mount_partition(old_part['device'], True, mount_points, entries)
			if new_root and old_root:
				scan = {}
				rows = _partition_changes(new_root, old_root, mount_points, (new_part['device'], old_part['device']), exclude, full_hash, max_workers, parallel_file_listing,
					input_files, metadata=_has_posix_metadata(new_part['fs_type']),
					fat=_is_fat(new_part['fs_type']) or _is_fat(old_part['fs_type']), scan=scan)
				_release_mount(old_root, mount_points)
				if rows:
					# Kept mounted: its payload is copied once the patch image exists.
					sections.append((number, new_root, rows, scan))
				else:
					_release_mount(new_root, mount_points)
			else:
				for root in (new_root, old_root):
					if root:
						_release_mount(root, mount_points)
				failed = ' and '.join(side for side, root in (('new', new_root), ('old', old_root)) if not root)
				print(f"Warning: partition {number} ({new_part['fs_type']}) could not be mounted on the {failed} side; comparing it as a raw area", file=sys.stderr)
				_refuse_raw_in_use(number, disks, entries, swaps)
				kind = 'raw'
				add_raw(number)
		kinds[number] = kind
	# Fallback raw areas were added after the up-front ones; keep the raws in partition order.
	raws.sort(key=lambda raw: int(raw[0][4:]))
	old_mbr, new_mbr = _hash_area(old_disk, 0, _MBR_BOOTCODE_SIZE), _hash_area(new_disk, 0, _MBR_BOOTCODE_SIZE)
	if old_mbr != new_mbr:
		raws.append(('mbr', new_disk, _MBR_BOOTCODE_SIZE, old_mbr, new_mbr))
	_, disk_rows = compare_disk_layouts(old_layout, new_layout, kinds)
	for row in disk_rows:
		if row.field == 'fs_params' and row.old != row.new:
			print(f"Warning: partition {row.number} was created with different mkfs parameters ({row.old} -> {row.new}); a patch cannot change them", file=sys.stderr)
	identity = [row for row in disk_rows if _is_identity_change(row, kinds)]
	raw_entries = [(name, 'raw/mbr-bootcode.bin' if name == 'mbr' else f'raw/{name}.img', old_sha, new_sha, size)
		for name, _, size, old_sha, new_sha in raws]
	section_rows = [(number, rows) for number, _, rows, _ in sections]
	meta = make_disk_patch_meta(section_rows, raw_entries, new_name, base_name, full_hash)
	print(f"Disk changes: {len(sections)} filesystem section(s), {len(raws)} raw area(s), {len(identity)} identity value(s); "
		f"{meta['added']} added, {meta['modified']} modified, {meta['deleted']} deleted; "
		f"payload {format_bytes(meta['payload_bytes'])}B, raw {format_bytes(meta['raw_bytes'])}B")
	if not sections and not raws and not identity:
		print(f"No differences between {sources[0]} and {bases[0]}; no patch image created.")
		return None
	init_size = (sum(_payload_allocation(rows) for _, rows in section_rows) + sum(_align_up(raw[2]) for raw in raws)
		+ 64 * 1024 * (len(sections) + 1))
	count, volume_size = _plan_dest_images(init_size, dest_image_size)
	# Again with every section's payload: sections can force the split the raw areas alone did not.
	_check_raw_fits_a_volume([raw[2] for raw in raws], init_size, dest_image_size)
	if os.geteuid() != 0:
		print("Warning: creating a patch image usually needs root privileges (loop devices, mkfs, mount)", file=sys.stderr)
	target_mount_point = tempfile.mkdtemp()
	mount_points.append(target_mount_point)
	_claim_volume_names(output, count)
	volume_roots = create_image(output, target_mount_point, loop_devices, [], mount_points, max_workers=max_workers,
		dest_image_size=dest_image_size, init_size=init_size)
	# Raw areas first, largest first: each must fit whole in one volume, and section payloads
	# copied before them would fill the first volume.
	for (name, device, size, old_sha, new_sha), entry in sorted(zip(raws, raw_entries), key=lambda pair: (-pair[0][2], pair[0][0])):
		_store_raw(volume_roots, entry[1], device, size, new_sha)
	for number, new_root, rows, scan in sections:
		section_roots = [os.path.join(volume_root, f'part{number}') for volume_root in volume_roots]
		for root in section_roots:
			_makedirs(root, exist_ok=True)
		_build_payload(rows, new_root, section_roots, max_workers, verbose, _payload_bytes(rows), full_hash, scan=scan)
	write_disk_patch_metadata(volume_roots, section_rows, disk_rows, meta)
	extra = f" (+{len(volume_roots) - 1} more volume(s))" if len(volume_roots) > 1 else ''
	print(f"Disk patch image written to {output}{extra}")
	return output

def _check_disk_entries(meta, disk_rows):
	"""Raise RuntimeError('Patch format error: ...') unless the manifest's sections and raw areas
	agree with disk.tsv.

	Patch images are untrusted input: a section must be a partition of kind fs, a raw partition
	area a partition of kind raw of exactly its recorded size, and the mbr area exactly the boot
	code (a longer one would write into the partition table). No partition may change its kind.
	"""
	parts = {}
	for row in disk_rows:
		if row.kind == 'part':
			parts.setdefault(row.number, {})[row.field] = row
	for number, fields in sorted(parts.items(), key=lambda item: int(item[0])):
		kind = fields.get('kind')
		if kind is not None and kind.old != kind.new:
			raise RuntimeError(f'Patch format error: the disk table changes the kind of partition {number} ({kind.old} -> {kind.new})')

	def partition(entry, name, wanted, label):
		"""disk.tsv fields of the partition entry name refers to, which must be of kind wanted."""
		number = name[len('part'):]
		if number not in parts:
			raise RuntimeError(f'Patch format error: {entry} {name} is for partition {number}, which the disk table does not list')
		kind = parts[number]['kind'].old if 'kind' in parts[number] else 'no kind'
		if kind != wanted:
			raise RuntimeError(f'Patch format error: {entry} {name} is for partition {number}, which the disk table records as {kind}, not {label}')
		return number, parts[number]

	for name, _, _ in meta['sections']:
		partition('section', name, 'fs', 'a filesystem')
	for name, _, _, _, size in meta['raws']:
		if name == 'mbr':
			if size != _MBR_BOOTCODE_SIZE:
				raise RuntimeError(f'Patch format error: raw area mbr is {size} bytes; the MBR boot code area is {_MBR_BOOTCODE_SIZE} bytes')
			continue
		number, fields = partition('raw area', name, 'raw', 'a raw area')
		recorded = fields['size'].old if 'size' in fields else 'no size'
		if str(size) != recorded or 'size' in fields and fields['size'].new != recorded:
			raise RuntimeError(f'Patch format error: raw area {name} is {size} bytes; the disk table records {recorded} for partition {number}')

def _refuse_if_image_in_use(image):
	"""Refuse an image-file target that is already attached to any loop device.

	A second read-write loop over the same file would leave the first one's page cache stale, and
	swap or a dm/LVM/md holder on the first one would not be seen as in use.
	"""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['losetup', '-j', image], quiet=True, extra_env=_C_LOCALE)
	finally:
		del ERRORS[error_count:]
	for line in output:
		loop = line.partition(':')[0].strip()
		if not loop:
			continue
		raise RuntimeError(f'Disk in use: {image} is attached as {loop}; detach it (losetup -d {loop}) or apply the patch to {loop}')

_BLKRRPART = 0x125f
# Identity tools that fail while hpcp holds their partition O_EXCL (found by
# test_identity_tools_work_while_the_partition_is_held); they run with that partition's hold dropped.
_TOOLS_NEEDING_EXCLUSIVE = frozenset(('e2fsck', 'btrfstune', 'btrfs', 'exfatlabel', 'udflabel', 'f2fslabel'))

class DiskHoldError(OSError):
	"""The write phase could not lock the disk or hold a partition (see _disk_write_hold)."""

def _exclusive_open(device):
	"""A read-only O_EXCL descriptor of block device device. Raises OSError (EBUSY) when it is mounted, active swap or
	held (LVM, md, dm, another exclusive opener) in any mount namespace: the kernel answers for all of them."""
	return os.open(device, os.O_RDONLY | os.O_EXCL | getattr(os, 'O_CLOEXEC', 0))

def _exclusive_refusals(devices):
	"""'partition N is ...' for each {number: device} that cannot be opened exclusively right now (opened, then
	closed at once)."""
	refusals = []
	for number, device in sorted(devices.items()):
		try:
			os.close(_exclusive_open(device))
		except OSError as e:
			if e.errno == errno.EBUSY:
				refusals.append(f'partition {number} is mounted, active swap or held (LVM, md, dm), in this or another mount namespace')
			else:
				refusals.append(f'partition {number} cannot be opened exclusively ({e.strerror})')
	return refusals

def _write_phase_devices(raws, changes, layout, skipped_identity):
	"""{number: device} of the partitions the write phase may write: every raw partition with an entry in the patch
	(raws: the manifest's raw entries; whether an area is already done is known only after hashing, and the
	exclusive check runs before any hash), and every partition whose filesystem UUID or label the patch sets. GPT
	fields and the MBR boot code are written through the whole disk, which the write phase locks."""
	devices = {}
	for name, _, _, _, _ in raws:
		if name != 'mbr':
			number = int(name[len('part'):])
			devices[number] = layout['parts'][number]['device']
	for change in changes:
		if change.number is not None and change.field in ('fs_uuid', 'fs_label') and (change.number, change.field) not in skipped_identity:
			devices[change.number] = layout['parts'][change.number]['device']
	return devices

def _reread_partition_table(disk_fd, disk):
	"""Ask the kernel to re-read disk's partition table (BLKRRPART on disk_fd; `partx -u` when the kernel refuses, a
	partition being in use). Quiet: udev settling afterwards shows the outcome."""
	try:
		import fcntl
		fcntl.ioctl(disk_fd, _BLKRRPART)
		return
	except (ImportError, OSError):
		pass
	if check_path('partx'):
		error_count = len(ERRORS)
		try:
			run_command_in_multicmd_with_path_check(['partx', '-u', disk], quiet=True, strict=False)
		finally:
			del ERRORS[error_count:]

@contextlib.contextmanager
def _disk_write_hold(disk, devices, timeout=10):
	"""Hold disk for the write phase; yields {device: descriptor} for devices (None once dropped, see
	_run_identity_commands).

	flock(LOCK_EX) on the whole disk (systemd's block-device locking convention: udev postpones its events for a disk
	locked so, and so does not re-read the partition table mid-phase), retried for up to timeout seconds while udev
	holds its shared lock; then an O_EXCL open of every device, kept to the end, so nothing can mount or claim them
	meanwhile. Raises DiskHoldError when disk cannot be opened or the lock or a hold cannot be taken (what was taken
	is released). On exit the holds are closed first (the kernel cannot re-read a table whose partitions are open
	exclusively), then the partition table is re-read, only when the lock was taken (another tool's lock means the
	table must be left alone), then the lock released (even when the re-read raises).
	"""
	import fcntl
	try:
		disk_fd = os.open(disk, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
	except OSError as e:
		raise DiskHoldError(e.errno, f'cannot open {disk} ({e.strerror})')
	holds = {}
	locked = False
	try:
		deadline = time.time() + timeout
		while True:
			try:
				fcntl.flock(disk_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
				locked = True
				break
			except OSError as e:
				if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN) or time.time() >= deadline:
					raise DiskHoldError(e.errno, f'cannot lock {disk} ({e.strerror})')
				time.sleep(0.1)
		for device in devices:
			try:
				holds[device] = _exclusive_open(device)
			except OSError as e:
				raise DiskHoldError(e.errno, f'{device} could not be held ({e.strerror})')
		yield holds
	finally:
		try:
			for fd in holds.values():
				if fd is not None:
					os.close(fd)
			if locked:
				_reread_partition_table(disk_fd, disk)
		finally:
			# Closing the descriptor releases the lock.
			os.close(disk_fd)

def _run_identity_commands(commands, holds):
	"""Run the identity commands while the write phase holds their partitions. A tool in _TOOLS_NEEDING_EXCLUSIVE
	opens its device exclusively itself: the hold of every held device its command names is dropped for that one
	command and taken again after it. Raises DiskHoldError when a hold cannot be taken again: the remaining commands
	do not run."""
	for command in commands:
		dropped = []
		if command[0] in _TOOLS_NEEDING_EXCLUSIVE:
			for device, fd in list(holds.items()):
				if fd is not None and device in command:
					os.close(fd)
					holds[device] = None
					dropped.append(device)
		if command[0] == _FAT_SERIAL_WRITER:
			_write_fat_serial(*command[1:])
		else:
			run_command_in_multicmd_with_path_check(command, strict=False)
		for device in dropped:
			try:
				holds[device] = _exclusive_open(device)
			except OSError as e:
				raise DiskHoldError(e.errno, f'{device} could not be held again after {command[0]} ({e.strerror})')

def apply_disk_patch(volume_roots, target, mount_points, loop_devices, force=False, max_workers=1, verbose=False, volume_names=None):
	"""Apply a disk patch (layout disk) to target: a GPT block device or disk-image file.

	Everything is checked before anything is written (the filesystem sections are mounted read-write before the raw
	areas are hashed, as before planning): the layout, which partitions are in use, every section's paths, the raw
	areas and the identity values. Every refusal that needs no content hash (in use, read-only live mounts, identity
	changes that cannot be made, the exclusive check, missing raw payloads, sections that cannot be mounted) comes
	before the first raw area is read: a raw area can be gigabytes. Every raw partition with an entry in the patch
	(including one whose area turns out to be already done) and every partition the patch sets a UUID / label on
	must open exclusively (O_EXCL) during planning, or nothing is applied (Disk in use): that also catches
	mounts in another mount namespace. During the raw writes and identity commands the whole disk is flock'ed and
	those partitions held exclusively; a hold that cannot be taken then (or again, after a tool that needs the
	partition to itself) stops the raw areas, boot code and identity changes not yet done, like a failed section.
	Filesystem sections are
	applied first (partition order, ESP last), then raw partition areas, the MBR boot code
	and identity changes; then everything is verified. A section that fails stops the apply
	there: later sections, raw areas, boot code and identity changes are left for a re-run.
	So does a raw area that cannot be written, or that does not hold the new content when read back
	right after its write: later raw areas, the boot code and identity changes.
	After the sections, a write one section skipped as already done that a delete in another took away
	(through a symlinked directory) is restored (see _restore_across_sections), and what one section
	or a restore changed in another section is verified again (see _verify_across_sections); a failure
	there stops the raw areas, boot code and identity changes like a failed section.
	Returns {'sections', 'raw_written', 'raw_skipped', 'identity_changed', 'forced', 'failed'}.
	"""
	meta, sections, volume_roots = _load_patch(volume_roots, volume_names)
	if meta.get('layout') != 'disk':
		raise RuntimeError('Invalid arguments: this is a single-filesystem patch; apply it to a directory or a filesystem image')
	disk_rows = read_disk_tsv(os.path.join(volume_roots[0], DIFF_DIR_NAME, meta['disk']))
	_check_disk_entries(meta, disk_rows)
	fs_types = {row.number: row.old for row in disk_rows if row.kind == 'part' and row.field == 'fs_type'}
	target = os.path.abspath(os.path.expanduser(target))
	entries = _mountinfo_entries()
	if not pathlib.Path(target).resolve().is_block_device():
		_refuse_if_image_in_use(target)
	disk = _attach_disk(target, False, loop_devices)
	pttype = _partition_table_type(disk)
	if not pttype:
		raise RuntimeError(f'Invalid arguments: {_no_partition_table(target)}; a disk patch needs a whole GPT disk (block device or disk image)')
	# fs_params False: nothing in apply reads them, and on EL7 their xfs probe mounts the partition
	layout = read_disk_layout(disk, name=target, pttype=pttype, fs_params=False)
	problems = match_target_layout(disk_rows, layout)
	if problems:
		raise RuntimeError('Disk layout mismatch: ' + '; '.join(problems) + '; nothing applied')
	swaps = _swap_devices()
	in_use = {number: _partition_in_use(part['device'], _device_number(part['device']), entries, swaps)
		for number, part in layout['parts'].items()}
	changes, refusals, warnings = _plan_identity(disk_rows, layout, in_use)
	# Every refusal that needs no content hash comes before the first one: a raw area can be gigabytes.
	raw_refusals = []
	for name, _, _, _, _ in meta['raws']:
		number = None if name == 'mbr' else int(name[len('part'):])
		if number is not None and in_use.get(number):
			# Refused without reading it: a live area (an LVM PV, say) can be hundreds of GB. So whether it
			# already holds the new content (a live re-run) is not known either.
			raw_refusals.append(f'partition {number} is {in_use[number]}; this patch carries its raw area, '
				'which cannot be checked or written while it is in use')
	for name, _, _ in sections:
		part = layout['parts'][int(name[len('part'):])]
		live, live_read_only = _live_mount(_device_number(part['device']), entries, part['device'])
		if live and live_read_only:
			refusals.append(f'partition {name[len("part"):]} is mounted read-only at {live}')
	if raw_refusals:
		done = 'it if it is' if len(raw_refusals) == 1 else 'each one that is'
		raise RuntimeError('Disk in use: ' + '; '.join(refusals + raw_refusals)
			+ f'; apply the patch offline (an offline apply skips {done} already done)')
	if refusals:
		raise RuntimeError('Disk in use: ' + '; '.join(refusals) + '; apply this patch offline')
	commands, unsupported = _identity_command_list(changes, layout, disk)
	if unsupported and not force:
		raise RuntimeError('Identity change error: ' + '; '.join(text for _, _, text in unsupported) + '; nothing applied (use -fa to apply the rest)')
	for _, _, text in unsupported:
		print(f'Warning: {text}; skipped', file=sys.stderr)
	for text in warnings:
		print(f'Warning: {text}', file=sys.stderr)
	skipped_identity = {(number, field) for number, fields, _ in unsupported for field in fields}
	# The kernel answers for every mount namespace and for hidden mounts, which _partition_in_use cannot see.
	write_devices = _write_phase_devices(meta['raws'], changes, layout, skipped_identity)
	exclusive = _exclusive_refusals(write_devices)
	if exclusive:
		raise RuntimeError('Disk in use: ' + '; '.join(exclusive) + '; nothing applied')
	for _, raw_file, _, _, size in meta['raws']:
		if not _raw_payload_present(volume_roots, raw_file, size):
			raise RuntimeError(f'Incomplete patch: raw payload {raw_file} is missing or damaged; nothing applied')
	with _patch_hash_settings(meta) as (compare_hash, full_hash):
		roots = {}
		for name, _, _ in sections:
			part = layout['parts'][int(name[len('part'):])]
			root = _mount_partition(part['device'], False, mount_points, entries)
			if root is None:
				raise RuntimeError(f"Disk layout mismatch: partition {name[len('part'):]} ({part['fs_type']}) cannot be mounted read-write; nothing applied")
			roots[name] = root
		# Only now the raw areas are read: every refusal that needs no hash has run.
		raw_plan = []
		for name, raw_file, old_sha, new_sha, size in meta['raws']:
			number = None if name == 'mbr' else int(name[len('part'):])
			device = disk if number is None else layout['parts'][number]['device']
			found = _hash_area(device, 0, size)
			action = _plan_raw(old_sha, new_sha, found)
			payload = None
			if action != 'skip':
				payload = _raw_payload(volume_roots, raw_file, new_sha, size)
				if payload is None:
					raise RuntimeError(f'Incomplete patch: raw payload {raw_file} is missing or damaged; nothing applied')
			raw_plan.append({'name': name, 'device': device, 'size': size, 'payload': payload, 'action': action,
				'found': found, 'new': new_sha})
		plans = []
		for name, payload_dir, rows in sections:
			part = layout['parts'][int(name[len('part'):])]
			fs_type = fs_types.get(name[len('part'):], '')
			plans.append(_plan_section(name, rows, _section_volume_roots(volume_roots, payload_dir), roots[name], compare_hash, full_hash, max_workers,
				_has_posix_metadata(fs_type), _has_case_insensitive_names(fs_type, part['device']), tz_tolerant=_is_fat(fs_type)))
		raw_conflicts = [entry for entry in raw_plan if entry['action'] == 'conflict']
		extra = [(f"raw {entry['name']}: content matches neither the patch base nor the new state (found {entry['found'][:12]})",
			f"raw {entry['name']}: content differs") for entry in raw_conflicts]
		for plan in plans:
			_preflight_section(plan)
		_resolve_section_conflicts(plans, force, extra)
		summaries = {}
		ran = []
		stopped = None
		order = sorted(plans, key=lambda plan: (layout['parts'][int(plan.name[len('part'):])]['type_guid'].upper() == _ESP_TYPE_GUID,
			int(plan.name[len('part'):])))
		for plan in order:
			# A failed section (ENOSPC on the root, say) stops everything after it: the ESP, raw areas,
			# boot code and identity changes could otherwise leave the node unbootable.
			if stopped is None:
				outcome = {}
				summaries[plan.name] = _apply_section(plan, max_workers, verbose, full_hash, compare_hash, outcome)
				ran.append((plan, summaries[plan.name], outcome))
				if summaries[plan.name]['failed']:
					stopped = plan.name
		# While every section is still mounted, before any raw area is written. A restore or a final verify that
		# fails stops the raw areas, boot code and identity changes like a failed section.
		_restore_across_sections(ran, max_workers, verbose, full_hash, compare_hash)
		_verify_across_sections(ran, max_workers, full_hash, compare_hash)
		if stopped is None:
			stopped = next((plan.name for plan, summary, _ in ran if summary['failed']), None)
		# Last mount first (see clean_up): order is the order they were applied in, plans the order they were mounted in.
		for plan in reversed(plans):
			# A target that is live (mounted by someone else) is only bound here, and taking the bind off writes nothing; one hpcp
			# mounted itself is written by its unmount.
			_syncfs(plan.target_root)
			_release_mount(plan.target_root, mount_points)
	failed = sum(summary['failed'] for summary in summaries.values())
	pending = [entry for entry in raw_plan if entry['action'] != 'skip'] if stopped is None else []
	attempted = []
	raw_stopped = None
	hold_error = None
	if stopped is None and (pending or commands):
		try:
			# The disk is locked (udev leaves it alone) and every partition written here is held exclusively
			# (nothing can mount it) from the first raw write to the last identity command.
			with _disk_write_hold(disk, sorted(write_devices.values())) as holds:
				# Partition areas first, the MBR boot code last. A write that fails stops the rest and the identity
				# changes: boot code pointing at a core image that is not there would leave the node unbootable.
				# So does one that raised no error but does not hold the new content when read back past the page cache.
				for entry in sorted(pending, key=lambda entry: entry['name'] == 'mbr'):
					attempted.append(entry)
					try:
						_write_area(entry['device'], entry['payload'], 0, entry['size'])
						if _hash_area(entry['device'], 0, entry['size'], direct=True) != entry['new']:
							entry['error'] = 'does not hold the new content after writing'
					except OSError as e:
						entry['error'] = str(e)
					if entry.get('error'):
						raw_stopped = entry['name']
						break
				if raw_stopped is None:
					_run_identity_commands(commands, holds)
		except DiskHoldError as e:
			hold_error = e.strerror
			failed += 1
			# Raw areas and earlier identity commands may be done already; a re-run skips them.
			eprint(f'Disk in use: {hold_error}; the write phase stopped there: the raw areas, boot code and identity changes '
				'not yet done were not applied; fix the cause and re-run')
	if stopped is None:
		# The holds are released and the partition table re-read: wait until udev shows every partition again.
		_settle_partitions(disk, sorted(layout['parts']))
	# Verify after the identity commands: sgdisk rewriting the GPT must not have undone a raw write.
	written = 0
	for entry in attempted:
		reason = entry.get('error')
		if not reason:
			try:
				if _hash_area(entry['device'], 0, entry['size'], direct=True) == entry['new']:
					written += 1
					continue
				reason = 'does not hold the new content after writing'
			except OSError as e:
				reason = str(e)
		failed += 1
		eprint(f"Raw area error: {entry['name']}: {reason}")
	changed = 0
	applied = ([change for change in changes if (change.number, change.field) not in skipped_identity]
		if stopped is None and raw_stopped is None and hold_error is None else [])
	if applied:
		after = read_disk_layout(disk, pttype=pttype, fs_params=False)
		for change in applied:
			current = str(after['disk'].get('guid', '')) if change.number is None else str(after['parts'][change.number][change.field])
			if current == change.new:
				changed += 1
			else:
				failed += 1
				where = 'disk' if change.number is None else f'partition {change.number}'
				eprint(f'Identity change error: {where} {change.field} is {current!r}, expected {change.new!r}')
	for name in sorted(summaries, key=lambda name: int(name[len('part'):])):
		item = summaries[name]
		print(f"Section {name}: deleted {item['deleted']}, added {item['added']}, modified {item['modified']}, "
			f"skipped {item['skipped']}, restored {item.get('restored', 0)}, forced {item['forced']}, failed {item['failed']}")
	summary = {
		'sections': summaries,
		'raw_written': written,
		'raw_skipped': sum(1 for entry in raw_plan if entry['action'] == 'skip'),
		'identity_changed': changed,
		'forced': sum(summary['forced'] for summary in summaries.values()) + sum(1 for entry in attempted if entry['action'] == 'conflict'),
		'failed': failed,
	}
	print(f"Disk patch summary: {len(summaries)} section(s), added {sum(item['added'] for item in summaries.values())}, "
		f"modified {sum(item['modified'] for item in summaries.values())}, deleted {sum(item['deleted'] for item in summaries.values())}, raw written {written}, raw skipped {summary['raw_skipped']}, "
		f"identity changed {changed}, forced {summary['forced']}, failed {failed}")
	if stopped is not None:
		# Same category as the section's own failures (_verify_rows); a re-run skips what is done.
		later = 'later sections, raw areas' if len(summaries) < len(plans) else 'raw areas'
		eprint(f'Patch verify error: stopping: section {stopped} failed; {later}, boot code and identity changes '
			f'were not applied; fix the cause and re-run')
	elif raw_stopped is not None:
		later = 'the identity changes were' if raw_stopped == 'mbr' else 'later raw areas, the boot code and identity changes were'
		eprint(f'Raw area error: stopping: raw area {raw_stopped} could not be written; {later} not applied; fix the cause and re-run')
	return summary

#%% ---- Remove Extra ----
def _in_root_lost_and_found(path, dest_root):
	"""True for a filesystem's own lost+found (directly at a mount point) below dest_root, or anything in it: fsck's
	recovered orphans, never extra."""
	path, dest_root = os.path.abspath(path), os.path.abspath(dest_root)
	while path != dest_root and path.startswith(dest_root.rstrip(os.sep) + os.sep):
		if os.path.basename(path) == 'lost+found' and os.path.ismount(os.path.dirname(path)):
			return True
		path = os.path.dirname(path)
	return False

def _covered_directories(src_paths, dests, dest_image=False, exclude=None, max_workers=56, parallel_file_listing=True):
	"""The destination directories a copy of src_paths into dests covers, as absolute paths, judged as -rme judges a file (by
	where the copy put it, _landings): where each directory of a directory source landed (from the source's listing, which a
	run's copy has cached), and each source's landing with every directory on the way to it (-di's parents, a list's)."""
	mapping = _src_to_dest_map(src_paths, dests, dest_image)
	by_source = _by_source(mapping)
	covered = set()
	for _, actual_dests in mapping:
		for actual_dest in actual_dests:
			covered.update(_path_and_parents(actual_dest))
	for src in src_paths:
		if not src or os.path.islink(src) or not os.path.isdir(src) or (exclude and is_excluded(_listed_root(os.path.abspath(src)), exclude)):
			continue
		# Spelled as the copy's job lists it (_copy_job: absolute, 'path/'), so a run's cached listing answers: no second walk.
		folders = get_file_list(os.path.abspath(src) + os.sep, max_workers=max_workers, exclude=exclude, parallel_file_listing=parallel_file_listing)[3]
		for folder in folders:
			covered.update(_landings(os.path.abspath(folder), by_source))
	return covered

def remove_extra_dirs(src_paths, dests,exclude=None,dest_image=False,keep=(),dest_roots=None,named=None,covered=None):
	"""
	Removes extra directories in destination that are not present in the source paths.

	A destination directory is the copy's when the copy covers it (covered: _covered_directories, which hpcp() takes from
	the copy's listings before -rme's file removal clears them; else listed here): where a directory of a directory source
	landed, or on the way to a source's landing. One under another spelling (case, normal form) is the copy's too, where
	the destination reads both as one name. Judged only below where a directory source lands (all of a -di image).
	Excluded directories, mount points and a filesystem's own lost+found, with everything below them, are left alone,
	and so are symlinks and the link targets create_sym_links tracked (keep, see _kept_paths); a directory that still
	holds entries (excluded files, say) is kept.

	:param src_paths: list of source paths
	:param dest: destination path
	:return: None
	"""
	# Skip if the dest path is too long
	newDests = []
	for d in dests:
		if len(d) > 4096:
			eprint(f'Destination path length error: {d} is too long')
		elif exclude and is_excluded(d,exclude):
			# A note: an excluded destination is left alone, as asked.
			print(f'Destination path excluded: {d} is excluded', file=sys.stderr)
		else:
			newDests.append(d)
	too_long = any(len(d) > 4096 for d in dests)
	dests = newDests
	if not dests:
		if too_long:
			eprint(f'All destination paths are too long: Skipping {src_paths}')
		return 0, 0, set(), frozenset()
	mapping = _src_to_dest_map(src_paths, dests, dest_image)
	if covered is None:
		covered = _covered_directories(src_paths, dests, dest_image, exclude)
	# Only below where a directory source lands (a list's directories, as remove_extra_files); -di is judged whole.
	scope = None if dest_image else {actual_dest for _, actual_dests in _directory_sources(mapping) for actual_dest in actual_dests}
	guarded = _source_guard(src_paths if named is None else named)
	kept = _kept_paths(keep, mapping)
	copied_roots = {actual_dest for _, actual_dests in mapping for actual_dest in actual_dests}
	excluded_roots = _excluded_roots(mapping, exclude)
	# {extra directory: (the destination root it lies below, its name as the walk spells it)}; each destination's
	# identity before the walk (pinned before the copy when hpcp() runs it, see remove_extra_files): one that now leads
	# elsewhere is not cleaned.
	extraDirs = {}
	if dest_roots is None:
		dest_roots = _destination_roots(dests)
	dests = [dest for dest in dests if os.path.abspath(dest) in dest_roots]
	visible_mounts = _visible_mount_points()
	print(pformat(dests))
	for dest in dests:
		# Another filesystem below it (a mount or bind mount, another device), as remove_extra_files' listing.
		boundary = _filesystem_boundary(dest, dest_roots[os.path.abspath(dest)][0], visible_mounts)
		dest_root = os.path.abspath(dest)
		dest_prefix = dest_root.rstrip(os.sep) + os.sep
		def keep_holding(path):
			# Another filesystem is in it: it, and the directories holding it, are not extra (up to the destination,
			# '/' too: dirname('/') is '/').
			path = os.path.abspath(path)
			while path != dest_root and path.startswith(dest_prefix):
				extraDirs.pop(path, None)
				path = os.path.dirname(path)
		def entered_elsewhere(path):
			# A filesystem mounted on entering it (an NFSv4 submount, an automount): not walked, and not extra.
			print(f"Skipping mount point: {path}")
			keep_holding(path)
		for dirpath, dirnames, _ in _walk(dest, device=boundary[1], foreign=entered_elsewhere):
			walk_into = []
			for dirname in dirnames:
				# No trailing separator: excludes match a directory as the listing does ('*/cache').
				dest_dir_path = os.path.join(dirpath, dirname)
				# The walk already pruned excluded parents: only the directory's own source path is left to check.
				if exclude and (is_excluded(dest_dir_path,exclude) or _excluded_at_source(os.path.abspath(dest_dir_path), mapping, exclude, parents_checked=True, excluded_roots=excluded_roots)):
					print(f"Skipping excluded directory: {dest_dir_path}")
					continue
				# A mount point first, by name (an lstat of it would reach the mounted filesystem: a dead NFS server hangs).
				if dest_dir_path in boundary[0]:
					print(f"Skipping mount point: {dest_dir_path}")
					keep_holding(dest_dir_path)
					continue
				# Another filesystem by device: the walk checks it as it enters it (entered_elsewhere). A symlink is
				# among the walk's filenames then (its d_type); without a device boundary (Windows), among dirnames.
				if boundary[1] is None and os.path.islink(dest_dir_path):
					continue
				if dirname == 'lost+found' and os.path.ismount(dirpath):
					# The filesystem's own (fsck's recovered orphans go there): not part of any copy.
					print(f"Skipping lost+found: {dest_dir_path}")
					continue
				walk_into.append(dirname)
				path = os.path.abspath(dest_dir_path)
				if (path not in covered and not _is_kept(path, kept, os.path.abspath(dest), copied_roots)
						and (scope is None or any(part in scope for part in _path_and_parents(path)))):
					extraDirs[path] = (os.path.abspath(dest), dest_dir_path)
			# Excluded trees and other filesystems are never walked into.
			dirnames[:] = walk_into
	# A destination may read two spellings of a name (case, normal form) as one: a covered directory is then listed under the
	# spelling that was there first. Asked of the destination (_RealPlaces: one probe a directory, for all its spellings); one
	# that cannot be asked keeps it.
	if extraDirs:
		folds = {_folded(path): [] for path in extraDirs}
		roots = tuple(os.path.abspath(dest).rstrip(os.sep) + os.sep for dest in dests)
		for path in covered:
			if path.startswith(roots):
				kin = folds.get(_folded(path))
				if kin is not None:
					kin.append(path)
		pairs = [(path, other) for path in extraDirs for other in folds[_folded(path)] if other != path]
		if pairs:
			places = _RealPlaces()
			places.expect(itertools.chain.from_iterable(pairs))
			for path in {path for path, other in pairs if places.same_place(path, other) is not False}:
				del extraDirs[path]
	for path, (_, name) in list(extraDirs.items()):
		if guarded(name, 'dir'):
			print(f"Not removing {name}: it is a source or lies in one")
			del extraDirs[path]
	# Once the walks have entered them all (one on another device then is no longer here).
	for _, name in extraDirs.values():
		print(f"Deleting extra directory: {name}")
	# Deepest first, by descriptors from each pinned destination root without following symlinks (_remove_listed_files).
	ordered = sorted(extraDirs, key=lambda path: (-len(pathlib.Path(path).parts), path))
	_remove_listed_files(_items_below(((directory, extraDirs[directory][0]) for directory in ordered), dest_roots, sort=False), rmdir=True,
		one_filesystem=True)

def _source_guard(src_paths):
	"""-rme's last word on a path it found extra: guarded(path, kind) is True when path (as the destination's walk spells
	it; kind 'dir', 'file' or 'link') is a source or lies in one, whatever the destination calls it: a source named
	inside the destination (hpcp out/sub/ b/ -d out/, a list entry inside it), a named link. Asked by identity,
	(st_dev, st_ino). The walks never follow a link, so each directory they found is what its name says: one stat of each
	(once) answers for everything below it, and a file or a link is stat'ed only in a directory that holds a file source. A link is
	judged as a link, never by where it leads: one that leads into a source is an extra like any other (a move that would
	land a file on its source is refused before any copy). (A directory that holds a source holds something, so it is
	never removed.) src_paths: the sources as named, those the copy left in place too."""
	def identity(path):
		try:
			st = os.stat(path)
			return st.st_dev, st.st_ino
		except OSError:
			return None
	dirs, files, file_parents, parents_seen = set(), set(), set(), {}
	for src in src_paths or ():
		try:
			st = os.lstat(src)
		except OSError:
			continue
		if stat.S_ISDIR(st.st_mode):
			dirs.add((st.st_dev, st.st_ino))
			continue
		# A file, or a link (copied as one): it, and what a link leads to; and the directories they are in (each once).
		files.add((st.st_dev, st.st_ino))
		places = [os.path.dirname(os.path.abspath(src))]
		target = identity(src) if stat.S_ISLNK(st.st_mode) else None
		if target:
			(dirs if os.path.isdir(src) else files).add(target)
			places.append(os.path.dirname(os.path.realpath(src)))
		for place in places:
			if place not in parents_seen:
				parents_seen[place] = identity(place)
				file_parents.add(parents_seen[place])
	file_parents.discard(None)
	# {directory: (it is or lies in a source, its identity)}, each asked once.
	known = {}
	def directory(path):
		# Up the directories to the first one answered, then each answered on the way down. (Above a destination only a
		# source that holds it can answer yes, and hpcp() then runs no -rme.)
		chain = []
		while path not in known:
			chain.append(path)
			parent = os.path.dirname(path)
			if parent == path:
				break
			path = parent
		answer = known[path][0] if path in known else False
		for part in reversed(chain):
			found = identity(part)
			answer = answer or found in dirs
			known[part] = answer, found
		return known[chain[0] if chain else path]
	def guarded(path, kind='file'):
		if not (dirs or files):
			return False
		path = os.path.abspath(path)
		if kind == 'dir':
			return directory(path)[0]
		inside, parent = directory(os.path.dirname(path))
		if inside:
			return True
		if parent not in file_parents:
			# No file or link source is in this directory: asked once a directory, not once an extra (an lstat each on NFS).
			return False
		if kind != 'link':
			return identity(path) in files
		# A link is itself a source only when it is a named one (copied as a link): where it leads does not make it one.
		# A named link, dangling too, put its own directory in file_parents.
		try:
			st = os.lstat(path)
		except OSError:
			return False
		return (st.st_dev, st.st_ino) in files
	return guarded

def _directory_sources(mapping):
	"""The directory sources of _src_to_dest_map's mapping: what -rme cleans below when a source came from a list."""
	return [(src_abs, actual_dests) for src_abs, actual_dests in mapping if os.path.isdir(src_abs) and not os.path.islink(src_abs)]

def _source_dest(src, dest, dest_image=False):
	"""Where source src lands for destination dest, as an absolute path: the one rule the copy and -rme share.
	See _src_to_dest_map for the layouts; a source from a list lands at dest/<its landing> (_ListedSource)."""
	if dest_image:
		return os.path.abspath(dest + os.path.abspath(src))
	if isinstance(src, _ListedSource):
		return os.path.abspath(os.path.join(dest, src.landing))
	# basename respects trailing separators: 'src/' -> '', 'src' -> 'src'. '.' and '..' give no name to land under either:
	# their contents land in the destination, and a landing never leaves it.
	sourceFolderName = os.path.basename(src)
	if os.path.basename(dest) == '' and sourceFolderName not in ('', os.curdir, os.pardir):
		return os.path.abspath(os.path.join(dest, sourceFolderName))
	return os.path.abspath(dest)

def _src_to_dest_map(src_paths, dests, dest_image=False):
	"""Build ``[(src_abs, [actual_dest_abs, ...]), ...]`` mirroring process_copy.

	``process_copy`` decides per-src whether each ``dest`` gains the src
	basename or receives the src contents directly:
	  * ``hpcp src dest/``   -> files land in ``dest/src/...``
	  * ``hpcp src/ dest/``  -> files land in ``dest/...``
	  * ``hpcp src dest``    -> ``get_dest_from_path`` first appends ``/`` to
	                            both, so it behaves like ``hpcp src/ dest/``
	  * with ``dest_image`` (``-di``) -> files land in ``dest/<absolute src>/...``
	                            for every dest, trailing separator or not
	  * a source from a list -> files land in ``dest/<its path>`` (``_ListedSource``)

	The list is sorted by ``src_abs`` length (longest first) so that nested
	srcs are attributed to the most specific entry.
	"""
	mapping = []
	for src in src_paths or []:
		if not src:
			continue
		mapping.append((os.path.abspath(src), [_source_dest(src, dest, dest_image) for dest in dests]))
	mapping.sort(key=lambda item: len(item[0]), reverse=True)
	return mapping

def _listed_root(src_abs):
	"""A source root as the copy names it: 'path/' for a directory, as is for a file or a symlink (recreated as a link)."""
	return src_abs.rstrip(os.sep) + os.sep if os.path.isdir(src_abs) and not os.path.islink(src_abs) else src_abs

def _excluded_roots(mapping, exclude):
	"""The source roots (from _src_to_dest_map's mapping) that exclude leaves out whole, as the copy names them."""
	return {src_abs for src_abs, _ in mapping if exclude and is_excluded(_listed_root(src_abs), exclude)}

def _excluded_at_source(dest_path, mapping, exclude, parents_checked=False, excluded_roots=None):
	"""True when dest_path is the copy of a source path that exclude leaves out: that path, or a directory above it
	below its source root (parents_checked: the caller already pruned those), or the source root itself as the copy
	names it (_listed_root: '*/cache/' matches a directory root, '*/cache' does not). -e names source paths: an
	absolute pattern never matches the destination side itself. mapping: _src_to_dest_map's result; excluded_roots:
	_excluded_roots' result, when the caller has it. When several sources share a destination and one is excluded,
	any path there may be its copy, and is kept."""
	if not exclude:
		return False
	if excluded_roots is None:
		excluded_roots = _excluded_roots(mapping, exclude)
	for src_abs, actual_dests in mapping:
		for actual_dest in actual_dests:
			if dest_path != actual_dest and not dest_path.startswith(actual_dest.rstrip(os.sep) + os.sep):
				continue
			if src_abs in excluded_roots:
				return True
			source = os.path.normpath(os.path.join(src_abs, os.path.relpath(dest_path, actual_dest)))
			parts = (source,) if parents_checked else _path_and_parents(source, stop=src_abs)
			if source != src_abs and any(is_excluded(part, exclude) for part in parts):
				return True
	return False

def _kept_paths(keep, mapping):
	"""The link targets -rme keeps, from create_sym_links' (target, target's source, link) triples. Left out: a target
	that holds its own link (a link to '..' would keep everything), and one inside a copied tree that is that tree's copy
	of the same source path (judged like any other). One that came from another source is kept."""
	kept = set()
	for dest_target, source_target, link_path in keep:
		dest_target = os.path.abspath(dest_target)
		if link_path == dest_target or link_path.startswith(dest_target.rstrip(os.sep) + os.sep):
			continue
		covered = False
		for src_abs, actual_dests in mapping:
			for actual_dest in actual_dests:
				if dest_target == actual_dest or dest_target.startswith(actual_dest.rstrip(os.sep) + os.sep):
					mapped = os.path.normpath(os.path.join(src_abs, os.path.relpath(dest_target, actual_dest)))
					covered = covered or mapped == os.path.normpath(os.path.abspath(source_target))
		if not covered:
			kept.add(dest_target)
	return kept

def _is_kept(path, kept, root, copied_roots=frozenset()):
	"""True when the nearest of path and the directories above it (below root) that is a kept path (_kept_paths) or a
	copied tree's root is a kept path: a copied tree inside a kept directory is still judged as that tree's copy."""
	for part in _path_and_parents(path, stop=root.rstrip(os.sep) or os.sep):
		if part in kept:
			return True
		if part in copied_roots:
			return False
	return False

def _expected_dest_paths(src_entries, src_paths, dests, dest_image=False, mapping=None):
	"""Compute the set of absolute dest paths each src entry should occupy (dest_image: see _src_to_dest_map;
	mapping: its result, when the caller has it)."""
	expected = set()
	if not src_entries:
		return expected
	if mapping is None:
		mapping = _src_to_dest_map(src_paths, dests, dest_image)
	by_source = _by_source(mapping)
	for entry in src_entries:
		if entry and os.path.isabs(entry):
			expected.update(_landings(os.path.abspath(entry), by_source))
	return expected

def _by_source(mapping):
	"""{source: its landing in each destination}, from _src_to_dest_map's mapping, for _landings."""
	by_source = {}
	for src_abs, actual_dests in mapping:
		by_source.setdefault(src_abs, []).extend(actual_dests)
	return by_source

def _landings(entry_abs, by_source):
	"""Where the copy put entry_abs, a path of a source (by_source: _by_source): one place in each destination for each source
	it lies in or is. By the entry's ancestors, not each entry against each source: a list gives one source per file. Every
	source above an entry counts: each copied it to its own place."""
	for src_abs in _path_and_parents(entry_abs):
		actual_dests = by_source.get(src_abs)
		if actual_dests is None:
			continue
		if src_abs == entry_abs:
			yield from actual_dests
		else:
			rel = os.path.relpath(entry_abs, src_abs)
			yield from (os.path.join(actual_dest, rel) for actual_dest in actual_dests)

def remove_extra_files(total_file_list, dests,max_workers,verbose,files_per_job,single_thread=False,exclude=None,parallel_file_listing=False,src_paths=None,sym_links=None,dest_image=False,keep=(),dest_roots=None,listed=None,named=None):
		"""Remove files in ``dests`` that have no counterpart in ``total_file_list``.

		``total_file_list`` may contain absolute src-side paths (as returned by
		``process_copy``) or already-trimmed relative paths (as loaded from a
		stored file list). When ``src_paths`` is provided, we compute the
		expected absolute dest path for each src entry by mirroring how
		``process_copy`` maps src into dest, then subtract from the actual
		dest listing. When ``src_paths`` is omitted, the legacy trim-based
		comparison is used (callers must then pass already-trimmed entries).
		``sym_links`` is the ``{src_link: [dest_links]}`` mapping returned by
		``process_copy``; its keys are added to the src side so legitimately
		copied symlinks are not mistaken for extras. ``dest_image``: the copy used
		the ``-di`` layout (see ``_src_to_dest_map``). ``keep``: destination paths
		(and everything below them) that are never extra: the link targets
		create_sym_links tracked (see _kept_paths). ``dest_roots``: the
		destinations as _checked_destination_roots gave them (pinned before
		the copy); a destination left out of it is not cleaned.
		"""
		print(f"Removing extra files from {dests} with {max_workers} workers")
		# Each destination's identity, before it is listed (pinned before the copy when hpcp() runs it): the removal,
		# after the prompt, goes from it by descriptors. A destination that now leads elsewhere is not cleaned.
		if dest_roots is None:
			dest_roots = _destination_roots(dests)
		dests = [dest for dest in dests if os.path.abspath(dest) in dest_roots]
		visible_mounts = _visible_mount_points()
		# {extra path: the destination root it was listed below}
		extra_roots = {}
		src_entries = set(total_file_list) if total_file_list else set()
		if sym_links:
			src_entries.update(sym_links.keys() if hasattr(sym_links, 'keys') else sym_links)
		inDestNotInSrc = set()
		unknown_lists = 0
		if src_paths:
			mapping = _src_to_dest_map(src_paths, dests, dest_image)
			expected = _expected_dest_paths(src_entries, src_paths, dests, dest_image, mapping)
			# Only what a directory source's copy covers is judged: a source says what to copy, not that the rest of the
			# destination is extra (9.81 moved list sources; 9.83 and earlier judged all of out/ for hpcp b -d out/, deleting
			# out/keep). src/ lands in the destination itself, which is all judged. -di mirrors each source at its absolute
			# path in the image, which is judged whole.
			listed_scope = None
			if _has_listed(src_paths) if listed is None else listed:
				mapping = _directory_sources(mapping)
				listed_scope = {actual_dest for _, actual_dests in mapping for actual_dest in actual_dests}
				print("Sources from a list: only the directories it names are cleaned of extra files")
			elif not dest_image:
				listed_scope = {actual_dest for _, actual_dests in _directory_sources(mapping) for actual_dest in actual_dests}
			guarded = _source_guard(src_paths if named is None else named)
			kept = _kept_paths(keep, mapping)
			copied_roots = {actual_dest for _, actual_dests in mapping for actual_dest in actual_dests}
			excluded_roots = _excluded_roots(mapping, exclude)
			for dest in dests:
				dest_abs_root = os.path.abspath(dest)
				# A filesystem mounted below the destination is left alone, as rsync -x: not even listed.
				dest_file_list, links, _, _ = get_file_list(dest, max_workers=max_workers, exclude=exclude, parallel_file_listing=parallel_file_listing,
					boundary=_filesystem_boundary(dest, dest_roots[dest_abs_root][0], visible_mounts))
				actual = {os.path.abspath(p) for p in dest_file_list}
				link_paths = {os.path.abspath(p) for p in links}
				actual.update(link_paths)
				# Only consider entries that actually live under this dest so we
				# never accidentally point os.remove outside of the destination.
				extra = actual - expected
				# A destination may answer to a name in another spelling (case, normal form): what the copy wrote is then
				# listed under the name that was there first. An expected name the listing lacks that is there all the same is that
				# file under another spelling (not by its number: exFAT through FUSE gives each spelling a number of its own).
				written = {_folded(path) for path in expected - actual if path.startswith(dest_abs_root + os.path.sep) and os.path.lexists(path)} if extra else ()
				for path in extra:
					if written and _folded(path) in written:
						continue
					if path == dest_abs_root or path.startswith(dest_abs_root + os.path.sep):
						if listed_scope is not None and not any(part in listed_scope for part in _path_and_parents(path)):
							continue
						# The copy of an excluded source path is not extra: it was left out, not deleted.
						if (not _excluded_at_source(path, mapping, exclude, excluded_roots=excluded_roots) and not _is_kept(path, kept, dest_abs_root, copied_roots)
								and not _in_root_lost_and_found(path, dest_abs_root)):
							if guarded(path, 'link' if path in link_paths else 'file'):
								print(f"Not removing {path}: it is a source or lies in one")
								continue
							inDestNotInSrc.add(path)
							extra_roots[path] = dest_abs_root
		else:
			for dest in dests:
				listed_root = os.path.abspath(dest)
				dest_file_list,links,_ ,_ = get_file_list(dest, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing,
					boundary=_filesystem_boundary(dest, dest_roots[listed_root][0], visible_mounts))
				dest_trimmed = trim_paths(dest_file_list,dest)
				dest_trimmed.update(trim_paths(links,dest))
				# trim_paths roots everything at os.path.dirname(dest); rebuild
				# absolute paths against the same base or we'd double up the
				# dest basename when ``dest`` lacks a trailing separator.
				# A bare name ('data', '.') has no directory part: its extras are relative to the CWD (not to /).
				dest_base = os.path.dirname(dest) or os.curdir
				not_comparable = 0
				if dest_trimmed and not (dest_trimmed & src_entries):
					# As a missing list: nothing is known of what belongs here.
					eprint(f"Remove skipped: the stored list matches nothing in {dest}: not removing extra files from it")
					unknown_lists += 1
					continue
				for file in dest_trimmed - src_entries:
					if '\n' in file:
						# A line break in a stored list separates entries: the list cannot say whether this name is in it.
						not_comparable += 1
						continue
					path = os.path.join(dest_base, file)
					inDestNotInSrc.add(path)
					extra_roots[path] = dest
				if not_comparable:
					print(f"{not_comparable} name(s) with a line break cannot be told from the stored list's lines: not treated as extra")
		print('-'*80)
		print("Files in dest but not in src:")
		for file in inDestNotInSrc:
			print(file)
		print('-'*80)
		if len(inDestNotInSrc) == 0:
			if not unknown_lists:
				print(f"No extra files found in {dests}")
		else:
			print(f"Files in dest but not in src count: {len(inDestNotInSrc)}")
			_confirm("Do you want to delete them? (y/n)", "Not deleting extra files")
			start_time = time.monotonic()
			# By descriptors from each pinned destination root, without following symlinks (_remove_listed_files): a
			# destination directory swapped for a symlink while the prompt waited is not removed through.
			_remove_items(_items_below(((path, extra_roots[path]) for path in inDestNotInSrc), dest_roots), single_thread, max_workers,
				verbose, files_per_job, one_filesystem=True)
			# The destinations' listings (cached) still show what was removed.
			_get_file_list_cache.clear()
			endTime = time.monotonic()
			print(f"Time taken to remove extra files: {endTime-start_time:0.4f} seconds")

#%% ---- Main Helper Functions ----
def mount_src_image(src_images: list,src_paths: list,mount_points: list,loop_devices: list):
	src_str = ''
	for src in src_images:
		if not os.path.exists(src):
			eprint(f"Source image not found: {src} does not exist")
			continue
		src_str += f"{os.path.basename(src)}-"
		# we will mount the all image all partitions to seperate temorary folders and add them to src_paths
		loop_device_dest = create_loop_device(src,read_only=True)
		loop_devices.append(loop_device_dest)
		partitions = get_partitions(loop_device_dest)
		if not partitions:
			# if there are no partitions, we mount the image directly
			partitions = [loop_device_dest]
		for partition in partitions:
			try:
				target_mount_point = tempfile.mkdtemp()
				mount_points.append(target_mount_point)
				print(f"Mounting {partition} at {target_mount_point}")
				run_command_in_multicmd_with_path_check(["mount","-o","ro",partition,target_mount_point],strict=False)
				# verify mount 
				if os.path.ismount(target_mount_point):
					src_paths.append(target_mount_point + os.path.sep)
				else:
					inStr = ''
					if sys.stdin.isatty():
						print(f"Partition {partition} cannot be mounted, what to do? (s/f/n)")
						print(f"s:  Skip {partition} \t:Skip this partition and continue mounting the rest ( default )")
						print(f"f:  Fix {partition} \t:Try to fix the partition and mount it again")
						print("n:  Exit")
						# Wait for user input with a 5 second timeout
						inStr = multiCMD.input_with_timeout_and_countdown(5)
					if (not inStr) or inStr.lower().startswith('s'):
						print(f"Partition {partition} cannot be mounted, skipping")
						continue
					elif inStr.lower().startswith('f'):
						fix_fs(partition)
						run_command_in_multicmd_with_path_check(["mount","-o","ro",partition,target_mount_point],strict=False)
						if os.path.ismount(target_mount_point):
							src_paths.append(target_mount_point + os.path.sep)
						else:
							print(f"Partition {partition} cannot be mounted after fixing, skipping")
							continue
					else:
						print("Exiting")
						raise RuntimeError("User requested stop: Exiting due to mount error")
			except Exception as e:
				eprint(f"Source image mount error: Error mounting partition {partition}, skipping")
				print(e)
				continue
	return src_str.strip('-')

def _present_sources(src_paths, exclude=None):
	"""(the sources that are there, the ones that are not, each reported). An excluded name that is not there is left out, as
	it would be if it were."""
	# lstat: a dangling symlink is a source like any other, copied as a link (find lists them).
	found, missing = [], []
	for src in src_paths:
		try:
			os.lstat(src)
			found.append(src)
		except OSError as e:
			if exclude and is_excluded(os.path.abspath(src), exclude):
				continue
			missing.append(src)
			if isinstance(e, (FileNotFoundError, NotADirectoryError)):
				eprint(f"Source path not found: {src} does not exist")
			else:
				# Nobody could look: it may well be there.
				eprint(f"Source path not found: {src} cannot be reached ({e.strerror})")
	return found, missing

def verify_src_path(src_paths: list, exclude=None):
	"""Keep the sources that are there; returns those that are not."""
	src_paths[:], missing = _present_sources(src_paths, exclude)
	if len(src_paths) == 0:
		raise RuntimeError("No source paths: No source paths specified, exiting")
	return missing

# True once this run read stdin as a list: stdin can then neither answer a prompt nor say whether a person is there.
_STDIN_HELD_LIST = False

def _confirm(question, stop):
	"""Ask question (y/n); anything but yes raises a 'User requested stop' with stop as its text. From stdin at a
	terminal; when stdin held the list (find ... | hpcp -rme), from /dev/tty. With neither, the same stop, saying why. A stop
	not acted on yet is, first: no question is asked of a run that was stopped."""
	_check_stop()
	if _STDIN_HELD_LIST:
		try:
			# Unbuffered bytes: a terminal cannot seek, which a buffered read-write file needs.
			with open('/dev/tty', 'r+b', buffering=0) as tty:
				tty.write((question + '\n').encode())
				answer = tty.readline().decode(errors='replace')
		except OSError as e:
			raise RuntimeError(f"User requested stop: {stop} as stdin held the source list and no terminal can be opened to ask ({e.strerror}); "
				"run it at a terminal, or give the list with -sfl FILE")
	else:
		if not sys.stdin.isatty():
			raise RuntimeError(f"User requested stop: {stop} as stdin is not a tty")
		print(question)
		answer = input()
	if not answer.lower().startswith('y'):
		raise RuntimeError(f"User requested stop: {stop}")

def _read_list_entries(file_list, stored=False):
	"""A list's non-empty entries as (names in order, exact). Bytes through os.fsdecode, so any file name works under any
	locale. Data holding a NUL byte is NUL-separated (find -print0, fd -0: a path cannot hold one) and exact; otherwise
	lines split on \\n alone, without a trailing \\r. A missing list file is reported and gives no entries. stored: a list
	hpcp wrote (-fl): always lines, and a \\r stays (a name may end in one; the callers strip)."""
	file_list_value = '' if file_list is None else str(file_list)
	if file_list_value in ('-', '/dev/stdin'):
		global _STDIN_HELD_LIST
		_STDIN_HELD_LIST = True
		# A stand-in stream without a binary side (a library caller's StringIO) gives text.
		binary = getattr(sys.stdin, 'buffer', None)
		data = binary.read() if binary is not None else os.fsencode(sys.stdin.read())
	elif not os.path.exists(file_list_value):
		eprint(f"File list not found: {file_list_value} does not exist")
		return [], False
	else:
		with open(file_list_value, 'rb') as f:
			data = f.read()
	exact = b'\0' in data and not stored
	names = data.split(b'\0') if exact else [line[:-1] if line.endswith(b'\r') and not stored else line for line in data.split(b'\n')]
	def decode(name):
		try:
			return os.fsdecode(name)
		except UnicodeDecodeError:
			# Windows: file names are UTF-8 to Python, but a list written by other tools is in the locale's encoding.
			return name.decode(locale.getpreferredencoding(False), 'surrogateescape')
	return [decode(name) for name in names if name], exact

def load_file_list(file_list):
	"""A stored list (-fl) as a set, for comparing: entries stripped of surrounding whitespace."""
	return frozenset(entry.strip() for entry in _read_list_entries(file_list, stored=True)[0] if entry.strip())

class _ListedSource(str):
	"""A source path that came from a list (-sfl, stdin): the path as given, plus landing, where it goes below each
	destination. Travels through src_paths as any other path; _source_dest reads it. See _list_landing."""
	__slots__ = ('landing',)

def _list_landing(path):
	"""The path a list entry keeps below the destination, as rsync --files-from and cp --parents: normalised, without a
	drive, a leading '/' or './', and without any '..', so it never leaves the destination. '' lands at the destination."""
	path = os.path.splitdrive(os.path.normpath(path))[1]
	return os.sep.join(part for part in path.split(os.sep) if part not in ('', os.curdir, os.pardir))

def _implied_parents(src):
	"""(source directory, its landing) for the directories above a _ListedSource's landing, deepest first."""
	landing, source = src.landing, os.path.normpath(src)
	while True:
		landing, source = os.path.dirname(landing), os.path.dirname(source)
		if not landing:
			return
		yield source, landing

def _sources_text(src_paths):
	"""Sources as a log line names them: a list can hold hundreds of thousands."""
	return f'{len(src_paths)} sources' if len(src_paths) > 20 else pformat(list(src_paths))

def _has_listed(src_paths):
	return any(isinstance(src, _ListedSource) for src in src_paths)

def load_source_list(file_list):
	"""A list of sources as _ListedSources, in list order. Hand-written lines stay forgiving: a line with surrounding
	whitespace is the exact name when that exists, else the stripped one (a NUL-separated list is exact). Entries are
	not expanded."""
	names, exact = _read_list_entries(file_list)
	sources = []
	for name in names:
		if not exact and name != name.strip() and not os.path.lexists(name):
			name = name.strip()
			if not name:
				continue
		source = _ListedSource(name)
		source.landing = _list_landing(name)
		sources.append(source)
	return sources

def _without_covered_sources(src_paths):
	"""src_paths without a list's repeats and the entries a directory listed above them already covers (plain find lists
	both), in order. Only by their spelling, so a covered entry would land where its directory's copy puts it anyway. A
	symlink to a directory covers nothing: it is copied as a link, unless the list also names entries below it
	(find -L lnk): those land in a directory of its name, which the link would only get in the way of, so the link goes."""
	listed = {os.path.normpath(src) for src in src_paths if isinstance(src, _ListedSource)}
	directories, links = set(), set()
	for path in listed:
		if os.path.isdir(path):
			(links if os.path.islink(path) else directories).add(path)
	def covered(path):
		if path == os.curdir:
			return False
		while True:
			parent = os.path.dirname(path)
			if parent == path:
				return False
			if not parent:
				# Relative: '.' holds it unless it climbs out of '.' ('../x').
				return os.curdir in directories and path != os.pardir
			if parent in directories:
				return True
			path = parent
	opened = set()
	if links:
		for path in listed:
			parent = os.path.dirname(path)
			while parent and parent != path:
				if parent in links:
					opened.add(parent)
				path, parent = parent, os.path.dirname(parent)
	kept, seen = [], set()
	for src in src_paths:
		if isinstance(src, _ListedSource):
			path = os.path.normpath(src)
			if path in seen or path in opened or covered(path):
				continue
			seen.add(path)
		kept.append(src)
	return kept

# shlex.quote's safe set: a word of only these needs no quotes
_SHELL_SAFE = re.compile(r'[\w@%+=:,./-]', re.ASCII)

def _shell_word(path):
	"""path as one word a POSIX shell (bash, sh, zsh) reads back as path, for a hint's command that an operator pastes: as it
	is when every character is shlex's safe set or a non-ASCII one the output's encoding shows; in single quotes (' as '\\'')
	when nothing in it is a control character or a character the output cannot show; else in $'...' with each byte outside
	printable ASCII as a three-digit octal escape (coreutils' shell-escape form; bash, ksh and zsh read it, dash 0.5.12 does
	not). A name that is not UTF-8 is spelled by its own bytes, never by Python's \\udcXX escapes."""
	encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
	def shows(character):
		if not character.isprintable():
			return False
		try:
			character.encode(encoding)
		except (UnicodeError, LookupError):
			return False
		return True
	if path and all(_SHELL_SAFE.match(character) or (ord(character) > 127 and shows(character)) for character in path):
		return path
	if all(shows(character) for character in path):
		return "'" + path.replace("'", "'\\''") + "'"
	spelled = []
	for byte in os.fsencode(path):
		if byte == 0x27:
			spelled.append("\\'")
		elif byte == 0x5c:
			spelled.append('\\\\')
		elif 0x20 <= byte < 0x7f:
			spelled.append(chr(byte))
		else:
			spelled.append('\\%03o' % byte)
	return "$'" + ''.join(spelled) + "'"

class _CannotSearch(RuntimeError):
	"""A refusal: directory cannot be searched. link: the last link on the path that was asked about. It is named when the
	directory is on the way to where it leads: the operator's way out is then the link, as the directory can belong to anyone."""
	def __init__(self, directory, reason):
		super().__init__(directory, reason)
		self.directory, self.reason, self.link = directory, reason, None
	def __str__(self):
		if self.link and _same_or_below(os.path.realpath(self.link), self.directory):
			return (f'Destination clash: {self.link} leads to {os.path.realpath(self.link)}, and {self.directory} on the way cannot be '
				f'searched ({self.reason}), so a clash could hide there; nothing is copied or removed. '
				f'To let it through: rm {_shell_word(self.link)}, or, as its owner, chmod u+rx {_shell_word(self.directory)}')
		return (f'Destination clash: {self.directory} cannot be searched ({self.reason}), so a link in it could hide a clash; '
			f'nothing is copied or removed. To let it through: chmod u+rx {_shell_word(self.directory)}')

def _last_link(path):
	"""The last symlink on the way down the absolute path, as far as it can be followed, or None."""
	found, current = None, os.sep
	for part in path.split(os.sep):
		current = os.path.join(current, part)
		try:
			if stat.S_ISLNK(os.lstat(current).st_mode):
				found = current
		except OSError:
			break
	return found

class _RealPlaces:
	"""Where paths really are, for one run. The one answer for every check that compares a source with a destination: a
	directory is looked at once (a lstat) and known by what it is, so a bind mount of it or another case of its name is the
	same place; below a destination nothing is followed (landing), so no file costs a stat of its own, however many a list
	names."""
	def __init__(self):
		self.directories = {}
		self.identities = {}
		# Below a destination: {spelled path: (real path, whether the resolution ended above or at it)} (landing).
		self.below = {}
		# The spellings the run has of a name in a directory ({(folded directory, folded name): {spelling: None}}), and what
		# each directory said about them (_names).
		self.kin = {}
		self.answers = {}
		self.folds = {}
	def directory(self, path):
		"""os.path.realpath(path) for an absolute path, by each directory on the way once, with one name for one directory.
		It need not exist."""
		try:
			return self._walk(path)
		except _CannotSearch as e:
			e.link = _last_link(path)
			raise
	def _walk(self, path):
		known, below = self.directories, []
		while path not in known:
			parent = os.path.dirname(path)
			if parent == path:
				known[path] = path
				break
			below.append(path)
			path = parent
		real = known[path]
		for path in reversed(below):
			parent, real = real, os.path.join(real, os.path.basename(path))
			# Set first: a loop of links ends here.
			known[path] = real
			try:
				st = os.lstat(real)
			except PermissionError as e:
				raise _CannotSearch(parent, e.strerror)
			except OSError:
				continue
			if stat.S_ISLNK(st.st_mode):
				real = known[path] = self._walk(os.path.realpath(real))
			elif stat.S_ISDIR(st.st_mode):
				real = known[path] = self.identities.setdefault((st.st_dev, st.st_ino), real)
		return real
	def place(self, path):
		"""Where the file or link path is: its real parent and its own name. A link is its own place."""
		parent, name = os.path.split(path)
		return os.path.join(self.directory(parent), name)
	def landing(self, root, path, whole=False):
		"""Where a copy to path, root or below it, really writes: root as directory() gives it (a destination the operator named is
		followed, as cp follows it), then each directory below it by lstat, known by what it is as _walk knows it. A link, a node,
		a file or a missing name there ends the resolution and the rest is spelled as is: nothing below a destination is followed
		(a link there is replaced, _dest_directory), so nothing below it exists yet. whole: path is a directory, resolved too; else
		its last name is its own."""
		root = root.rstrip(os.sep) or os.sep
		path = path.rstrip(os.sep) or os.sep
		real = self.directory(root)
		if path == root:
			return real
		rest = path[len(root.rstrip(os.sep)) + 1:].split(os.sep)
		spelled, ended = root, False
		for part in rest if whole else rest[:-1]:
			spelled = os.path.join(spelled, part)
			known = self.below.get(spelled)
			if known is None:
				candidate = os.path.join(real, part)
				st = None
				if not ended:
					try:
						st = os.lstat(candidate)
					except OSError:
						# Not there, or a directory that cannot be searched: spelled as is. Nothing below it is followed, so no
						# link there can send a file elsewhere, and the directory sync gives it the source's mode first.
						pass
				if st is not None and stat.S_ISDIR(st.st_mode):
					known = (self.identities.setdefault((st.st_dev, st.st_ino), candidate), False)
				else:
					known = (candidate, True)
				self.below[spelled] = known
			real, ended = known
		return real if whole else os.path.join(real, rest[-1])
	def expect(self, paths):
		"""Paths the run will compare (same_place, bucket): each directory is then asked about all the spellings it gets of a
		name at once, one probe however many there are (_ask_names). A path compared without it is asked about too, later."""
		for path in paths:
			for _ in self._names(path, ask=False):
				pass
	def _names(self, path, ask=True):
		"""(name, its fold, its spelling where the directory it lands in reads it, the answer of that directory: (answers, why)
		of _ask_names) for each name on path, as far as it is read. The spelling is the first spelling of the name that the run
		has there and the directory takes for one name with it, or None when that cannot be asked; a name the run spells one
		way only is its own spelling, unasked, and has no answer. The directories are those of these spellings, or the nearest
		that exists, as a new one takes after that. Each name is remembered as one the run spells there."""
		real = os.sep if path.startswith(os.sep) else ''
		folded = real
		for name in path.split(os.sep):
			if name in ('', os.curdir, os.pardir):
				real = os.path.join(real, name)
				continue
			fold = self.folds.get(name)
			if fold is None:
				fold = self.folds[name] = _folded(name)
			kin = self.kin.setdefault((folded, fold), {})
			kin.setdefault(name, None)
			spelling, answer = name, None
			if ask and len(kin) > 1:
				answer = self._answer(real or os.curdir, folded, fold, kin)
				spelling = answer[0].get(name)
			yield name, fold, spelling, answer
			real = os.path.join(real, name if spelling is None else spelling)
			folded = folded + os.sep + fold
	def _answer(self, directory, folded, fold, kin):
		"""What directory, or the nearest that exists, says about the spellings kin, which the run has there: asked once, and
		again with them all when the run has a new one (in their order, so a spelling's answer stays)."""
		key = (directory, folded, fold)
		answer = self.answers.get(key)
		if answer is None or answer[2] != len(kin):
			while directory not in (os.sep, os.curdir) and not os.path.isdir(directory):
				directory = os.path.dirname(directory) or os.curdir
			answer = self.answers[key] = _ask_names(directory, list(kin)) + (len(kin),)
		return answer
	def bucket(self, path):
		"""A key that two paths share whenever same_place may take them for one (True or None), so that what may meet path is
		found at once: the spellings that the directories read, up to a name of which the run has a spelling that cannot be
		asked about, and from there the folded path."""
		key = []
		for _, _, spelling, answer in self._names(path):
			if answer is not None and answer[1]:
				return tuple(key) + (None, _folded(path))
			key.append(spelling)
		return tuple(key)
	def same_place(self, first, second):
		"""Do landings first and second, spelled differently and folding alike (_folded), lead to one file? True, False, or None
		when that cannot be asked (cannot_ask says why). Each name where they differ is asked about in the directory it lands in
		(_names)."""
		self.expect((first, second))
		for one, other in itertools.zip_longest(self._names(first), self._names(second)):
			if one is None or other is None:
				return False
			if one[0] != other[0]:
				if None in (one[2], other[2]):
					return None
				if one[2] != other[2]:
					return False
		return True
	def cannot_ask(self, first, second):
		"""Why same_place(first, second) could not answer."""
		for one, other in zip(self._names(first), self._names(second)):
			if one[0] != other[0] and None in (one[2], other[2]):
				return '; '.join(dict.fromkeys(one[3][1][name] for name in (one[0], other[0]) if name in one[3][1]))
		return 'the destination cannot be asked'

def _identity_or_none(path):
	"""(st_dev, st_ino) of what path leads to, or None when it cannot be stat'ed."""
	try:
		st = os.stat(path)
		return st.st_dev, st.st_ino
	except OSError:
		return None

def _inside_destinations(src_paths, dests, places):
	"""(src_paths without the list entries that lie inside a destination and the arguments that are one, how many entries
	that were, the paths of the destinations that a kept directory holds, as its walk spells them, those arguments). An entry inside a destination (or the
	destination itself) stays where it is: copying it onto its own place is nothing, and moving it would destroy it.
	Absolute entries always stay. A relative entry is copied (cd overlay; find etc | hpcp -d ..) only when it and the
	working directory share a directory strictly inside every destination that holds the entry (by each spelling of the
	entry: as given and by its real parent), and its landing is neither the entry nor below it: then it lands in another place, however it is spelled, and a second run
	finds the same. A directory that holds the destination (find . | hpcp -d out, hpcp . -d out/) is kept: the walk below it
	must leave the destination out (_exclude_roots), or it would read files other sources are writing. A directory named as
	an argument that is a destination (hpcp * -d backup/) goes, as a list entry does: it would land below itself. 'a/' has
	no name to land under, so it lands on itself, a thing _settle_destinations judges."""
	roots = set()
	for dest in dests:
		roots.update((os.path.abspath(dest), places.directory(os.path.abspath(dest))))
	prefixes = tuple(root.rstrip(os.path.sep) + os.path.sep for root in roots)
	root_ids = {_identity_or_none(os.path.abspath(dest)) for dest in dests} - {None}
	kept, skipped, holding, left_out = [], 0, set(), []
	try:
		real_cwd = os.path.realpath(os.getcwd())
	except OSError:
		real_cwd = None
	for src in src_paths:
		path = os.path.abspath(src)
		directory = os.path.isdir(path) and not os.path.islink(path)
		spellings = (path, places.directory(path) if directory else places.place(path))
		if isinstance(src, _ListedSource) and any(spelling in roots or spelling.startswith(prefixes) for spelling in spellings):
			if os.path.isabs(src) or not _lands_elsewhere(spellings, src.landing, roots, real_cwd):
				if directory and any(spelling in roots for spelling in spellings):
					# The destination itself, named by a list as by an argument: all that is in it belongs to that source (-rme checks nothing).
					left_out.append(path)
				else:
					skipped += 1
				continue
		if directory and not isinstance(src, _ListedSource) and any(places.directory(os.path.abspath(dest)) == spellings[1] and
				_source_dest(src, dest) != os.path.abspath(dest) for dest in dests):
			left_out.append(path)
			continue
		if not directory and os.path.islink(path) and _identity_or_none(path) in root_ids:
			# A link to the destination, named or listed, is that destination: copied, it would be a link to its own directory,
			# and -rme would keep all that it leads to. Asked by identity: a link whose way cannot be searched is not the
			# destination, and is copied as a link.
			left_out.append(path)
			continue
		if directory:
			# Named as this entry's walk will spell it, whichever spelling of the entry holds the destination; a spelling
			# the directory takes for the entry's own counts too (FUSE exFAT gives D and d numbers of their own).
			for spelling in spellings:
				above = spelling.rstrip(os.path.sep) + os.path.sep
				depth = len(spelling.rstrip(os.path.sep).split(os.path.sep))
				for root in roots:
					if root.startswith(above):
						holding.add(path.rstrip(os.path.sep) + os.path.sep + root[len(above):])
						continue
					rest = root.rstrip(os.path.sep).split(os.path.sep)[depth:]
					if not rest:
						continue
					answer = _holds_in_another_spelling(spelling, root, places)
					if answer is None:
						head = os.path.sep.join(root.split(os.path.sep)[:depth])
						raise RuntimeError(f'Invalid arguments: {root} may lie inside the source {path}, and the destination cannot be asked: '
							f'{places.cannot_ask(head, spelling.rstrip(os.path.sep))}; nothing is copied or removed')
					if answer:
						holding.add(path.rstrip(os.path.sep) + os.path.sep + os.path.sep.join(rest))
		kept.append(src)
	return kept, skipped, holding, left_out

def _holds_in_another_spelling(directory, path, places):
	"""Is path, or what lies below it, inside directory (both real paths) under another spelling of directory, one that folds
	like it (_folded) and that the destination takes for it? True, False, or None when it cannot be asked (cannot_ask says
	why). One probe per pair whose folds meet and whose spellings do not: none in a run with no such pair. The one answer
	of the walk's check (_inside_destinations) and the move's (_settle_destinations)."""
	directory = directory.rstrip(os.sep) or os.sep
	head = os.sep.join(path.split(os.sep)[:len(directory.split(os.sep))])
	if head == directory or _folded(head) != _folded(directory):
		return False
	return places.same_place(head, directory)

def _lands_elsewhere(spellings, landing, roots, real_cwd):
	"""True when a relative list entry that lies inside a destination is still copied (see _inside_destinations)."""
	prefixes = [root.rstrip(os.path.sep) + os.path.sep for root in roots]
	for spelling in spellings:
		# Each spelling is judged against the destinations that hold it: a path that is inside by its given spelling and
		# outside by its real one (through a link) still has to part from the destination.
		inside = spelling.rstrip(os.path.sep) + os.path.sep
		for prefix in prefixes:
			if inside.startswith(prefix):
				try:
					shared = os.path.commonpath((spelling, real_cwd)).rstrip(os.path.sep) + os.path.sep
				except ValueError:
					# The working directory is on another drive (Windows): nothing to part from.
					return False
				if not shared.startswith(prefix) or shared == prefix:
					return False
	for root in roots:
		lands = os.path.normpath(os.path.join(root, landing)).rstrip(os.path.sep) + os.path.sep
		if any(lands.startswith(spelling.rstrip(os.path.sep) + os.path.sep) for spelling in spellings):
			return False
	return True

def _exclude_roots(exclude, roots):
	"""exclude, and the paths roots (see _inside_destinations) as patterns: a listing leaves them, and all below, out."""
	return frozenset(exclude or ()) | {glob.escape(root) for root in roots}

def _trim_source_paths(paths, src):
	"""trim_paths for one source of a stored list (-fl): entries are named where the copy lands them, so a source from a
	list keeps its tree. Paths may carry a ':<digest>' tail, so relative to the directory above src."""
	if not isinstance(src, _ListedSource):
		return trim_paths(paths, src)
	base = os.path.normpath(src)
	if not src.landing:
		return {os.path.relpath(path, base) for path in paths}
	parent = os.path.dirname(src.landing)
	return {os.path.join(parent, os.path.relpath(path, os.path.dirname(base))) for path in paths}

def _stored_names(lines):
	"""Every name a line of a stored list (-fl) may mean, so that none of them is taken for extra: the line, stripped of
	surrounding whitespace as well (a name may have some, a hand-edited line may add it), and each part of either before a
	':' (a stored line is 'name:digest', or 'link:target'; a name may hold a ':' itself). Wrongly keeping a file is the
	safe mistake."""
	names = set()
	for line in lines:
		for text in (line, line.strip()):
			if not text:
				continue
			names.add(text)
			colon = text.find(':')
			while colon != -1:
				names.add(text[:colon])
				colon = text.find(':', colon + 1)
	return frozenset(names)

def store_file_list(file_list, src_paths: list, single_thread=False, max_workers=multiprocessing.cpu_count(), verbose=False,
					files_per_job=1, compare_file_list=False, remove_extra=False, parallel_file_listing=False, exclude=None,
					diff_file_list=None,tar_diff_file_list = False, src_str=None,append_hash=True,full_hash=False):
	"""
	Process a file list by performing various operations such as removing extra files, getting file lists from source paths,
	comparing file lists, and writing the final file list to a file.

	Args:
		file_list (str): The path to the file list.
		src_paths (list): A list of source paths from which to get the file list.
		single_thread (bool, optional): Whether to use a single thread for file operations. Defaults to False.
		max_workers (int, optional): The maximum number of worker processes to use for parallel file listing. Defaults to 4 times the number of CPU cores.
		verbose (bool, optional): Whether to print verbose output. Defaults to False.
		files_per_job (int, optional): The number of files to process per job when removing extra files. Defaults to 1.
		compare_file_list (bool, optional): Whether to compare the file list with the source paths. Defaults to False.
		remove_extra (bool, optional): Whether to remove extra files from the file list. Defaults to False.
		parallel_file_listing (bool, optional): Whether to use parallel file listing. Defaults to False.
		exclude (str, optional): A pattern to exclude files from the file list. Defaults to None.

	Returns:
		None
	"""

	if os.name == 'nt' and not file_list.endswith('_file_list.txt'):
		file_list += '_file_list.txt'
	if remove_extra:
		print('-' * 80)
		if file_list not in ('-', '/dev/stdin') and not os.path.exists(file_list):
			# Nothing is known of what belongs: every file would be extra.
			eprint(f"File list not found: {file_list} does not exist, not removing extra files")
			raise RuntimeError("Not removing extra files, exiting")
		file_list_file = _stored_names(_read_list_entries(file_list, stored=True)[0])
		# Not with a source from a list: the stored names keep the list's tree, and this comparison would take them all for
		# extra.
		if len(src_paths) == 1 and not _has_listed(src_paths):
			# file_list_file already holds trimmed (relative) paths, so we don't
			# pass src_paths -- entries are used as-is for the comparison.
			remove_extra_files(file_list_file, [src_paths[0]], max_workers, verbose, files_per_job, single_thread, exclude=exclude, parallel_file_listing=parallel_file_listing)
			print('-' * 80)
		else:
			print("Currently only supports removing extra files for a single src_path, given as an argument, when using file_list")
		raise RuntimeError("Completed removing extra files, exiting")
	if compare_file_list or diff_file_list:
		# This means we have a file_list and a src_path so we compare them
		print(f"Comparing file list from {_sources_text(src_paths)} with {file_list}")
		if diff_file_list == 'auto':
			if not src_str:
				src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
			diff_file_list = f'DIFF_{src_str}_TO_{os.path.basename(os.path.realpath(file_list))}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
		workers = 1 if single_thread else max_workers
		new_map = {}
		for src in src_paths:
			new_map.update(_source_entry_map(src, exclude=exclude, max_workers=workers, parallel_file_listing=parallel_file_listing))
		states = _states_for_paths(set(new_map.values()), full_hash=full_hash, max_workers=workers)
		new_states = {key: states[path] for key, path in new_map.items()}
		new_entries = {key: _value_from_state(state) for key, state in new_states.items()}
		base_entries = _parse_file_list_lines(load_file_list(file_list), live_keys=new_entries)
		print(f"Number of entries in src: {len(new_entries)}")
		print(f"Number of entries in {file_list}: {len(base_entries)}")
		report_changes(_rows_from_list_entries(new_entries, base_entries, new_states), diff_file_list, tar_diff_file_list)
	else:
		fileList = set()
		for src in src_paths:
			files, links, _,folders  = get_file_list(src, max_workers=max_workers, exclude=exclude,append_hash=append_hash,full_hash=full_hash,parallel_file_listing=parallel_file_listing)
			fileList.update(_trim_source_paths(files, src))
			fileList.update(_trim_source_paths(links, src))
			fileList.update([folder_path + os.path.sep for folder_path in _trim_source_paths(folders, src)])
			if isinstance(src, _ListedSource):
				fileList.update(landing + os.path.sep for _, landing in _implied_parents(src))
		print(f"Number of files: {len(fileList)}")
		print(f"Writing file list to {file_list}")
		# sort the file list
		fileList = natural_sort(fileList)
		# Bytes, as _read_list_entries reads them: a name that is not UTF-8 goes back as it came.
		with open(file_list, 'wb') as f:
			for file in fileList:
				f.write(os.fsencode(file + '\n'))

def _is_filesystem_root(path):
	"""True when path is / itself, by device and inode (a bind mount of / too), as rm's --preserve-root checks; a
	symlink to / is just a link."""
	try:
		return not os.path.islink(path) and os.path.samestat(os.stat(path), os.stat(os.sep))
	except OSError:
		return False

def process_remove(src_paths: list,single_thread = False, max_workers = multiprocessing.cpu_count(),verbose = False,
				  files_per_job = 1, remove_force = False,exclude=None,batch = False,parallel_file_listing=False):
	"""-rm without a destination: remove the source trees; excluded paths stay, with the directories holding them.
	Each source is pinned (_path_identity) before the confirmation prompt, and everything is removed by directory
	descriptors from it without following symlinks: a source replaced, or a directory in it swapped for a symlink,
	while this runs (a tree other users can write to; this often runs as root) is not removed from, and is reported.
	Serially, one descriptor walk per source (_remove_tree_fd); with -pfl, pool jobs empty one directory each
	(_remove_trees_parallel) before that walk; with -rwloff, the trees are listed first and the listed files removed
	in pool jobs (_remove_listed_files). All sources are removed together, so batch (-b) changes nothing here. -F
	(FILES_RATE_LIMIT) paces the unlinks."""
	processedPaths = []
	for src in src_paths:
		if not os.path.lexists(src):
			continue
		# 'src/' is src itself ('ln/' the link ln, as the copy names it).
		path = os.path.abspath(src)
		if _is_filesystem_root(path):
			raise RuntimeError(f'Invalid arguments: -rm will not remove the filesystem root ({src}), as rm --preserve-root')
		if exclude and is_excluded(path,exclude):
			print(f'Skipping excluded path: {path}')
		else:
			processedPaths.append(path)
	source_ids = _source_identities(processedPaths)
	print(f"Removing files from {_sources_text(processedPaths)} with {max_workers if not single_thread else '1'} workers")
	error_start = len(ERRORS)
	if not remove_force:
		_confirm("Do you want to continue? (y/n)", "Not removing files")
	print('-'*80)
	start_time = time.monotonic()
	rate_limiter = _new_rate_limiter()
	removed = 0
	trees = []
	for path in processedPaths:
		if os.path.isdir(path) and not os.path.islink(path):
			trees.append(path)
			continue
		_file_list_rate_limit(rate_limiter)
		if _remove_one_listed(path, source_ids[path], '.'):
			removed += 1
	parallel = not single_thread and max_workers > 1
	if trees and parallel and not REMOVE_FILES_WHILE_LISTING:
		# -rwloff: list everything first, then remove the listed files in pool jobs.
		pairs = []
		for root in trees:
			files, links, _, _ = get_file_list(root, max_workers=max_workers, exclude=exclude, parallel_file_listing=parallel_file_listing, drop_cache=True)
			pairs.extend((path, root) for path in set(files) | set(links))
		removed += _remove_items(_items_below(pairs, {root: (root, source_ids[root], source_ids[root]) for root in trees}), single_thread, max_workers,
			verbose, files_per_job)
		# The walk's -F pace starts after the listing (its time is no credit).
		rate_limiter = _new_rate_limiter()
	elif trees and parallel and parallel_file_listing:
		removed += _remove_trees_parallel([(root, source_ids[root]) for root in trees], max_workers, exclude, rate_limiter)
	for root in trees:
		count, _, errors = _remove_tree_fd(root, source_ids[root], exclude, remove_files=True, rate_limiter=rate_limiter)
		removed += count
		for path, e in errors:
			_report_unremoved(path, e)
	print(f"Removed {removed} files and links from {pformat(processedPaths)} in {time.monotonic()-start_time:0.4f} seconds")
	print('-'*80)
	# A listing cached before (or by -rwloff) would show the removed trees.
	_get_file_list_cache.clear()
	# The -rwloff listing's copy notes (a special file, a path too long to list) are no errors here: the walk removes them.
	ERRORS[error_start:] = [error for error in ERRORS[error_start:] if not (error.startswith('Special file skipped:')
		or (error.startswith('Listing error:') and error.endswith(': path longer than 4096 characters, not listed')))]

_BELOW_FLAGS = _DIRECTORY_FLAGS | getattr(os, 'O_NOFOLLOW', 0)
# A parent only looked up in (its entry removed, or opened below it): O_PATH where there is one, so search permission
# is enough, as for shutil.rmtree (a drop box, mode 0333).
_PARENT_FLAGS = (getattr(os, 'O_PATH', 0) or os.O_RDONLY) | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)
# A descriptor walk or cursor holds one directory descriptor a level: a tree deeper than the soft open-files limit
# (often 1024) needs more, so the soft limit is raised to the hard one (often 524288), where the platform allows.
if resource is not None:  # (imported in the import header)
	try:
		_soft_nofile, _hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
		if _soft_nofile < _hard_nofile:
			resource.setrlimit(resource.RLIMIT_NOFILE, (_hard_nofile, _hard_nofile))
	except (ValueError, OSError):
		pass

def _path_identity(path):
	"""(st_dev, st_ino) of path itself (a symlink is not followed), or None when it cannot be read."""
	try:
		st = os.lstat(path)
	except OSError:
		return None
	return (st.st_dev, st.st_ino)

def _directory_identity(path):
	"""_path_identity of path, a directory as opened: a filesystem mounted on entering it (an automount, an NFSv4
	submount not mounted yet) is mounted, so this is what the removal, which opens it (_open_source_root), finds there;
	lstat and stat do not mount one and report the directory below it. Anything else (a symlink, a file: one lstat), or
	a directory that cannot be opened: as _path_identity."""
	try:
		st = os.lstat(path)
	except OSError:
		return None
	if stat.S_ISDIR(st.st_mode):
		try:
			fd = os.open(path, _BELOW_FLAGS)
		except OSError:
			pass
		else:
			try:
				st = os.fstat(fd)
			except OSError:
				return None
			finally:
				os.close(fd)
	return (st.st_dev, st.st_ino)

def _source_identities(src_paths):
	"""{source as the copy names it (abspath: 'ln/' is the link ln): its _directory_identity}, taken before the copy."""
	return {os.path.abspath(src): _directory_identity(os.path.abspath(src)) for src in src_paths}

def _replaced(path):
	return OSError(errno.ESTALE, 'it was replaced since the copy listed it', path)

def _open_source_root(root, root_id):
	"""A descriptor of the source root directory, opened from its parent with O_NOFOLLOW and checked against root_id (its
	identity when the copy started): a root renamed away and replaced by a symlink or another directory raises (ELOOP,
	ESTALE) instead of leading the removal into the replacement. The parent is opened by the path as given."""
	if os.path.dirname(root) == root:
		fd = os.open(root, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0))
	else:
		parent_fd = os.open(os.path.dirname(root) or os.curdir, _PARENT_FLAGS)
		try:
			fd = os.open(os.path.basename(root), _BELOW_FLAGS, dir_fd=parent_fd)
		finally:
			os.close(parent_fd)
	st = os.fstat(fd)
	if root_id is None or (st.st_dev, st.st_ino) != tuple(root_id):
		os.close(fd)
		raise _replaced(root)
	return fd

def _parent_below(root, root_id, rel):
	"""(descriptor or None, name, identity or None) for entry rel below the source root. The descriptor is rel's
	directory, reached from the pinned root (_open_source_root) one component at a time with O_NOFOLLOW, so a directory
	swapped for a symlink since the copy listed it raises (ELOOP, ENOTDIR) instead of leading outside the source. rel
	'.' is the root itself, named in its parent (opened by the path as given), and must still have identity root_id.
	Without descriptor support (_DIR_FD_WALK) the descriptor is None and the paths are checked instead (open to a swap
	between the check and the use)."""
	if rel in ('', '.'):
		if not _DIR_FD_WALK:
			return None, os.path.basename(root), root_id
		parent = os.path.dirname(root) or os.curdir
		return os.open(parent, _PARENT_FLAGS), os.path.basename(root), root_id
	parent_rel, name = os.path.split(rel)
	if not _DIR_FD_WALK:
		if os.path.islink(root) or _path_identity(root) != (tuple(root_id) if root_id else None):
			raise _replaced(root)
		if os.path.realpath(os.path.join(root, parent_rel)) != os.path.normpath(os.path.join(os.path.realpath(root), parent_rel)):
			raise OSError(errno.ELOOP, 'a directory on the path is a symlink', os.path.join(root, rel))
		return None, name, None
	return _descend(_open_source_root(root, root_id), [] if parent_rel in ('', '.') else parent_rel.split(os.sep), _BELOW_FLAGS), name, None

def _remove_one_listed(root, root_id, rel):
	"""Unlink one listed entry relative to its directory's descriptor (_parent_below), never a directory. For the root
	itself (rel '.') and for platforms without descriptor support; _remove_listed_files does the rest."""
	path = root if rel in ('', '.') else os.path.join(root, rel)
	fd = None
	try:
		fd, name, identity = _parent_below(root, root_id, rel)
		st = os.stat(name, dir_fd=fd, follow_symlinks=False) if fd is not None else os.lstat(path)
		if identity is not None and (st.st_dev, st.st_ino) != tuple(identity):
			raise _replaced(path)
		if stat.S_ISDIR(st.st_mode):
			return False
		if fd is not None:
			os.unlink(name, dir_fd=fd)
		else:
			os.unlink(path)
		return True
	except FileNotFoundError:
		pass
	except OSError as exc:
		_report_unremoved(path, exc)
	finally:
		if fd is not None:
			os.close(fd)
	return False

def _report_rmdir_failed(path, exc):
	if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.ESTALE):
		eprint(f'Remove directory failed: not removing {path}: it, or a directory on its path, is no longer what was listed ({exc.strerror})')
	else:
		eprint(f'Remove directory failed: {path}: {exc}')

def _report_unremoved(path, exc):
	if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.ESTALE):
		eprint(f'Remove process exception: not removing {path}: it, or a directory on its path, is no longer what the copy listed ({exc.strerror})')
	else:
		eprint(f'Remove process exception:\n Deleting {path} generated an exception: {exc}')

def _unlink_failed(name, dir_fd, path, exc):
	"""Report a failed unlink, unless name is a directory (_is_directory_now): left as it is."""
	if not _is_directory_now(name, dir_fd, exc):
		_report_unremoved(path, exc)

class _HeldDirs:
	"""The directories on the path a removal walk or cursor (the copy's door, _dest_directory) is at, by descriptor, from a
	pinned root: one a level."""
	__slots__ = ('names', 'fds')

	def __init__(self, root_fd):
		self.names, self.fds = [None], [root_fd]

	def __len__(self):
		return len(self.fds)

	def top(self):
		return self.fds[-1]

	def push(self, name, fd):
		self.names.append(name)
		self.fds.append(fd)

	def pop(self):
		"""Leave the deepest directory (closed)."""
		os.close(self.fds.pop())
		self.names.pop()

	def truncate(self, depth):
		"""Leave the directories below the one at depth (0: the root)."""
		while len(self.fds) > depth + 1:
			self.pop()

	def close_all(self):
		for fd in self.fds:
			os.close(fd)
		self.names, self.fds = [], []

def _remove_listed_files(items, rmdir=False, rate_limiter=None, one_filesystem=False):
	"""Worker body for remove_moved_sources and remove_extra_files: unlink these (root, its identity, path below it)
	entries, files and links, in path order (with rmdir, remove_extra_dirs: rmdir them, non-empty ones kept). Never a directory: unlink refuses one, so a listed path that became a directory keeps what it
	holds; one already gone is no error. Each goes relative to its directory's descriptor, reached from the pinned root
	(_open_source_root) one component at a time with O_NOFOLLOW: a source root replaced, or a directory swapped for a
	symlink, in a tree other users can write to (this often runs as root) stops the removal instead of deleting outside
	the source. The descriptors of the current directory and of the ones above it stay open while the entries share
	them, so each directory is opened about once per job; nothing is stat'ed (a stat costs a server round trip on
	Lustre). rate_limiter (-F, in this process) paces the unlinks. With one_filesystem (-rme), a directory on another
	device than its root (a filesystem mounted below it that mountinfo did not list, a btrfs subvolume: as rsync -x) is
	not entered, with a note. Returns (bytes, seconds) as delete_file_bulk does; bytes are not counted."""
	start_time = time.monotonic()
	def remove(name, fd, path):
		try:
			if rmdir:
				os.rmdir(name, dir_fd=fd)
			else:
				if rate_limiter is not None:
					_file_list_rate_limit(rate_limiter)
				os.unlink(name, dir_fd=fd)
		except (FileNotFoundError, IsADirectoryError):
			pass
		except OSError as exc:
			if not rmdir:
				_unlink_failed(name, fd, path, exc)
			elif exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
				print(f"Keeping non-empty directory: {path}")
			elif exc.errno == errno.EBUSY and _is_mount_point(path):
				# Mounted on since it was listed: another filesystem, left alone.
				print(f"Keeping mount point: {path}")
			else:
				_report_rmdir_failed(path, exc)
	fallback_devices = {}
	fallback_foreign = {}
	def foreign(root, rel):
		"""Without descriptors: a directory between root and rel is on another device than root (one_filesystem), as
		the descriptor cursor checks each one it opens. Each directory is looked at once."""
		try:
			if root not in fallback_devices:
				fallback_devices[root] = os.lstat(root).st_dev
			path = root
			for part in rel.split(os.sep)[:-1]:
				path = os.path.join(path, part)
				if path not in fallback_foreign:
					fallback_foreign[path] = os.lstat(path).st_dev != fallback_devices[root]
				if fallback_foreign[path]:
					return True
			return False
		except OSError:
			return False
	def remove_one(root, root_id, rel):
		if not rmdir:
			_file_list_rate_limit(rate_limiter)
			_remove_one_listed(root, root_id, rel)
			return
		e = _rmdir_below(root, root_id, rel)
		if e is None:
			return
		path = root if rel in ('', '.') else os.path.join(root, rel)
		if e.errno in (errno.ENOTEMPTY, errno.EEXIST):
			print(f"Keeping non-empty directory: {path}")
		elif e.errno == errno.EBUSY and _is_mount_point(path):
			print(f"Keeping mount point: {path}")
		else:
			_report_rmdir_failed(path, e)
	if not _DIR_FD_WALK:
		for root, root_id, rel in items:
			# A serial removal's stop not acted on yet is, before the next entry goes (a no-op in a pool's worker).
			_check_stop()
			if one_filesystem and rel not in ('', '.') and foreign(root, rel):
				eprint(f'Other filesystem left: {os.path.dirname(os.path.join(root, rel))} is another filesystem below {root}: '
					'nothing in it is removed')
				continue
			remove_one(root, root_id, rel)
		return 0, time.monotonic() - start_time
	# held: the directories down to the current one, from the root (_HeldDirs); None before the first root.
	held, current_root = None, None
	# (root, components) of a directory that could not be entered: nothing listed below it is removed.
	refused = None
	# The directory held shows (root, its path below the root): the next file in it needs no descriptor work.
	shown = None
	# With one_filesystem: the device of the current root.
	root_dev = None
	try:
		for root, root_id, rel in items:
			# A serial removal's stop not acted on yet is, before the next entry goes (a no-op in a pool's worker).
			_check_stop()
			if rel in ('', '.'):
				remove_one(root, root_id, rel)
				continue
			parent_rel, _, name = rel.rpartition(os.sep)
			if shown is not None and shown[0] == root and shown[1] == parent_rel:
				remove(name, held.top(), os.path.join(root, rel))
				continue
			shown = None
			parent = parent_rel.split(os.sep) if parent_rel else []
			if refused is not None and refused[0] == root and parent[:len(refused[1])] == refused[1]:
				continue
			if root != current_root:
				if held is not None:
					held.close_all()
					held = None
				current_root = None
				try:
					held = _HeldDirs(_open_source_root(root, root_id))
					if one_filesystem:
						root_dev = os.fstat(held.top()).st_dev
				except OSError as exc:
					if held is not None:
						held.close_all()
						held = None
					refused = (root, [])
					if not isinstance(exc, FileNotFoundError):
						_report_unremoved(root, exc)
					continue
				current_root = root
			depth = 0
			while depth < len(parent) and depth + 1 < len(held) and held.names[depth + 1] == parent[depth]:
				depth += 1
			held.truncate(depth)
			other_filesystem = False
			try:
				for part in parent[depth:]:
					fd = os.open(part, _BELOW_FLAGS, dir_fd=held.top())
					if one_filesystem:
						try:
							device = os.fstat(fd).st_dev
						except OSError:
							os.close(fd)
							raise
						if device != root_dev:
							# Another device below the root (one the listing did not see): not entered, closed now.
							os.close(fd)
							refused = (root, parent[:len(held)])
							eprint(f'Other filesystem left: {os.path.join(root, *refused[1])} is another filesystem below {root}: '
								'nothing in it is removed')
							other_filesystem = True
							break
					held.push(part, fd)
			except OSError as exc:
				refused = (root, parent[:len(held)])
				if not isinstance(exc, FileNotFoundError):
					_report_unremoved(os.path.join(root, *parent[:len(held)]), exc)
				continue
			if other_filesystem:
				continue
			shown = (root, parent_rel)
			remove(name, held.top(), os.path.join(root, rel))
	finally:
		if held is not None:
			held.close_all()
	return 0, time.monotonic() - start_time

_SCANDIR_FD = os.scandir in os.supports_fd
# Python 3.6 has no scandir on a descriptor. On Linux /proc/self/fd/N is the very directory descriptor N is open on (no
# lookup by path), and scandir of it reads d_type as scandir of N does.
_PROC_FD = not _SCANDIR_FD and os.path.isdir('/proc/self/fd')

def _scan_target(fd, path):
	"""What os.scandir reads directory descriptor fd (open on directory path) through: fd itself; else /proc/self/fd/N
	(Python 3.6 on Linux); else path while it still names the directory held (one stat; the descriptor, still held,
	keeps an automount there mounted), None when it no longer does (swapped since the open, for a symlink say)."""
	if _SCANDIR_FD:
		return fd
	if _PROC_FD:
		return '/proc/self/fd/%d' % fd
	try:
		held, named = os.fstat(fd), os.stat(path)
	except OSError:
		return None
	return path if (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino) else None

def _destination_roots(dests):
	"""{abspath(dest): (its real path, that path's _path_identity, dest's own identity)} for -rme: what lies below a
	destination is reached from its real path, as given (a symlink in it included: 'link/' is where the copy wrote) and
	then by descriptors. The destination itself, should it be extra, is the name as given: a symlink there is the link,
	never its target."""
	roots = {}
	for dest in dests:
		listed = os.path.abspath(dest)
		real = os.path.realpath(listed)
		roots[listed] = (real, _directory_identity(real), _path_identity(listed))
	return roots

def _checked_destination_roots(dests, pinned=None):
	"""_destination_roots now, checked against pinned (taken before the copy): a destination that resolves elsewhere
	now (swapped for a symlink, or a directory on its path), or that existed then and is another directory now, is
	reported and left out, so -rme does not clean it. One the copy created gets its identity now."""
	current = _destination_roots(dests)
	if pinned is None:
		return current
	checked = {}
	for listed, now in current.items():
		before = pinned.get(listed)
		if before is not None and (now[0] != before[0] or (before[1] is not None and now[1] != before[1])):
			eprint(f'Remove skipped: not removing extra entries from {listed}: it now leads to {now[0]}, not to the '
				f'destination the copy wrote ({before[0]})')
			continue
		checked[listed] = now
	return checked

def _items_below(pairs, roots, sort=True):
	"""[(root, its identity, path below it)] for pairs (path, the root it was listed below), for _remove_listed_files:
	roots maps each listed root (abspath) to (the root to remove from, its identity, the listed root's own identity).
	A relative path is taken from the CWD. The listed root itself is (listed root, its own identity, '.'): removed as
	the name given, never resolved. A path not below its root (it cannot be) is reported, not removed. With sort, in
	path order (neighbours share their directories' descriptors)."""
	items = []
	# Per listed root (a few): its absolute path, prefix and roots entry.
	known = {}
	# Paths compare natively; each root's entries stay together (they share its prefix).
	for path, listed_root in (sorted(pairs) if sort else pairs):
		root_info = known.get(listed_root)
		if root_info is None:
			absolute = os.path.abspath(listed_root)
			root_info = known[listed_root] = (absolute, absolute.rstrip(os.sep) + os.sep, roots[absolute])
		absolute, prefix, entry = root_info
		if not os.path.isabs(path):
			path = os.path.abspath(path)
		if path.startswith(prefix):
			items.append((entry[0], entry[1], path[len(prefix):]))
		elif path == absolute:
			items.append((absolute, entry[2], '.'))
		else:
			eprint(f'Remove process exception: not removing {path}: it is not below {absolute}')
	return items

def _new_rate_limiter():
	"""A _file_list_rate_limit state for -F (FILES_RATE_LIMIT) starting now, or None without -F."""
	return {'start_time': time.monotonic(), 'count': 0, 'limit': FILES_RATE_LIMIT} if FILES_RATE_LIMIT > 0 else None

def _remove_items(items, single_thread=False, max_workers=1, verbose=False, files_per_job=1, one_filesystem=False):
	"""Unlink items (see _remove_listed_files) in this process, paced by -F, or in pool jobs of neighbouring entries,
	large enough to share their directories' descriptors (with -F as given, as delete_file_list_parallel paces by
	job). With one_filesystem (-rme) nothing on another filesystem than its root is removed. Returns the number of
	items handled."""
	if not items:
		return 0
	if single_thread or max_workers <= 1:
		_remove_listed_files(items, rate_limiter=_new_rate_limiter(), one_filesystem=one_filesystem)
		return len(items)
	job_size = files_per_job if FILES_RATE_LIMIT > 0 else max(files_per_job, min(256, len(items) // (max_workers * 4) or 1))
	remover = functools.partial(_remove_listed_files, one_filesystem=True) if one_filesystem else _remove_listed_files
	return delete_file_list_parallel(items, max_workers, verbose, job_size, remover=remover)[0]

def _dir_entries(fd):
	"""[(name, is a directory)] for directory descriptor fd, a symlink being no directory: from d_type where scandir can
	read the descriptor (_scan_target; no stat per entry, a server round trip each on Lustre), else listdir and a stat
	each. None: unknown (no d_type, and its stat failed)."""
	result = []
	if _SCANDIR_FD or _PROC_FD:
		with os.scandir(_scan_target(fd, None)) as entries:
			for entry in entries:
				try:
					result.append((entry.name, entry.is_dir(follow_symlinks=False)))
				except OSError:
					# Unknown (no d_type, and its stat failed): None; the walk tries it as a directory first.
					result.append((entry.name, None))
		return result
	for name in os.listdir(fd):
		try:
			result.append((name, stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode)))
		except FileNotFoundError:
			continue
		except OSError:
			result.append((name, None))
	return result

def _is_directory_now(name, dir_fd, exc):
	"""True when an unlink of name failed because it is a directory (EISDIR; macOS says EPERM)."""
	if isinstance(exc, IsADirectoryError):
		return True
	if isinstance(exc, PermissionError):
		try:
			return stat.S_ISDIR(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
		except OSError:
			return False
	return False

def _changed(fd, expected):
	"""fd, when it is the directory expected (an lstat) was taken of; else closed, and OSError ESTALE."""
	try:
		st = os.fstat(fd)
		if (st.st_dev, st.st_ino) == (expected.st_dev, expected.st_ino):
			return fd
		raise OSError(errno.ESTALE, 'changed while being removed; not entered')
	except BaseException:
		os.close(fd)
		raise

def _remove_tree_fd(root, root_id, exclude=None, remove_files=False, rate_limiter=None, spelled=None, parent_fd=None,
		mounts=None):
	"""Walk the directory tree at root by descriptors from the pinned root (_open_source_root), each directory opened
	relative to its parent with O_NOFOLLOW, deepest first, as shutil.rmtree does (one open and one listing per directory):
	rmdir each directory once its entries are handled. With remove_files (plain -rm) its other entries are unlinked
	first; without (a move's second pass) they are left. Excluded paths are not entered or removed. Returns (entries
	unlinked, entries left in place, excluded ones aside; errors as (path, OSError)): a directory swapped for a symlink,
	or a root replaced, is an error and nothing below it is touched. A directory that still holds something stays.
	rate_limiter (a _file_list_rate_limit state, -F) paces the unlinks. spelled: root as the caller wrote it, the base of
	the paths excludes are matched against (and errors name).

	For the apply's forced removal (_remove_tree_around_mounts): parent_fd, a descriptor of root's parent, starts the
	walk there instead of at the pinned root: root is its entry basename(root), opened with O_NOFOLLOW, checked against
	root_id (a stat_result), and removed relative to it. mounts: (mount points, device): an entry that is a mount point
	(of any type) or a directory on another device is left in place, in left, with the directories above it. Each entry
	is then stat'ed, each directory checked against that stat once opened, and a directory whose listing fails is still
	tried with rmdir."""
	removed, left, errors = 0, [], []
	if not _DIR_FD_WALK:
		base = spelled if spelled is not None else root
		directories = []
		for dirpath, dirnames, filenames in _walk(base):
			directories.append(dirpath)
			# os.walk lists a symlink to a directory among the directories: it is a link, removed like a file.
			links = [name for name in dirnames if os.path.islink(os.path.join(dirpath, name))]
			for name in filenames + links:
				path = os.path.join(dirpath, name)
				if exclude and is_excluded(path, exclude):
					continue
				if not remove_files:
					left.append(path)
				else:
					_file_list_rate_limit(rate_limiter)
					if _remove_one_listed(root, root_id, os.path.relpath(path, base)):
						removed += 1
			dirnames[:] = [name for name in dirnames if name not in links and not (exclude and is_excluded(os.path.join(dirpath, name), exclude))]
		for directory in sorted(directories, key=lambda path: path.count(os.sep), reverse=True):
			e = _rmdir_below(root, root_id, os.path.relpath(directory, base))
			if e is not None and e.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.EBUSY):
				errors.append((directory, e))
		return removed, left, errors
	try:
		if parent_fd is None:
			root_fd = _open_source_root(root, root_id)
		else:
			root_fd = _changed(os.open(os.path.basename(root), _BELOW_FLAGS, dir_fd=parent_fd), root_id)
	except FileNotFoundError:
		return removed, left, errors
	except OSError as e:
		return removed, left, [(root, e)]
	held = _HeldDirs(root_fd)
	# frames: [path, subdirectory names, next one, name in the parent], one per directory held (the root first)
	frames = []
	def enter(fd, path, name):
		"""Handle the entries of directory fd (just opened, or the root's), then hold it (frames, held)."""
		nonlocal removed
		try:
			entries = _dir_entries(fd) if mounts is None else [(entry, None) for entry in os.listdir(fd)]
		except OSError as e:
			errors.append((path, e))
			if mounts is None:
				if name is not None:
					os.close(fd)
				return False
			entries = []
		subdirectories = []
		for entry, is_dir in entries:
			child = os.path.join(path, entry)
			if exclude and is_excluded(child, exclude):
				continue
			if mounts is not None:
				try:
					info = os.stat(entry, dir_fd=fd, follow_symlinks=False)
				except FileNotFoundError:
					continue
				except OSError as e:
					errors.append((child, e))
					continue
				if child in mounts[0] or (stat.S_ISDIR(info.st_mode) and info.st_dev != mounts[1]):
					left.append(child)
				elif stat.S_ISDIR(info.st_mode):
					subdirectories.append((entry, False, info))
				else:
					try:
						os.unlink(entry, dir_fd=fd)
						removed += 1
					except FileNotFoundError:
						pass
					except OSError as e:
						errors.append((child, e))
				continue
			if is_dir is None:
				# Unknown: tried as a directory first (its open says ENOTDIR / ELOOP if it is none).
				subdirectories.append((entry, True, None))
			elif is_dir:
				subdirectories.append((entry, False, None))
			elif not remove_files:
				left.append(child)
			else:
				if rate_limiter is not None:
					_file_list_rate_limit(rate_limiter)
				try:
					os.unlink(entry, dir_fd=fd)
					removed += 1
				except FileNotFoundError:
					pass
				except OSError as e:
					if _is_directory_now(entry, fd, e):
						# A directory now (it changed since it was read): removed as one.
						subdirectories.append((entry, False, None))
					else:
						errors.append((child, e))
		if name is not None:
			held.push(name, fd)
		frames.append([path, subdirectories, 0, name])
		return True
	busy = (errno.ENOTEMPTY, errno.EEXIST) if remove_files else (errno.ENOTEMPTY, errno.EEXIST, errno.EBUSY)
	try:
		if not enter(held.top(), spelled if spelled is not None else root, None):
			return removed, left, errors
		while frames:
			frame = frames[-1]
			if frame[2] < len(frame[1]):
				entry, unknown, expected = frame[1][frame[2]]
				frame[2] += 1
				try:
					fd = os.open(entry, _BELOW_FLAGS, dir_fd=held.top())
					if expected is not None:
						fd = _changed(fd, expected)
				except FileNotFoundError:
					continue
				except OSError as e:
					child = os.path.join(frame[0], entry)
					if not (unknown and e.errno in (errno.ENOTDIR, errno.ELOOP)):
						errors.append((child, e))
					elif not remove_files:
						left.append(child)
					else:
						# No directory after all (a file or a symlink): unlinked.
						_file_list_rate_limit(rate_limiter)
						try:
							os.unlink(entry, dir_fd=held.top())
							removed += 1
						except FileNotFoundError:
							pass
						except OSError as unlink_error:
							errors.append((child, unlink_error))
					continue
				enter(fd, os.path.join(frame[0], entry), entry)
				continue
			frames.pop()
			if not frames:
				if parent_fd is None:
					e = _rmdir_below(root, root_id, '.')
				else:
					try:
						os.rmdir(os.path.basename(root), dir_fd=parent_fd)
						e = None
					except FileNotFoundError:
						e = None
					except OSError as rmdir_error:
						e = rmdir_error
				if e is not None and e.errno not in busy:
					errors.append((root, e))
				break
			held.pop()
			try:
				os.rmdir(frame[3], dir_fd=held.top())
			except FileNotFoundError:
				pass
			except OSError as e:
				if e.errno not in busy:
					errors.append((frame[0], e))
	finally:
		held.close_all()
	return removed, left, errors

# (shared count, start time, files per second) in a -pfl pool's workers under -F (_init_rate_share)
_RATE_SHARE = None

def _init_rate_share(counter, start_time, limit):
	"""Pool initializer for -pfl under -F: the jobs take their unlink slots from one shared counter."""
	global _RATE_SHARE
	_RATE_SHARE = (counter, start_time, limit)

def _shared_rate_wait():
	"""-F across pool jobs: take the next slot of the shared count and sleep until its time (a no-op without -F)."""
	if _RATE_SHARE is None:
		return
	counter, start_time, limit = _RATE_SHARE
	with counter.get_lock():
		counter.value += 1
		slot = counter.value
	wait = slot / limit - (time.monotonic() - start_time)
	if wait > 0:
		time.sleep(wait)

def _remove_directory_entries(path, expected_id, exclude=None):
	"""Worker body for the parallel plain -rm (-pfl): unlink the entries of directory path that are no directories,
	relative to its descriptor, and return (entries unlinked, [(subdirectory path, its identity)]) for the caller to
	queue. path is opened with O_NOFOLLOW and must still be expected_id, what its parent's listing (or the root's pin)
	saw: reached through a directory swapped for a symlink it is another directory, refused and reported."""
	try:
		fd = os.open(path, _BELOW_FLAGS)
	except FileNotFoundError:
		return 0, []
	except OSError as exc:
		# Too deep to open by path: left to _remove_tree_fd, which reaches it relative to its parent.
		if exc.errno != errno.ENAMETOOLONG:
			_report_unremoved(path, exc)
		return 0, []
	removed, subdirectories = 0, []
	try:
		st = os.fstat(fd)
		if expected_id is None or (st.st_dev, st.st_ino) != tuple(expected_id):
			_report_unremoved(path, _replaced(path))
			return 0, []
		for name, is_dir in _dir_entries(fd):
			child = os.path.join(path, name)
			if exclude and is_excluded(child, exclude):
				continue
			if is_dir or is_dir is None:
				try:
					child_st = os.stat(name, dir_fd=fd, follow_symlinks=False)
				except FileNotFoundError:
					continue
				except OSError as exc:
					_report_unremoved(child, exc)
					continue
				if stat.S_ISDIR(child_st.st_mode):
					subdirectories.append((child, (child_st.st_dev, child_st.st_ino)))
					continue
				if is_dir:
					continue
			_shared_rate_wait()
			try:
				os.unlink(name, dir_fd=fd)
				removed += 1
			except FileNotFoundError:
				pass
			except OSError as exc:
				if not _is_directory_now(name, fd, exc):
					_report_unremoved(child, exc)
					continue
				# A directory now: queued like the others.
				try:
					child_st = os.stat(name, dir_fd=fd, follow_symlinks=False)
				except OSError:
					continue
				subdirectories.append((child, (child_st.st_dev, child_st.st_ino)))
	except OSError as exc:
		_report_unremoved(path, exc)
	finally:
		os.close(fd)
	return removed, subdirectories

def _remove_trees_parallel(trees, max_workers, exclude=None, rate_limiter=None):
	"""-pfl: empty the directories of trees ([(root, its identity)]) in pool jobs, one directory each
	(_remove_directory_entries), queueing the subdirectories each job reports. With rate_limiter (-F), the jobs pace
	each unlink from one shared count (_shared_rate_wait), and the count carries on into the walk after. Returns the
	entries unlinked; the directories themselves are left for _remove_tree_fd."""
	queue = deque(trees)
	futures = {}
	removed = 0
	share = None
	if rate_limiter:
		# Every job paces each unlink from one shared count (a wide directory too, not just the queueing of jobs). Python 3.6's
		# pool takes no initializer: its workers fork from this process at the first submit and inherit it (_pool).
		counter = multiprocessing.Value('d', rate_limiter['count'])
		share = (counter, rate_limiter['start_time'], rate_limiter['limit'])
	with _pool(max_workers, rate=share) as executor:
		while queue or futures:
			while queue and len(futures) < max_workers * 2:
				path, identity = queue.popleft()
				futures[_submit_task(executor, _remove_directory_entries, path, identity, exclude)] = path
			for future in _wait(futures, executor):
				futures.pop(future)
				count, subdirectories = _task_result(future, (0, []), 'Remove process exception')
				removed += count
				queue.extend(subdirectories)
	if rate_limiter:
		rate_limiter['count'] = counter.value
	return removed

def _rmdir_below(root, root_id, rel):
	"""rmdir root/rel relative to its parent's descriptor (_parent_below). Returns the OSError, or None when it is gone."""
	fd = None
	# 'root/.' cannot be removed by path (EINVAL): the root itself is root.
	path = root if rel in ('', '.') else os.path.join(root, rel)
	try:
		fd, name, identity = _parent_below(root, root_id, rel)
		if identity is not None:
			st = os.stat(name, dir_fd=fd, follow_symlinks=False) if fd is not None else os.lstat(path)
			if (st.st_dev, st.st_ino) != tuple(identity):
				raise _replaced(root)
		if fd is not None:
			os.rmdir(name, dir_fd=fd)
		else:
			os.rmdir(path)
	except FileNotFoundError:
		return None
	except OSError as e:
		return e
	finally:
		if fd is not None:
			os.close(fd)
	return None

def _same_contents(first, second, second_open=None):
	"""True when two regular files hold the same bytes, read to the end; an OSError is the caller's. second_open: a callable
	that opens the second (is_file_identical's dest_open)."""
	with open(first, 'rb') as one, (open(second, 'rb') if second_open is None else second_open()) as other:
		while True:
			chunk = one.read(1 << 20)
			if chunk != other.read(1 << 20):
				return False
			if not chunk:
				return True

# Filesystems where a stat by name is a round trip to a server for each part of the path the client checks again: there
# the copy check looks at a directory's landings relative to it, opened once (_lstat_below: one path walk a directory, not
# one a landing). Measured on NFS only (an EL8 v4.1 loopback mounted actimeo=0, a rerun's copy check of 100,000 landings
# in 10,000 directories: 2.4 times fewer GETATTRs, 2 to 2.4 times faster); the others are listed because their clients
# ask the server along a path walk in the same way, not measured. On a local filesystem the opens cost more than the
# path walks they save (about 10 to 25% slower, Docker), so there a landing is looked at by its path, as in 9.86.
_NETWORK_FS = frozenset(('nfs', 'nfs4', 'lustre', 'gpfs', 'cifs', 'smb3', 'ceph', 'beegfs'))

def _on_network_fs(paths):
	"""True when one of paths (the destinations) is on a _NETWORK_FS: one stat a destination, its type from the mount table
	(_mount_type, Linux; elsewhere never, and a destination that cannot be stat'ed is not). macOS too takes every destination
	by path: on a local APFS disk the check by directory cost 2.3 times as much for 10,000 directories of one file each (each
	open and close about 20 us), and a rerun 12 % more; a macOS NFS or SMB mount is not measured yet (docs/limitations.md)."""
	for path in paths:
		try:
			if _mount_type(os.stat(path).st_dev) in _NETWORK_FS:
				return True
		except OSError:
			pass
	return False

# A landing directory opened to look at the names in it (_lstat_below): on Linux by O_PATH, which needs no read permission.
_LOOK_DIRECTORY_FLAGS = getattr(os, 'O_PATH', os.O_RDONLY) | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0)

class _OpenedDirectories(dict):
	"""{directory: its descriptor, None where it is not there, False where it is asked by path}: the landing directories one
	chunk of the copy check opened (_lstat_below), closed at the chunk's end."""
	def __enter__(self):
		return self
	def __exit__(self, *exc):
		for fd in self.values():
			if type(fd) is int:
				os.close(fd)

def _lstat_below(path, opened):
	"""os.lstat(path), relative to its directory opened once for all the names in it (opened: _OpenedDirectories): one path
	walk a directory, not one a name. The same answer: the kernel resolves the directory as the path walk does (a link to it
	followed, an 8.3 alias of the name found), and a directory that is not there has no names. One that cannot be opened for
	this (another error) is asked by path."""
	directory, name = os.path.split(path)
	if not name:
		return os.lstat(path)
	if directory not in opened:
		try:
			opened[directory] = os.open(directory, _LOOK_DIRECTORY_FLAGS)
		except OSError as e:
			opened[directory] = None if e.errno in (errno.ENOENT, errno.ENOTDIR) else False
	fd = opened[directory]
	if fd is None:
		raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
	if fd is False or os.stat not in os.supports_dir_fd:
		return os.lstat(path)
	return os.lstat(name, dir_fd=fd)

def _not_held(entries, src_paths, dests, dest_image=False, max_workers=1, move=True):
	"""The files and links of a copy (entries: absolute source paths) that no landing holds, as {path: why}: a move keeps
	them. A backstop that needs no rule on names: whatever the destination took for one name (an 8.3 alias, a legacy
	charset's other case, names of different lengths), one source's landing then holds another's content, or is one
	file with another source's landing, or is the source itself.

	move=False, the end of a copy that removes nothing: only a landing that is one file with another source's landing
	is reported, the clash itself, as both of its sources' lines, and a source that is at none of its landings. A landing
	of the source's own file is held, and one that differs from a source changed during the copy is what cp copied:
	neither is a clash, and nothing is read.

	A landing (each source above an entry gives one in each destination, as -rme's expected paths) holds its source when
	it is there and of its kind (a file's landing that is a symlink the operator named with -d is what it leads to: the copy
	wrote through it; one below a destination is no landing of a file, as the copy replaces it); a
	regular file's of its size and, unless -co, of its modification time (_mtimes_agree, as the skip check) or else of its
	contents; not the source itself (one inode, unless the source has other names that keep it, and never through a
	symlink); and, where the two share a directory as spelled and only their numbers differ (exFAT through FUSE gives each
	spelling a number of its own), the directory is asked by making both names (_ask_names). A landing that is one file
	with the landing of another source (another file) holds neither: both stay, whatever the content each now shows. One
	lstat of each source and each landing (a copy: of each landing), in max_workers threads where a stat waits; a file is
	read only when its landing's time disagrees. Where a destination is on a network filesystem (_NETWORK_FS), the entries go
	by source directory and each landing is lstat'ed relative to its directory, opened once a chunk (_lstat_below); elsewhere
	by its path. Where two landings may be one file, only the entries whose fingerprint met another's are looked at again.
	Memory: entries as a list, 16 bytes for each landing (and 8 for each entry, its place by directory, when grouped), and
	records only for what is reported or asked; the stats are dropped once looked at."""
	by_source = _by_source(_src_to_dest_map(src_paths, dests, dest_image))
	# A file's landings are its directory's, with its name: each directory is mapped once (a source itself goes by _landings).
	in_directory = {}
	def landings_of(path):
		if path in by_source:
			return list(dict.fromkeys(_landings(path, by_source)))
		parent, name = os.path.split(path)
		if parent not in in_directory:
			in_directory[parent] = list(dict.fromkeys(_landings(parent, by_source)))
		return [os.path.join(landing, name) for landing in in_directory[parent]]
	def judge(path, st, landing, lt, through_link):
		# Why landing does not hold path, or None when it does (the directory may still be asked, below).
		if (lt.st_dev, lt.st_ino) == (st.st_dev, st.st_ino):
			if landing == path or st.st_nlink < 2 or through_link:
				return f'{landing} is the source itself'
			return None
		if stat.S_IFMT(lt.st_mode) != stat.S_IFMT(st.st_mode):
			return f'{landing} is another kind of entry'
		if (stat.S_ISCHR(st.st_mode) or stat.S_ISBLK(st.st_mode)) and lt.st_rdev != st.st_rdev:
			return f'{landing} is another device'
		if not stat.S_ISREG(st.st_mode):
			return None
		if lt.st_size != st.st_size:
			return f'{landing} holds other content ({lt.st_size} bytes, not {st.st_size})'
		if CONTENT_ONLY or _mtimes_agree(st.st_mtime_ns, lt.st_mtime_ns):
			return None
		# A time the landing could not keep is no proof of other content: vfat keeps none before 1980 (it sets
		# 315532800), so such a file's landing is read against it; a read only where the time disagrees.
		try:
			if _same_contents(path, landing):
				return None
		except OSError as e:
			return f'{landing} could not be compared with it ({e.strerror})'
		return f'{landing} holds other content (modified at another time)'
	def source_stat(path):
		try:
			return os.lstat(path)
		except OSError:
			return None
	def landed(path, st, lstat):
		"""(the source's lstat or None, [(landing, its stat, written through a link)] for each landing there). lstat: os.lstat,
		or _lstat_below with the chunk's opened directories."""
		found = []
		for landing in landings_of(path):
			try:
				lt = lstat(landing)
			except OSError:
				continue
			if stat.S_ISLNK(lt.st_mode) and st is None:
				# A copy asks the source only beside a link at its landing, or where two sources may share a landing (below).
				st = source_stat(path)
				if st is None:
					return None, []
			# Only a link the operator named (-d) is written through, as cp follows its destination; one below a destination
			# was replaced by the copy (_cp_once), so a link there now is not what it wrote.
			through_link = stat.S_ISLNK(lt.st_mode) and not stat.S_ISLNK(st.st_mode) and os.path.abspath(landing) in NAMED_DESTS
			if through_link:
				try:
					lt = os.stat(landing)
				except OSError:
					# A link that leads nowhere holds nothing.
					through_link = False
			found.append((landing, lt, through_link))
		return st, found
	# A landing file with other names was not written by this run (the copy replaces a file with other names by a new one,
	# _cp_once): it was there, and held the source already (a destination deduplicated by hard links). Any other landing file is fingerprinted, (st_dev, st_ino) hashed, to find two that may be one file.
	def look(chunk):
		"""(the fingerprints of chunk's landing files with the position of each one's entry, {path: why} not held, {directory:
		[(path, landing)]} to ask, and {path: its held landings} for those asked about)."""
		prints, owners, not_held, ask, held = array.array('q'), array.array('q'), {}, {}, {}
		with _OpenedDirectories() as opened:
			lstat = functools.partial(_lstat_below, opened=opened) if grouped else os.lstat
			for position in chunk:
				path = entries[position]
				st = source_stat(path) if move else None
				if move and st is None:
					# Gone already: the removal has nothing to take.
					continue
				st, found = landed(path, st, lstat)
				for _, lt, _ in found:
					if lt.st_nlink < 2:
						prints.append(hash((lt.st_dev, lt.st_ino)))
						owners.append(position)
				if not move:
					# A copy whose landing is not there, though its source still is: lost (on vfat, beside an 8.3 alias of its
					# name). A copy that failed has said so, and hpcp() leaves these out for its sources.
					if not found and (st or source_stat(path)):
						not_held[path] = f'it is at none of its landings ({", ".join(landings_of(path))})'
					continue
				why, mine, pairs = None, [], []
				for landing, lt, through_link in found:
					reason = judge(path, st, landing, lt, through_link)
					if reason:
						why = why or reason
						continue
					if os.path.dirname(landing) == os.path.dirname(path) and (lt.st_dev, lt.st_ino) != (st.st_dev, st.st_ino):
						pairs.append((path, landing))
					mine.append(landing)
				if not mine:
					not_held[path] = why or f'it is at none of its landings ({", ".join(landings_of(path))})'
				elif pairs:
					ask.setdefault(os.path.dirname(path), []).extend(pairs)
					held[path] = mine
		return prints, owners, not_held, ask, held
	entries = entries if isinstance(entries, list) else list(entries)
	grouped = _on_network_fs(dests)
	if grouped:
		# The entries by source directory (a source directory's files share their landing directories, each opened once a
		# chunk), as their positions.
		by_directory = {}
		for index, path in enumerate(entries):
			by_directory.setdefault(path.rpartition(os.sep)[0], []).append(index)
		order = array.array('q', itertools.chain.from_iterable(by_directory.values()))
		del by_directory
	else:
		order = range(len(entries))
	def chunks(positions):
		# 1024 entries a chunk: _looked_at's thread handoff
		it = iter(positions)
		return iter(lambda: list(itertools.islice(it, 1024)), [])
	prints, owners, not_held, ask, held = array.array('q'), array.array('q'), {}, {}, {}
	for part_prints, part_owners, part_not_held, part_ask, part_held in _looked_at(chunks(order), look, max_workers):
		prints.extend(part_prints)
		owners.extend(part_owners)
		not_held.update(part_not_held)
		for directory, pairs in part_ask.items():
			ask.setdefault(directory, []).extend(pairs)
		held.update(part_held)
	# A move in one directory under another name: are the two one name there? Each directory once, about all its names.
	for directory, pairs in ask.items():
		answers, unknown = _ask_names(directory, list(dict.fromkeys(os.path.basename(name) for pair in pairs for name in pair)))
		for path, landing in pairs:
			mine, theirs = os.path.basename(path), os.path.basename(landing)
			if mine in unknown or theirs in unknown or answers.get(mine) == answers.get(theirs):
				held[path] = [other for other in held.get(path, ()) if other != landing]
				reason = unknown.get(mine) or unknown.get(theirs)
				why = f'{landing} may be the source itself, and the directory cannot be asked: {reason}' if reason else f'{landing} is the source itself, under another spelling'
				if not held[path]:
					not_held[path] = why
	# Two sources (two files) at one landing file, whatever it holds now: the destination took their names for one, and
	# one of the contents is not there. Fingerprints met twice are looked at again, for those landings only.
	shared, previous = set(), None
	for fingerprint in sorted(prints):
		if fingerprint == previous:
			shared.add(fingerprint)
		previous = fingerprint
	if not shared:
		return not_held
	# Only the entries whose fingerprint met another's are looked at again (by their positions).
	again = sorted({owners[k] for k in range(len(prints)) if prints[k] in shared})
	del prints, owners
	def collect(chunk):
		met = []
		with _OpenedDirectories() as opened:
			lstat = functools.partial(_lstat_below, opened=opened) if grouped else os.lstat
			for position in chunk:
				path = entries[position]
				st, found = landed(path, None, lstat)
				met.extend(((lt.st_dev, lt.st_ino), path, landing) for landing, lt, _ in found
					if lt.st_nlink < 2 and hash((lt.st_dev, lt.st_ino)) in shared)
		return met
	by_file = {}
	for met in _looked_at(chunks(again), collect, max_workers):
		for file_id, path, landing in met:
			by_file.setdefault(file_id, []).append((path, landing))
	identity = {}
	def identity_of(path):
		if path not in identity:
			st = source_stat(path)
			identity[path] = st and (st.st_dev, st.st_ino)
		return identity[path]
	for file_id, sharing in by_file.items():
		if len(sharing) < 2:
			continue
		# A landing that is the source's own file is no other source's.
		sharing = [(path, landing) for path, landing in sharing if identity_of(path) not in (None, file_id)]
		if len({identity_of(path) for path, _ in sharing}) > 1:
			for path, landing in sharing:
				others = ', '.join(f'{other_landing} (from {other})' for other, other_landing in sharing if identity_of(other) != identity_of(path))
				not_held[path] = f'{landing} is one file with {others}: the destination took the names for one'
	return not_held

class _Joined:
	"""Several collections read as one, as often as asked, without a list of them all."""
	def __init__(self, *parts):
		self.parts = parts
	def __iter__(self):
		return itertools.chain(*self.parts)

def _looked_at(chunks, look, max_workers):
	"""look(chunk) for each of chunks (an iterator), yielded as each is done. Threads pay only where a stat waits (network
	storage, a cold disk): in GIL handoffs they cost more than a stat that does not. 100,000 files on a local disk took
	0.31 s in one thread and 0.55 s in 32 for a copy's check (3 us an entry), 0.72 s and 1.2 s for a move's (7 us); on NFS
	mounted actimeo=0, 10,000 took 6.4 s in one and 3.0 s in 4. So chunks are looked at here while each takes under 20 us
	an entry, and the rest in max_workers threads once one takes longer, with two chunks a thread in flight, so that the
	chunks are never all held at once.

	A pool region (_DEFER): a stop is acted on after each chunk, and within 1 s while a thread's chunk is awaited. Left by an
	exception, the threads are not joined (_ABANDONED): an lstat on a dead NFS mount never returns, and the run ends by
	os._exit (main())."""
	global _DEFER, _ABANDONED
	_DEFER += 1
	try:
		for chunk in chunks:
			started = time.monotonic()
			result = look(chunk)
			_check_stop()
			yield result
			if max_workers > 1 and time.monotonic() - started > 20e-6 * len(chunk):
				break
		else:
			return
		pool = concurrent.futures.ThreadPoolExecutor(max_workers)
		def settled(future):
			while True:
				try:
					return future.result(timeout=1)
				except concurrent.futures.TimeoutError:
					_check_stop()
		try:
			pending = deque()
			for chunk in chunks:
				pending.append(pool.submit(look, chunk))
				if len(pending) >= 2 * max_workers:
					result = settled(pending.popleft())
					_check_stop()
					yield result
			while pending:
				result = settled(pending.popleft())
				_check_stop()
				yield result
		except BaseException:
			_ABANDONED = True
			if sys.version_info >= (3, 9):
				pool.shutdown(wait=False, cancel_futures=True)
			else:
				pool.shutdown(wait=False)
			_LEFT.append(pool)
			raise
		pool.shutdown(wait=True)
		# Let go of it inside the region, as _pool does its executor.
		del pool
	finally:
		_DEFER -= 1

def remove_moved_sources(src_paths, file_list, sym_links, single_thread=False, max_workers=multiprocessing.cpu_count(),
		verbose=False, files_per_job=1, remove_force=False, exclude=None, root_ids=None, dests=None, dest_image=False):
	"""-rm with a destination (a move): remove what the copy listed, file_list and the links (sym_links' keys), then
	each source directory that is empty after that, deepest first. Anything the copy did not list stays, with the
	directories holding it: a file written to a source after the copy listed its directory, a special file, an excluded
	path. What stays, other than excluded paths, is named in a 'Move left entries' note. root_ids: {source root: its
	_directory_identity when the copy started}; a root no longer that one removes nothing (taken now when not given).
	dests, dest_image: where the copy went; given, each file and link whose landing does not hold it stays, as a
	'Destination clash' (_not_held)."""
	roots = []
	for src in src_paths:
		path = os.path.abspath(src)
		if not os.path.lexists(path):
			continue
		if exclude and is_excluded(path, exclude):
			print(f'Skipping excluded path: {path}')
			continue
		roots.append(path)
	root_ids = {root: (root_ids or {}).get(root, _directory_identity(root) if root_ids is None else None) for root in roots}
	# Each listed entry with the outermost source root it lies below, so its removal is checked from there down: a
	# lookup of the entry's ancestors (-sfl gives one root per file). Only a root that is a directory holds what lies below
	# it: lb/x named beside lb, a link, is a root of its own (opened by its path, through the link), whatever its spelling.
	root_set = set(roots)
	holders = {root for root in roots if os.path.isdir(root) and not os.path.islink(root)}
	def outermost_root(path):
		found, candidate = (path if path in root_set else None), path
		while True:
			parent = os.path.dirname(candidate)
			if parent == candidate:
				return found
			candidate = parent
			if candidate in holders:
				found = candidate
	pairs = []
	for path in set(file_list) | set(sym_links or ()):
		if not os.path.isabs(path):
			path = os.path.abspath(path)
		root = outermost_root(path)
		if root is not None:
			pairs.append((path, root))
	kept = _not_held([path for path, _ in pairs], src_paths, dests, dest_image, 1 if single_thread else max_workers) if dests else {}
	for path in sorted(kept):
		eprint(f'Destination clash: not removing {path}: {kept[path]}')
	pairs = [(path, root) for path, root in pairs if path not in kept]
	listed = _items_below(pairs, {root: (root, root_ids[root], root_ids[root]) for root in roots})
	print(f"Removing the {len(listed)} files and links copied from {_sources_text(roots)} with {max_workers if not single_thread else '1'} workers")
	if not remove_force:
		_confirm("Do you want to continue? (y/n)", "Not removing files")
	# Nothing is removed for a run that was stopped.
	_check_stop()
	start_time = time.monotonic()
	# A named link goes last, after the trees too: a path named through it (lb/x, lb/d/) is reached while it is there.
	last = [item for item in listed if item[2] == '.' and item[0] not in holders and os.path.islink(item[0])]
	listed = [item for item in listed if item not in last]
	_remove_items(listed, single_thread, max_workers, verbose, files_per_job)
	left = []
	for root in roots:
		# A root inside another one is walked with it.
		if not os.path.isdir(root) or os.path.islink(root) or outermost_root(root) != root:
			continue
		_, root_left, errors = _remove_tree_fd(root, root_ids[root], exclude)
		left.extend(path for path in root_left if os.path.abspath(path) not in kept)
		for directory, e in errors:
			if e.errno in (errno.ELOOP, errno.ENOTDIR, errno.ESTALE):
				eprint(f'Remove directory failed: not removing {directory}: it, or a directory on its path, is no longer what the copy listed ({e.strerror})')
			else:
				eprint(f'Remove directory failed: {directory}: {e}')
	if last:
		_remove_items(last, single_thread, max_workers, verbose, files_per_job)
	if left:
		shown = ', '.join(pformat(path) for path in left[:10]) + (f' and {len(left) - 10} more' if len(left) > 10 else '')
		eprint(f'Move left entries: {len(left)} source entries the copy did not list (written during the move, or special '
			f'files) are left in place: {shown}')
	print(f"Time taken to remove files: {time.monotonic()-start_time:0.4f} seconds")
	# The copy's listings (cached) still show the removed sources.
	_get_file_list_cache.clear()

def get_dest_from_image(dest_image,mount_points: list,loop_devices: list):
	dest = ''
	target_mount_point = tempfile.mkdtemp()
	mount_points.append(target_mount_point)
	# see if dest_image file exist.
	# if it does, then we will setup a loop device and attempt to mount it.
	if os.path.exists(dest_image):
		print(f"Destination image {dest_image} exists, attempting to mount it.")
		# setup a loop device
		target_loop_device_dest = create_loop_device(dest_image)
		loop_devices.append(target_loop_device_dest)
		target_partition = get_largest_partition(target_loop_device_dest)
		# mount the loop device to a temporary folder
		print(f"Mounting {target_partition} at {target_mount_point}")
		run_command_in_multicmd_with_path_check(_rw_mount_command(target_partition, target_mount_point),strict=False)
		# verify mount 
		if os.path.ismount(target_mount_point):
			dest = target_mount_point + os.path.sep
		else:
			inStr = ''
			if sys.stdin.isatty():
				print("Destination image cannot be mounted, do you want to continue? (f/d/n)")
				print(f"f:  Fix {target_partition} \t:Try to fix the partition and mount it again ( default )")
				print(f"d:  Delete {dest_image} \t:Delete the old and create a new image (Warning: this will overwrite the existing image)")
				print("n:  Exit")
				inStr = multiCMD.input_with_timeout_and_countdown(15)
			if (not inStr) or inStr.lower().startswith('f'):
				try:
					# First find out the fs type
					fs_type = get_fs_type(target_partition)
					fix_fs(target_partition,fs_type)
					# mount it again
					run_command_in_multicmd_with_path_check(_rw_mount_command(target_partition, target_mount_point),strict=False)
					# verify mount
					if os.path.ismount(target_mount_point):
						dest = target_mount_point + os.path.sep
					else:
						raise RuntimeError("Destination image broken: Destination image cannot be mounted after fixing, exiting")
				except Exception as e:
					raise RuntimeError(f"Cannot fix destination image: Error fixing {target_partition}: {e}, exiting")
			elif inStr.lower().startswith('d'):
				run_command_in_multicmd_with_path_check(['losetup','-d',target_loop_device_dest])
				delete_file_bulk([dest_image])
				dest = None
			else:
				raise RuntimeError("User requested stop: Exiting due to mount error")
	else:
		dest = None
	return dest , target_mount_point

def get_dest_from_path(dest_path,src_paths: list,src_path,can_be_none = False):
	dest = ''
	src_path = list(src_path)
	if not dest_path:
		if can_be_none:
			return None
		if len(src_path) > 1:
			try:
				cwd = os.path.join(os.getcwd()) + os.path.sep
			except Exception:
				cwd = None
			inStr = ''
			if sys.stdin.isatty():
				print("Destination path not specified, do you want to continue? (l/y/n/...)")
				print(f"l:  {src_path[-1]} \t:Use last src_path in list ( default )")
				if cwd:
					print(f"y:  {cwd} \t:Use current working directory")
				print("n:  Exit")
				print("...:  Enter custom destination path")
				inStr = multiCMD.input_with_timeout_and_countdown(60)
			if (not inStr) or inStr.lower() == 'l':
				dest = str(src_path[-1])
				src_paths.remove(dest) if dest in src_paths else None
				if not src_paths:
					raise RuntimeError(f"No source paths: only the destination {dest} is left, exiting")
				print(f"Destination path not specified, using {dest}")
			elif cwd and inStr.lower() == 'y':
				dest = cwd
				print(f"Destination path not specified, using {dest}")
			elif inStr.lower() == 'n':
				raise RuntimeError("User requested stop: Exiting due to no destination path specified")
			else:
				dest = inStr
		else:
			try:
				cwd = os.path.join(os.getcwd()) + os.path.sep
			except Exception:
				cwd = None
			inStr = ''
			if sys.stdin.isatty():
				print("Destination path not specified, src_path length is 1, do you want to continue? (y/n/...)")
				if cwd:
					print(f"y:  {os.getcwd() + os.path.sep} \t:Use current working directory ( default )")
				print("n:  Exit")
				print("...:  Enter custom destination path")
				inStr = multiCMD.input_with_timeout_and_countdown(60)
			if (not inStr) or cwd and inStr.lower() == 'y':
				dest = cwd
				print(f"Destination path not specified, using {dest}")
			elif inStr.lower() == 'n':
				raise RuntimeError("User requested stop: Exiting due to no destination path specified")
			else:
				dest = inStr
	else:
		dest = dest_path
	if not dest:
		return None
	try:
		dest = str(dest)
	except Exception:
		eprint(f"Dest path DataType error: Error converting dest_path {dest} to string, ignoring it")
		return None
	# A list always means the destination is a directory, even with one entry.
	if _has_listed(src_paths):
		return dest if dest.endswith(os.path.sep) else dest + os.path.sep
	if len(src_paths) == 1 and os.path.isdir(src_paths[0]) and not src_paths[0].endswith(os.path.sep) and not dest.endswith(os.path.sep):
		src_paths[0] += os.path.sep
		dest += os.path.sep
	# A file named alone goes into a directory that exists (cp's rule), so the destination is that directory.
	if (len(src_paths) > 1 or src_paths[0].endswith(os.path.sep) or os.path.isdir(dest)) and not dest.endswith(os.path.sep):
		dest += os.path.sep
	return dest

def get_dests(dest_paths,dest_image,mount_points: list,loop_devices: list,src_paths: list,src_path,can_be_none = False):
	dests = []
	target_mount_point = ''
	if dest_image:
		if os.geteuid() != 0:
			eprint("Permission warning, mounting dest image likely requires root privileges, please expect errors if continuing. continuing in 5 seconds...")
			time.sleep(5)
		imgDest , target_mount_point = get_dest_from_image(dest_image,mount_points,loop_devices)
		if imgDest:
			dests.append(imgDest)
		elif dest_paths:
			print(f"Destination image {dest_image} does not exist, using dest_paths {dest_paths}")
	if dest_paths:
		for dest in dest_paths:
			if '*' in dest or '?' in dest or '[' in dest:
				# One pattern: expand_paths takes a list (a bare str would be globbed per character, '/' included).
				expanded_dests = expand_paths([dest])
			else:
				expanded_dests = [os.path.expanduser(dest)]
			for path in expanded_dests:
				# Folded once, lexically, as the copy itself takes it (abspath): l/../out is out, whatever l leads to, so no
				# directory is made where a link before '..' leads. A trailing separator (copy into) is kept. An empty one stays
				# empty: no destination given (normpath would make it '.').
				if path:
					trailing = path.endswith(os.sep) or bool(os.altsep) and path.endswith(os.altsep)
					path = os.path.normpath(path)
					if trailing and not path.endswith(os.sep):
						path += os.sep
				pathDest = get_dest_from_path(path,src_paths,src_path,can_be_none=can_be_none)
				if pathDest:
					dests.append(pathDest)
	elif not target_mount_point:
		pathDest = get_dest_from_path(None,src_paths,src_path,can_be_none=can_be_none)
		if pathDest:
			dests.append(pathDest)
	# get the str representation of the dests
	dest_str = '_'.join([os.path.basename(os.path.realpath(dest)) for dest in dests])
	if not dest_str:
		dest_str = 'undefined'
	return dests, dest_str, target_mount_point
	
def process_compare_file_list(src_paths: list, dests, max_workers = multiprocessing.cpu_count(),
							  parallel_file_listing = False,exclude=None,dest_image = None,diff_file_list = None,tar_diff_file_list = False,
							  append_hash = True,full_hash = False,src_image_roots = ()):
	# append_hash is kept for signature compatibility: live comparisons always use full entry states.
	dests = [dest for dest in dests if dest and os.path.exists(dest) and os.path.isdir(dest)]
	if not dests:
		raise RuntimeError(f"No destination paths in cpl: Destination image {dest_image} does not exist or dests {dests} is empty when comparing file lists, exiting")
	print(f"Comparing file list from {_sources_text(src_paths)} with {dests}")
	new_map, base_map = {}, {}
	if dest_image:
		for src in src_paths:
			if src in src_image_roots:
				# An image source and the -di image store, both mounted: patterns match '/' + the path inside each.
				src_new, src_base = build_rooted_maps(src, dests[0], exclude, max_workers, parallel_file_listing)
			else:
				src_new, src_base = build_store_maps(src, dests[0], exclude=exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing)
			new_map.update(src_new)
			base_map.update(src_base)
	else:
		# With image sources, patterns match '/' + the path inside each tree on both sides (as -diff -si -d DIR does).
		rooted = bool(src_image_roots)
		for src in src_paths:
			src_exclude = _rooted_excludes(exclude or (), src) if src in src_image_roots else exclude
			new_map.update(_source_entry_map(src, exclude=src_exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
		for dest in dests:
			dest_exclude = _rooted_excludes(exclude or (), dest) if rooted else exclude
			base_map.update(build_entry_map(dest, _trim_prefix(dest), exclude=dest_exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
	print('-'*80)
	rows = compute_changes(new_map, base_map, full_hash=full_hash, max_workers=max_workers)
	report_changes(rows, diff_file_list, tar_diff_file_list)

def create_image(dest_image,target_mount_point,loop_devices: list,src_paths: list,mount_points, max_workers = multiprocessing.cpu_count(),
				parallel_file_listing = False,exclude=None,dest_image_size=0,init_size=None):
	if target_mount_point and dest_image:
		# This means we were supplied a dest_image that does not exist, we need to create it and initialize it
		returnMountPoints = []
		currentMountPoint = target_mount_point + os.path.sep
		if init_size is None:
			init_size = 0
			for src in src_paths:
				src = os.path.abspath(src + os.path.sep)
				_,_,size,_ = get_file_list(src, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)
				init_size += size
		needed = _align_up(int(1.05 * init_size + _FS_IMAGE_SLAG))
		number_of_images, image_file_size = _plan_dest_images(init_size, dest_image_size)
		if dest_image_size <= 0:
			print(f"Estimated content size {format_bytes(init_size)}B Creating {dest_image} with size {format_bytes(image_file_size)}B")
		elif number_of_images == 1:
			print(f"Destination image size {format_bytes(image_file_size)}B is larger than estimated content size {format_bytes(needed)}B, using {format_bytes(image_file_size)}B")
		else:
			print(f"Destination image size {format_bytes(image_file_size)}B is smaller than estimated content size {format_bytes(needed)}B, creating {number_of_images} images of size {format_bytes(image_file_size)}B")
		for i in range(number_of_images):
			if i > 0:
				imageName = _split_dest_image_name(dest_image, i)
				currentMountPoint = tempfile.mkdtemp() + os.path.sep
				mount_points.append(currentMountPoint)
			else:
				imageName = dest_image
			resize_image(imageName, image_file_size)
			# setup a loop device
			target_loop_device_dest = create_loop_device(imageName)
			loop_devices.append(target_loop_device_dest)
			# zero the superblocks
			print(f"Clearing {target_loop_device_dest} and create GPT partition table")
			run_command_in_multicmd_with_path_check(['dd','if=/dev/zero','of='+target_loop_device_dest,'bs=1M','count=16'])
			#run_command_in_multicmd_with_path_check(f"parted -s {target_loop_device_dest} mklabel gpt")
			#run_command_in_multicmd_with_path_check(['sgdisk','-Z',target_loop_device_dest])

			print(f"Loop device {target_loop_device_dest} created")
			target_partition = get_largest_partition(target_loop_device_dest) # should just return the loop device itself, but just in case.
			# format the partition
			# check if mkudffs is available and image file size is smaller than 8 TiB
			# if shutil.which('mkudffs') and image_file_size < 8 * 1024 * 1024 * 1024 * 1024:
			# 	print(f"Formatting {target_partition} as udf")
			# 	run_command_in_multicmd_with_path_check(f"mkudffs --utf8 --media-type=hd --blocksize=2048 --lvid=HPCP_disk_image --vid=HPCP_img --fsid=HPCP_img --vsid=HPCP_img {target_partition}")
			# format with xfs if it is available and size bigger then 300 MiB
			if shutil.which('mkfs.xfs') and image_file_size > 300 * 1024 * 1024:
				print(f"Formatting {target_partition} as xfs")
				run_command_in_multicmd_with_path_check(['mkfs.xfs','-f',target_partition])
			else:
				print(f"Formatting {target_partition} as ext4")
				run_command_in_multicmd_with_path_check(['mkfs.ext4','-F',target_partition])
			# mount the loop device to a temporary folder
			print(f"Mounting {target_partition} at {currentMountPoint}")
			run_command_in_multicmd_with_path_check(["mount",target_partition,currentMountPoint],strict=False)
			# verify mount
			if not os.path.ismount(currentMountPoint):
				raise RuntimeError(f"Cannot mount destination image: Destination image {imageName} cannot be mounted, exiting.")
			returnMountPoints.append(currentMountPoint)
		return returnMountPoints
	else:
		raise RuntimeError("No destination image path: Destination path not specified, exiting.")

def _sync_implied_parents(src, dests, synced):
	"""Make the directories above a _ListedSource's landing in each of dests that are missing, top first, with their
	sources' metadata, and those where a link, FIFO or node is (_dest_directory replaces it). A directory or a file that
	exists is left as it is: it is not this run's to change.
	synced: the landings looked at so far in this run (one look each). Returns the finish's records (finish_directories)."""
	parents = []
	records = []
	for source, landing in _implied_parents(src):
		if landing in synced:
			break
		synced.add(landing)
		parents.append((source, landing))
	for source, landing in reversed(parents):
		# A link (or a FIFO, a node) there is a name of the tree, which the directory replaces (_dest_directory).
		missing = [path for path in (os.path.join(dest, landing) for dest in dests) if not os.path.isdir(path) or os.path.islink(path)]
		missing = [path for path in missing if not (os.path.lexists(path) and stat.S_ISREG(os.lstat(path).st_mode))]
		if missing:
			# A link to a directory (lnk/f) is made from the directory it leads to: the link itself has no mode to keep.
			records += _sync_directory(os.path.realpath(source) if os.path.islink(source) and os.path.isdir(source) else source, missing)[3]
	return records

def _copy_job(source, dests, dest_image=None):
	"""(src, its destinations, the destinations it must not take) as a copy job names them: 'path/' for a directory, as is
	for a file or a symlink. A destination the operator named that is a device node the copy would replace
	(_would_unlink_device) is not among its destinations, so each file lands on the others, and the job has none left only
	when every one is such a node. A name any of dests gives counts as named."""
	src = os.path.abspath(source)
	directory = os.path.isdir(src) and not os.path.islink(src)
	if directory:
		src += os.path.sep
	kept, refused = [], []
	named = {os.path.abspath(dest) for dest in dests}
	for dest in dests:
		actual = _source_dest(source, dest, bool(dest_image))
		# A directory onto a node is refused by its directory sync, which says so (it replaces nothing). A landing on a name
		# any -d gives counts (-d out/ -d out/n), not only the one -d names with its own.
		if not directory and actual in named and _would_unlink_device(src, actual):
			refused.append(actual)
		else:
			kept.append(actual)
	return src, kept, refused

def _same_or_below(path, root):
	"""True when path is root or lies below it, by their spelling."""
	return (path.rstrip(os.sep) + os.sep).startswith(root.rstrip(os.sep) + os.sep)

def _roots_overlap(roots):
	"""True when two of roots (paths) are the same or one lies below the other. Ended with a separator and sorted, the
	paths below a root follow it directly, so each one is compared with its neighbour only."""
	ordered = sorted(root.rstrip(os.path.sep) + os.path.sep for root in roots)
	return any(below.startswith(root) for root, below in zip(ordered, ordered[1:]))


# What lets a probe be made, by why it could not be.
_PROBE_FIXES = {errno.EACCES: 'make it writable', errno.EPERM: 'make it writable', errno.EROFS: 'mount it read-write',
	errno.ENOSPC: 'free some space there', errno.EDQUOT: 'free some of the quota'}

# Why a directory cannot hold a name: no other place fixes that.
_REFUSED_NAME = (errno.EINVAL, errno.EILSEQ, errno.ENAMETOOLONG)

def _ask_names(directory, names):
	"""Which of names, spellings that fold alike (_folded), does directory take for one name? ({name: the first of names it is
	one name with, itself when there is none}, {name: why that cannot be asked}), each name in one of them. Nothing else answers
	for every filesystem: it is a directory's flag on ext4 and f2fs, the mount's on vfat, exFAT and NTFS, the server's on SMB,
	and each folds by rules of its own. Nor does a file's number: exFAT through FUSE gives each spelling of one file a number of
	its own. So it is asked by doing it, all names at once, in a new directory beside what is there, which takes after its
	parent: each name in turn is made with O_EXCL and holds its place among those made; a name that is there already is one
	name with the file that holds it. Only where nothing can be made does a listing answer: it has each name once, so the
	names in it are names of their own. The directory keeps its times (its owner's: a group member who may write it changes
	them). SIGINT, SIGTERM and SIGHUP wait until the probe is gone. SIGKILL cannot wait, and a signal sent while another
	thread of the process exists may still reach that thread."""
	answers, why = {}, {}
	probe = '.hpcp-case-probe-%d-%s' % (os.getpid(), uuid.uuid4().hex)
	held = signal.pthread_sigmask(signal.SIG_BLOCK, _STOPS) if hasattr(signal, 'pthread_sigmask') else None
	try:
		base = probe_base = None
		try:
			st = os.stat(directory)
			# Relative to descriptors where the platform can (Linux, the BSDs): a name then costs its own length, never the
			# probe's full path (a landing near PATH_MAX). The directory's by O_PATH: search permission is enough.
			base = os.open(directory, _PARENT_FLAGS) if _PROBE_BY_FD else directory
			os.mkdir(*_at(base, probe), 0o700, **_fd_of(base))
		except OSError as e:
			failed = e
		else:
			made = []
			try:
				try:
					probe_base = os.open(probe, _BELOW_FLAGS, dir_fd=base) if _PROBE_BY_FD else os.path.join(directory, probe)
				except OSError as e:
					failed = e
				else:
					failed = _make_names(probe_base, directory, names, made, answers, why)
			finally:
				left = set(made)
				if probe_base is not None:
					try:
						left.update(os.listdir(probe_base))
					except OSError:
						pass
				undoing = [functools.partial(os.unlink, *_at(probe_base, name), **_fd_of(probe_base)) for name in left if probe_base is not None]
				undoing.append(functools.partial(os.rmdir, *_at(base, probe), **_fd_of(base)))
				# By path: futimens on an O_PATH descriptor fails.
				undoing.append(functools.partial(os.utime, directory, ns=(st.st_atime_ns, st.st_mtime_ns)))
				for undo in undoing:
					try:
						undo()
					except OSError:
						pass
				if isinstance(probe_base, int):
					os.close(probe_base)
		finally:
			if isinstance(base, int):
				os.close(base)
		if failed is not None:
			answers.clear()
			why.clear()
			try:
				listed = set(os.listdir(directory))
			except OSError:
				listed = set()
			for name in names:
				if name in listed:
					answers[name] = name
				else:
					why[name] = f'nothing can be made in {directory} to ask it ({failed.strerror}): {_PROBE_FIXES.get(failed.errno, "let files be made there")} and run again'
	finally:
		if held is not None:
			signal.pthread_sigmask(signal.SIG_SETMASK, held)
	return answers, why

# A probe is made relative to descriptors where every call it makes takes one.
_PROBE_BY_FD = (os.mkdir in os.supports_dir_fd and os.open in os.supports_dir_fd and os.unlink in os.supports_dir_fd and
	os.rmdir in os.supports_dir_fd and os.listdir in os.supports_fd and bool(getattr(os, 'O_PATH', 0)))

def _at(base, name):
	"""(name,) relative to base, a directory's descriptor, or (its path,) below base, a path."""
	return (name,) if isinstance(base, int) else (os.path.join(base, name),)

def _fd_of(base):
	return {'dir_fd': base} if isinstance(base, int) else {}

def _make_names(probe, directory, names, made, answers, why):
	"""Make each of names in probe (a descriptor or a path: _at), the new directory _ask_names made in directory, filling
	answers and why, and made with what was made, in order; each file holds its place in made. The error that stopped it,
	or None."""
	for name in names:
		try:
			fd = os.open(*_at(probe, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, **_fd_of(probe))
		except FileExistsError:
			# Made, not looked up: vfat with iocharset=utf8 finds only the one spelling, and still refuses the other.
			holder, reasons = (made[0], ()) if len(made) == 1 else _holder(probe, directory, name, made)
			if holder is None:
				why[name] = '; '.join([f'{directory} takes {name} for a name it holds, and cannot say which'] + list(dict.fromkeys(reasons)))
			else:
				answers[name] = holder
			continue
		except OSError as e:
			if e.errno in _REFUSED_NAME:
				why[name] = f'{directory} refuses the name {name} ({e.strerror}), so it cannot hold it'
				continue
			return e
		try:
			os.write(fd, b'%d' % len(made))
		except OSError:
			pass
		finally:
			os.close(fd)
		made.append(name)
		answers[name] = name
	return None

def _holder(probe, directory, name, made):
	"""Which of made, the names made in probe (_make_names) so far, holds name, that is there already, and why the questions
	that could not say could not: (the name or None, reasons). The file says; where it cannot (a lookup that misses the
	spelling it refused, a write that found no space), each one is asked on its own."""
	try:
		fd = os.open(*_at(probe, name), os.O_RDONLY, **_fd_of(probe))
		try:
			return made[int(os.read(fd, 32))], ()
		finally:
			os.close(fd)
	except (OSError, ValueError, IndexError):
		pass
	reasons = []
	for other in made:
		answers, why = _ask_names(directory, [other, name])
		if answers.get(name) == other:
			return other, ()
		reasons.extend(why.values())
	return None, reasons

# Default_Ignorable_Code_Point: ext4 and f2fs passed over these until Linux 6.14.
_IGNORABLE = re.compile('[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0'
	'\ufff0-\ufff8\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e0fff]')

# vfat and exFAT drop the dots that end a name: 'a.' is 'a'.
_TRAILING_DOTS = re.compile(r'\.+(?=%s|$)' % re.escape(os.sep))

def _folded(path):
	"""What two paths have in common when some filesystem could read them as one: a wide net, as the filesystem itself is asked
	about what it catches (_ask_names). Read as UTF-8 whatever the locale: under C, Python 3.6 holds 'É' as two escapes, which
	nothing folds. A name that is not UTF-8 is in a legacy encoding (cp1251, koi8-r, latin-1), which a filesystem mounted with
	that charset folds by its own table: there any letter, ASCII too ('i' and iso8859-9 'İ'), any byte past ASCII, and the
	punctuation that ends a double-byte character (cp932 'а' is 84 70, 'А' 84 40), may be another's case, so each is one mark,
	even where a few of them happen to make a UTF-8 character ('Пё' in cp1251), which its other case does not. Dotless 'ı' is 'i': a vfat with codepage 850 or 857 gives 'kız.txt' the short name 'KIZ.TXT'."""
	raw = os.fsencode(path)
	try:
		path = raw.decode('utf-8')
	except UnicodeDecodeError:
		path = os.sep.join(_legacy_name(name) for name in raw.split(os.fsencode(os.sep)))
	path = _IGNORABLE.sub('', unicodedata.normalize('NFD', unicodedata.normalize('NFD', path).casefold().replace('\u0131', 'i')))
	# Most names have a dot, and few end one: the pattern runs for those only.
	return _TRAILING_DOTS.sub('', path) if path.endswith('.') or '.' + os.sep in path else path

# In a name that is not UTF-8, the bytes a mount's own table may take for another's case: ASCII letters, every byte past ASCII,
# and 40-7E, which can end a double-byte character in the kernel's cp932, cp936, cp949 and cp950 ('@[\]^_`{|}~' among them).
# Digits and the punctuation below 40 never do there; they keep names of one length apart.
_LEGACY_MARKS = dict.fromkeys([*range(0x40, 0x7f), *range(0x80, 0x100)], '\ufffd')

def _legacy_name(name):
	"""A name of a path that is not UTF-8 as a whole, as _folded reads it: as UTF-8 where it is, else its digits and punctuation
	as they are and every other byte as the one mark U+FFFD."""
	try:
		return name.decode('utf-8')
	except UnicodeDecodeError:
		return name.decode('latin-1').translate(_LEGACY_MARKS)

def _names_may_fold(jobs, listing):
	"""The landings of jobs (of _settle_destinations) that another landing matches in everything but spelling: none when no two
	differ only in what _folded leaves out. A directory source is listed for it, which costs nothing: the copy reuses the
	listing. A file's landing brings the directories above it, which a listed path (l/x/f) makes: l/x meets a file l/X."""
	seen, shared, above = {}, set(), set()
	for src, actual_dests, directory, _, _ in jobs:
		landings = [actual_dests[0]]
		if directory:
			files, links, _, folders = listing(src)
			src_root, dest_root = src.rstrip(os.sep), actual_dests[0].rstrip(os.sep)
			landings = (dest_root + path[len(src_root):] for path in itertools.chain(files, links, folders))
		else:
			# Up to the first one already counted: a list's files share their directories.
			parent = os.path.dirname(actual_dests[0])
			while parent not in above and parent != os.path.dirname(parent):
				above.add(parent)
				landings.append(parent)
				parent = os.path.dirname(parent)
		for landing in landings:
			first = seen.setdefault(_folded(landing), landing)
			if first != landing:
				shared.update((first, landing))
	return shared

def _would_unlink_device(src, dest):
	"""True when a copy of src to dest would take the name of the device node that dest is, or leads to (a link to a disk
	is the disk to whoever writes through it). cp -a (_cp_flags) unlinks what is there for anything but a regular file; a
	regular file writes through, unless dest has other names, which goes first either way. Not when src is a link, which
	runs no cp (create_sym_links leaves a name alone), nor when src is a node of the same kind and number, which would only
	make the node again."""
	try:
		src_st, dest_st = os.lstat(src), os.stat(dest)
	except OSError:
		return False
	src_mode, dest_mode = src_st.st_mode, dest_st.st_mode
	if not (stat.S_ISBLK(dest_mode) or stat.S_ISCHR(dest_mode)) or stat.S_ISLNK(src_mode):
		return False
	if stat.S_IFMT(src_mode) == stat.S_IFMT(dest_mode) and src_st.st_rdev == dest_st.st_rdev:
		return False
	return not stat.S_ISREG(src_mode) or dest_st.st_nlink > 1

def _settle_destinations(src_paths, dests, dest_image=None, directory_only=False, exclude=None, max_workers=1, parallel_file_listing=False,
		remove=False, places=None):
	"""Everything decided before a copy starts, so the answer is the same in every copy mode, for arguments and lists.

	The rule, on real paths and on what the copy really takes: a run is refused when a file would land on a source file
	(its own too, in a move), unless, in a copy, that file belongs to a directory that lands on itself, which reads and
	writes nothing. cp rewrites a file in place, so another cp reading it gets nothing or a part, and a move then removes
	the path that held the other content. A landing where no source has a file is safe, inside a source's tree too: a
	move removes only what it listed. Refused as 'Destination clash' (213), or 'Invalid arguments' (241) for a whole
	source moved onto or into itself. -do copies no file and -di lands in its own image, so neither looks.

	Who then writes each destination file when sources share one, in source order, with symlinks counted as files:
	  * two files that land together because directory sources merge: the later source wins (an overlay, as a sequential
	    copy always gave), counted for each file that ends up replaced;
	  * a file named directly (an argument, a list entry) on either side: a clash. The first stays, as cp refuses to
	    overwrite a file it just created, and it is reported;
	  * a file where another source makes a directory (itself, or the directory a listed path implies): the directory
	    stays, whichever came first, and the file is reported as a clash (cp would nest it, or fail);
	  * a file that would take the name of a device node the destination names itself (-d /dev/sdb): the node stays, the
	    file lands on the other destinations (_copy_job) with a note; with none left it is reported as a clash and nothing is copied.
	    Below a destination a node is a file of the tree, and a source replaces it;
	  * two names that differ in spelling and that the directory they land in takes for one name (_names_may_fold, then
	    _RealPlaces asks it, in every destination, once about all the spellings of that name it gets, and finds which may
	    meet by bucket): a clash even between directory sources, and in
	    one tree, as nothing says which the operator meant. The first stays, by source and then by name, so a rerun keeps
	    the same one. Directories that differ only in case merge, as they will there.
	A source met again at its own place is no clash.

	Cost: each directory source is listed, which the copy would do next, and every name is folded in memory. Files are
	compared when two real roots meet (all of them), or only those whose names fold alike with another spelling (a pair
	anywhere in a tree costs that pair, not every file, and a name spelled n ways costs one probe of n names); a copy with
	disjoint roots and no such pair compares none. Up to
	there a file source costs what its parent directory costs (_RealPlaces). Nothing below a destination is followed: a
	link there is replaced by what the source has there (_dest_directory), so it hides no landing. Listings are cached, so the copy reads the same
	ones. Returns ({job: paths it must not copy}, files a later source replaced, the jobs of directories that land
	on themselves, which run no copy), a job being (its src, tuple(its dests))."""
	dropped = {}
	in_place = set()
	places = places or _RealPlaces()
	named = [os.path.abspath(dest) for dest in dests]
	def rooted(path, whole=False):
		"""Where a copy to path really writes, from the destination it lies in (_RealPlaces.landing)."""
		root = max((dest for dest in named if path == dest or path.startswith(dest.rstrip(os.sep) + os.sep)), key=len, default=path)
		return places.landing(root, path, whole)
	def listing(src):
		return get_file_list(src, max_workers=max_workers, exclude=exclude, parallel_file_listing=parallel_file_listing and max_workers != 1)
	# (src, its destinations, is it a directory, its real place, its real landing in each destination)
	jobs = []
	taken = []
	for source in src_paths:
		src, actual_dests, refused = _copy_job(source, dests, dest_image)
		if exclude and is_excluded(src, exclude):
			continue
		taken.extend((src, dest, tuple(actual_dests)) for dest in refused)
		if not actual_dests:
			continue
		directory = src.endswith(os.path.sep)
		root = src.rstrip(os.path.sep) or os.path.sep
		if directory:
			real, lands = places.directory(root), [rooted(dest, True) for dest in actual_dests]
		else:
			real, lands = places.place(src), [rooted(dest) for dest in actual_dests]
		if remove:
			for land in lands:
				# Every file is identical to itself, so the copy skips it and the removal would delete the only copy (9.72 did).
				# Nor may it be the same file under another spelling (X onto x, D into d/sub where the directory folds): asked of the
				# directory, as two spellings of one directory can have two numbers (exFAT through FUSE).
				if _same_or_below(land, real) or _holds_in_another_spelling(real, land, places) is not False:
					raise RuntimeError(f'Invalid arguments: -rm would move {root} onto itself ({land} is it or lies inside it); '
						'nothing is copied or removed')
		# Tuples of strings: the cyclic GC does not track them, so a million entries cost no collection time (lists did: about
		# 0.4 s a full collection per million, run several times while they were built).
		jobs.append((src, tuple(actual_dests), directory, real, tuple(lands)))
		if directory and real in lands:
			in_place.add((src, tuple(actual_dests)))
	if directory_only:
		# Nothing is copied.
		return dropped, 0, set()
	for src, dest, left in dict.fromkeys(taken):
		if left:
			# The file lands on another destination: the run goes on as any other.
			print(f'Note: {dest} is a device, and copying {src} would replace it: leaving it out; it goes to {", ".join(left)}')
		else:
			eprint(f'Destination clash: {dest} is a device, and copying {src} would replace it: not copying it. '
				'-dd clones a disk; remove the node to replace it')
	# A source whose landing lies below the landing of a link source (lb at out/lb, and lb/x at out/lb/x): the link is copied,
	# the path through it is a clash, so the outcome never hangs on which copy runs first. (A file there gives way to the
	# directory a path below it makes: add_directory, below.)
	singles = {}
	for src, actual_dests, directory, _, _ in jobs:
		if not directory and os.path.islink(src):
			for dest in actual_dests:
				singles.setdefault(dest, src)
	for src, actual_dests, directory, _, _ in jobs if singles else ():
		for dest in actual_dests:
			parent = os.path.dirname(dest.rstrip(os.sep))
			while parent not in singles and parent != os.path.dirname(parent):
				parent = os.path.dirname(parent)
			if parent in singles:
				above = singles[parent]
				eprint(f'Destination clash: {dest.rstrip(os.sep)} lies below {parent}, where this run puts the link {above}: '
					f'not copying {src.rstrip(os.sep)}')
				dropped.setdefault((src, tuple(actual_dests)), set()).add(src)
				break
	folds = _names_may_fold(jobs, listing)
	roots = [land for job in jobs for land in job[4]]
	# An image is a tree of its own. A source that holds its mount point (hpcp / -di) reads what an earlier run put there.
	if not dest_image:
		roots.extend(job[3] for job in jobs)
	overlap = _roots_overlap(roots)
	if not folds and not overlap:
		return dropped, 0, in_place
	def files_of(job):
		"""(path, its place, where it lands in each destination) for each file of job that the copy really takes: the listing's."""
		src, actual_dests, directory, real, _ = job
		if directory:
			src_root, real_root = src.rstrip(os.path.sep), real.rstrip(os.path.sep)
			dest_roots = [dest.rstrip(os.path.sep) for dest in actual_dests]
			files, links = listing(src)[:2]
			entries = ((path, real_root + path[len(src_root):], [dest_root + path[len(src_root):] for dest_root in dest_roots]) for path in itertools.chain(files, links))
		else:
			entries = ((src, real, actual_dests),)
		for path, place, targets in entries:
			# Where no root meets another, only a file whose name is shared can clash: the rest are not resolved.
			if overlap or targets[0] in folds:
				yield path, place, [rooted(target) for target in targets]
	if not dest_image:
		reads = {}
		for job in jobs:
			for path, place, landings in files_of(job):
				if remove and place in landings:
					raise RuntimeError(f'Destination clash: {path} would land on itself at {place}; nothing is copied or removed')
				if not (job[2] and job[3] in job[4]):
					reads[place] = path
		# A source file can be landed on in another spelling too, where the directory takes the two for one name, or cannot
		# say. Every file read is folded: no cheaper rule says which spellings a filesystem joins (lower() keeps a final sigma apart from a sigma).
		folded_reads = {}
		for read in reads:
			folded_reads.setdefault(_folded(read), []).append(read)
		meets = {}
		for job in jobs:
			for path, place, landings in files_of(job):
				if place not in landings:
					for land in landings:
						if land in reads:
							raise RuntimeError(f'Destination clash: {path} would land on {land}, which is the source file {reads[land]}; '
								'nothing is copied or removed')
						folded = _folded(land)
						if folded in folded_reads:
							meets.setdefault(folded, []).append((path, place, land))
		for folded, lands in meets.items():
			places.expect(itertools.chain(folded_reads[folded], (land for _, _, land in lands)))
			alike_reads = {}
			for read in folded_reads[folded]:
				alike_reads.setdefault(places.bucket(read), []).append(read)
			for path, place, land in lands:
				for read in alike_reads.get(places.bucket(land), ()):
					answer = read != place and places.same_place(land, read)
					if answer:
						raise RuntimeError(f'Destination clash: {path} would land on {land}, which is the source file {reads[read]}; '
							'nothing is copied or removed')
					if answer is None:
						raise RuntimeError(f'Destination clash: {path} may land on the source file {reads[read]}: {land} and {read} may be one '
							f'name, and the destination cannot be asked: {places.cannot_ask(land, read)}; nothing is copied or removed')
	replaced = 0
	holders = {}
	# The directories the sources make at the destinations. Those of the other destinations count where names fold alike: a
	# file can meet a directory of another source in any destination it may go to (files spill, -rds picks at random), and
	# only a later one may take the two for one name.
	dirs = set()
	def clash(path, job, why):
		eprint(f'Destination clash: {why}, not copying {path} over it')
		dropped.setdefault(job, set()).add(path)
	def spelled(path, job):
		"""Where path, a file of a directory job, lands in each destination of the job, as the source spells it."""
		root = job[0].rstrip(os.path.sep)
		return [dest.rstrip(os.path.sep) + path[len(root):] for dest in job[1]]
	def alike(path, job, pairs):
		"""pairs: (where the file it meets lands, where path lands), one for each destination. A clash, said for the first
		destination where the directory takes them for one name or cannot say, when there is one (True then)."""
		answers = [places.same_place(*pair) for pair in pairs]
		for answer in (True, None):
			if answer in answers:
				theirs, ours = pairs[answers.index(answer)]
				clash(path, job, f'{ours} and {theirs} are one name where the destination folds case' if answer else
					f'{ours} and {theirs} may be one name, and the destination cannot be asked: {places.cannot_ask(theirs, ours)}')
				return True
		return False
	def add_directory(path):
		"""path and the directories above it are directories at the destination: a file held there gives way."""
		while path not in dirs:
			dirs.add(path)
			held = holders.get(path)
			if held is not None and not os.path.islink(held[0]):
				del holders[path]
				clash(held[0], held[1], f'{path} is a directory of another source')
			parent = os.path.dirname(path)
			if parent == path:
				return
			path = parent
	for entry in jobs:
		src, actual_dests, directory, _, lands = entry
		job = (src, tuple(actual_dests))
		# Only names that fold alike are checked against the other destinations' directories.
		count_later = folds and len(lands) > 1
		if directory:
			src_root = src.rstrip(os.path.sep)
			for dest, land in zip(actual_dests, lands) if count_later else [(actual_dests[0], lands[0])]:
				add_directory(land)
				dest_root = dest.rstrip(os.path.sep)
				for folder in listing(src)[3]:
					add_directory(rooted((dest_root + folder[len(src_root):]).rstrip(os.path.sep), True))
		else:
			add_directory(os.path.dirname(lands[0]))
			for land in lands[1:] if count_later else ():
				add_directory(os.path.dirname(land))
		for path, place, landings in sorted(files_of(entry)) if folds else files_of(entry):
			if directory and place in landings:
				continue
			dest = landings[0]
			held = holders.get(dest)
			if dest in dirs and not os.path.islink(path):
				clash(path, job, f'{dest} is a directory of another source')
			elif held is None:
				holders[dest] = (path, job, directory, landings)
			elif held[0] == path:
				continue
			elif directory and held[2]:
				# Only one spelling is an overlay: two that the directory reads as one name are the clash of two names, which
				# an existing directory in the one case would hide (places.directory gave both its place).
				if alike(path, job, [pair for pair in zip(spelled(held[0], held[1]), spelled(path, job)) if pair[0] != pair[1] and _folded(pair[0]) == _folded(pair[1])]):
					continue
				dropped.setdefault(held[1], set()).add(held[0])
				replaced += 1
				# The same source met again (a/ b/ a/) wins back what it lost on the way: counted once, by its end.
				lost = dropped.get(job)
				if lost and path in lost:
					lost.discard(path)
					replaced -= 1
				holders[dest] = (path, job, directory, landings)
			else:
				clash(path, job, f'{dest} is already copied from {held[0]}')
	if folds:
		made = {}
		for path in dirs:
			made.setdefault(_folded(path), []).append(path)
		names = [_folded(dest) for dest in holders]
		count = Counter(names)
		# Only names that meet another are compared, and each directory is asked about all its spellings of one at once.
		shared = [(name, held) for name, held in zip(names, list(holders.items())) if count[name] > 1 or name in made]
		# The fold of each landing, in each destination: the first is the holder's own.
		landing_names = [[name] + [_folded(land) for land in held[1][3][1:]] for name, held in shared]
		places.expect(itertools.chain.from_iterable(itertools.chain(held[1][3], *(made.get(key, ()) for key in keys)) for keys, (_, held) in zip(landing_names, shared)))
		directories = {}
		for key in {key for keys in landing_names for key in keys}:
			for path in made.get(key, ()):
				directories.setdefault(places.bucket(path), []).append(path)
		kept = {}
		for order, (name, (dest, (path, job, _, landings))) in enumerate(shared):
			buckets = [places.bucket(land) for land in landings]
			if not os.path.islink(path):
				# In every destination: the file may go to any of them.
				verdicts = {}
				for land, key in zip(landings, buckets):
					for place in directories.get(key, ()):
						verdicts.setdefault(places.same_place(land, place), (land, place))
				if True in verdicts:
					clash(path, job, f'{verdicts[True][0]} is a directory of another source where the destination folds case')
					continue
				if None in verdicts:
					land, place = verdicts[None]
					clash(path, job, f'{land} and {place} may be one name, and the destination cannot be asked: {places.cannot_ask(land, place)}')
					continue
			# The first, in order, that is one name with it in any destination, or may be.
			firsts = sorted({first[0]: first for index, key in enumerate(buckets) for first in kept.get((index, key), ())}.values())
			if not any(alike(path, job, list(zip(first_landings, landings))) for _, first_landings in firsts):
				for index, key in enumerate(buckets):
					kept.setdefault((index, key), []).append((order, landings))
	return dropped, replaced, in_place

def process_copy(src_paths: list, dests:list = [], single_thread = False, max_workers = multiprocessing.cpu_count(),verbose = False, 
				directory_only = False,no_directory_sync = False, full_hash = False, files_per_job = 1, parallel_file_listing = False,
				exclude=None,dest_image = None,batch = False, exit_not_enough_space = False, dropped = None, in_place = ()):
	"""dropped, in_place: what _settle_destinations decided (hpcp() settles them first). Returns (the files, {link: dests},
	the directory sync's records for the finish: finish_directories)."""
	total_file_list = set()
	total_sym_links = {}
	dir_records = []
	taskCtr = 0
	argDest = dests.copy()
	if batch:
		print(f"Copying from {_sources_text(src_paths)} to {dests}")
	max_workers = max_workers if not single_thread else 1
	cs = copy_scheduler(max_workers=max_workers,full_hash=full_hash,verbose=verbose,files_per_job=files_per_job,parallel_file_listing=parallel_file_listing,exclude=exclude,exit_not_enough_space=exit_not_enough_space,
		dropped=dropped)
	synced_parents = set()
	for source in src_paths:
		taskCtr += 1
		src, dests, _ = _copy_job(source, argDest, dest_image)
		if not dests:
			# Every destination is a device it would replace (_settle_destinations said so).
			continue
		if src in (dropped or {}).get((src, tuple(dests)), ()):
			# A named source that was refused has nothing left to copy: listing it would only report it a second time.
			continue
		if (src, tuple(dests)) in in_place:
			# It lands on itself. Another source may be writing these very files: not read, but -rme has to know them.
			print(f"{src} lands on itself: nothing to copy")
			files, links = get_file_list(src, max_workers=max_workers, exclude=exclude, parallel_file_listing=parallel_file_listing and max_workers != 1)[:2]
			total_file_list.update(files)
			total_sym_links.update(_link_dests(links, src, dests))
			continue
		if dest_image:
			# if the destination is a mounted image, then we use the full path for src and add that to dest.
			# the dest path have the src full src path, we will also recersively copy the dir meta data
			# Gather all parent dirs of src
			srcParentDirs = []
			srcParent = os.path.dirname(os.path.abspath(src))
			while srcParent != '/':
				srcParentDirs.append(srcParent)
				srcParent = os.path.dirname(srcParent)
			srcParentDirs.reverse()
			# create the parent dirs in dest
			#destParentDirs = [ os.path.abspath(dest + srcParentDir + os.path.sep) for srcParentDir in srcParentDirs]
			dir_records += sync_directory_metadata_bulk(srcParentDirs,argDest,src_path='/')[3]
		if isinstance(source, _ListedSource):
			# The directories its path implies that are missing take its sources' mode and owner, so a private one stays
			# private. (With dest_image, above.) Its parents may not exist yet, so nothing to check for being writable.
			if not (dest_image or no_directory_sync or NO_CREATE_DIR or (exclude and is_excluded(src, exclude))):
				dir_records += _sync_implied_parents(source, argDest, synced_parents)
		else:
			# verify dest is writable
			for dest in dests:
				if not os.access(os.path.dirname(os.path.abspath(dest)), os.W_OK):
					print(f"Destination {dest} is not writable, continue with caution.")
		if not batch:
			print('-'*80)
			print(f"Task {taskCtr} of {len(src_paths)}, copying from {src} to {dests}")
		if os.path.islink(src):
			# A copy job of its own, so that it takes its turn against the other sources (_settle_destinations).
			print(f"{src} is a symlink, creating symlink in dests")
			cs.add_copy(src, dests)
		else:
			# if os.path.isfile(src):
			# 	print("Copying single file")
			# 	copy_file(src, dests,full_hash=full_hash,verbose=verbose)
			# 	continue
			if os.path.isdir(src):
				if no_directory_sync:
					print("Skipping directory sync")
					if not NO_CREATE_DIR:
						for dest in dests:
							try:
								_close_base(_make_dest_directory(src, dest))
							except OSError as e:
								eprint(f'Directory sync error: {src} to {dest}: {e}')
				else:
					cs.add_dir_sync(src, dests)
			if not directory_only:
				cs.add_copy(src, dests)
		if not batch:
			fl, sl, dr = cs.process()
			total_file_list.update(fl)
			total_sym_links.update(sl)
			dir_records += dr
			cs.clear()
			print('-'*80)
	if batch:
		fl, sl, dr = cs.process()
		total_file_list.update(fl)
		total_sym_links.update(sl)
		dir_records += dr
		cs.clear()
		print(f"Processed {len(total_file_list)} files and {len(total_sym_links)} symlinks in total")
		print('-'*80)
	return total_file_list, total_sym_links, dir_records

def validate_dd_source_path(src_path,loop_devices = None):
	# `is None`, not `if not loop_devices`: an empty list is falsy, so the old
	# check rebound the parameter to a fresh local list whenever the caller passed
	# one that was still empty - which is exactly what hpcp() does. The source loop
	# was then appended to a list nobody owned and clean_up() never detached it,
	# leaking one read-only loop device per -dd run. Matches clean_up() below.
	if loop_devices is None:
		loop_devices = []
	dd_src = src_path
	if not dd_src:
		raise RuntimeError("No DD source: DD Source not specified, exiting.")
	if len(dd_src) != 1:
		raise RuntimeError("Too many DD sources: DD Source is not 1, exiting.")
	dd_src = dd_src[0]
	if not os.path.exists(dd_src):
		raise RuntimeError(f"DD source not found: DD Source {dd_src} does not exist, exiting.")
	# check if dd_src is a block device
	if not pathlib.Path(dd_src).resolve().is_block_device():
		# check if dd_src is a file
		if not os.path.isfile(dd_src):
			raise RuntimeError(f"DD source type error: DD Source {dd_src} is not a block device or a file, exiting.")
		# mount as a loop device
		print(f"DD Source {dd_src} is a file, mounting as a loop device")
		dd_src = create_loop_device(dd_src,read_only=True)
		loop_devices.append(dd_src)
	return dd_src

# Bytes reserved on a destination image for GPT's own structures: the primary
# header + partition array at the head (and the 1M alignment before the first
# partition), and the backup header in the final 33 sectors.
_GPT_STRUCTURE_RESERVE = 2 * 1024 * 1024

def create_dd_dest_part_table(dd_src,dd_resize = [],src_path = None, dest_path = None, dest_image_size=0):
	"""Create / resize dest image and write a matching partition table.

	Returns:
		tuple: (partition_infos, delayed_commands, loop_device). On cancel / error setup,
			returns (None, [], None). loop_device is owned by the caller when not None.
	"""
	src_path = src_path if src_path else dd_src
	if not dest_path:
		print("Destination path not specified.")
		return None, [], None
	partition_infos = get_partition_infos(dd_src)
	disk_name = dd_src
	if len(partition_infos) == 1:
		print(f"Source device {dd_src} is not partitioned, exiting.")
		return None, [], None
	# sort the partitions by size
	disk_info = partition_infos.pop(disk_name)
	sorted_partitions = sorted(partition_infos.keys(), key=lambda x: partition_infos[x]['size'])
	# change the partitions sizes if provided
	if dd_resize:
		# dd resize is a list of sizes for each partition, we sort it first
		dd_resize = [format_bytes(size,to_int=True) for size in dd_resize]
		dd_resize = sorted(dd_resize,reverse=True)
		# we resize the partitions according to the dd_resize list, 
		# we use as much info from the dd_resize list as possible, assuming it specifies the sizes of the partitions from the largest to the smallest
		# which means if it specified two sizes, we resize the two largest partitions to the specified sizes
		# largest_partition = sorted_partitions[-2]
		# partition_infos[largest_partition]['size'] = format_bytes(dd_resize,to_int=True)
		for i in range(min(len(dd_resize),len(sorted_partitions))):
			partition_infos[sorted_partitions[-i-1]]['size'] = dd_resize[i]

		# recaclulate the disk size, also include 1M extra for each partition 
	# ...plus room for GPT itself. GPT keeps a primary header and partition array
	# at the start of the disk and a backup copy in the last 33 sectors, and sgdisk
	# refuses to create a partition that would run into either. The per-partition
	# 1M slack above is consumed by the 1M start alignment, so nothing reserved the
	# tail: a single partition spanning its source disk could not be recreated at
	# all ("Could not create partition 1 from ...", sgdisk rc 4). Verified: a lone
	# partition of size S needs S + 2M, where S + 1M fails.
	disk_info['size'] = sum([partition_infos[partition]['size'] for partition in partition_infos]) + 1024*1024*len(partition_infos) + _GPT_STRUCTURE_RESERVE
	disk_info['size'] = _apply_dis_to_dd_image(disk_info['size'], dest_image_size, dest_path)
	partition_infos[disk_name] = disk_info
	sorted_partitions.append(disk_name)

	# if dest_path exists, print a confirmation message
	if os.path.exists(dest_path):
		inStr = ''
		if sys.stdin.isatty():
			print(f"Warning: Destination path {dest_path} exists.")
			print(f"Source device {src_path} will be copied to {dest_path} with the following partition info:")
			print(partition_infos)
			print(f"All data on {dest_path} will be lost, do you want to continue? (y/n) Default : y")
			inStr = multiCMD.input_with_timeout_and_countdown(60)
		if inStr and not inStr.lower().startswith('y'):
			print("Exiting.")
			return None, [], None
	resize_image(dest_path, partition_infos[disk_name]['size'])
	delayed_commands, loop_device = create_partition_table(dest_path,partition_infos,sorted_partitions)
	print("Image created successfully.")
	return partition_infos, delayed_commands, loop_device

def dd_partition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest):
	# First verify the two partition size is the same
	src_part_info = get_partition_infos(dd_src)
	dest_part_info = get_partition_infos(dd_dest)
	if src_part_info[partition]['size'] > dest_part_info[partition]['size']:
		raise RuntimeError(f"Source partition larger than destination partition: {src_part_info[partition]['size']} > {dest_part_info[partition]['size']}, cannot use DD, exiting.")
	run_command_in_multicmd_with_path_check(['dd','if='+src_partition_path,'of='+dest_partition_path,'bs=1024M'],timeout=0,strict=True)

def _syncfs(path):
	"""Write the filesystem path is on, and no other: syncfs(2), through ctypes (Python has no os.syncfs). os.sync and the sync
	command write every filesystem on the host. No libc, no syncfs in it, or an error is a flush that was not made, which a
	run does without (as a copy with cp does). Not for a move: before Linux 5.8 syncfs reports no write error at all (see
	_fsync_file, which a move uses on each copy)."""
	try:
		import ctypes
		libc = ctypes.CDLL(None, use_errno=True)
		fd = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
		try:
			libc.syncfs(fd)
		finally:
			os.close(fd)
	except (OSError, AttributeError, ImportError):
		pass

# fsync's errors that say the file cannot be written to disk this way at all (some FUSE filesystems), not that it failed:
# the copy goes on without it, as it did before 9.85.
_NO_FSYNC = frozenset(getattr(errno, name) for name in ('EINVAL', 'ENOTSUP', 'EOPNOTSUPP', 'ENOSYS') if hasattr(errno, name))

def _fsync_file(path, base=None, name=None):
	"""Write the data of path, a regular file this run made (a temporary copy about to be renamed over a landing, or a move's
	landing), to disk: renamed unsynced, a power cut can leave the landing empty where the old copy was (XFS does not flush on
	such a rename, and no filesystem orders the data before the rename); and a move's source, removed after an unsynced copy,
	is gone with it. On ext4 and XFS it also commits the transaction that made the file's name, and every earlier one (its
	directories), so no directory is fsynced. Raises OSError when the filesystem fails it, or the file cannot be opened; returns
	when the filesystem cannot fsync at all (_NO_FSYNC). A directory is fsynced as a file is (the directories a move made
	links in: create_sym_links). Anything else returns at once: a FIFO or node made in its place (cp -a of one)
	has no data, and opening a FIFO waits for a writer.

	A file whose mode lets its owner neither read nor write it (a copy of another user's file readable by group or others
	only, mode 0044) cannot be opened as it is: the owner (this run's effective uid) is given read for the open, and the exact
	mode is put back through the descriptor (fchmod) before the fsync, so the file has the wider mode for microseconds, never
	afterwards. Anyone else who cannot open it gets the error. Off POSIX it returns at once. base, name: path is name in base,
	the door's result for its directory (_dest_directory), and every call goes relative to it."""
	if os.name != 'posix':
		return  # Windows' fsync wants a descriptor open for writing, and nothing here has been run there
	at, how = ((path,), {}) if base is None else (_at(base, name), _fd_of(base))
	st = os.lstat(*at, **how)
	directory = stat.S_ISDIR(st.st_mode)
	if not (directory or stat.S_ISREG(st.st_mode)):
		return
	flags = getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
	try:
		fd = os.open(*at, os.O_RDONLY | flags, **how)
	except PermissionError as first:
		try:
			if directory:
				raise first  # a directory opens for reading only
			fd = os.open(*at, os.O_WRONLY | flags, **how)
		except PermissionError:
			if st.st_uid != os.geteuid():
				if directory:
					# Another user's directory this run may not read (a write-only drop box): not this run's to open.
					return
				raise first
			mode = stat.S_IMODE(st.st_mode)
			os.chmod(*at, mode | stat.S_IRUSR, **how)
			try:
				fd = os.open(*at, os.O_RDONLY | flags, **how)
			except BaseException:
				os.chmod(*at, mode, **how)
				raise
			try:
				os.fchmod(fd, mode)
			except BaseException:
				os.close(fd)
				raise
	try:
		os.fsync(fd)
	except OSError as e:
		if e.errno not in _NO_FSYNC:
			raise
	finally:
		os.close(fd)

class _NotOnDisk(Exception):
	"""A move's copy that the filesystem failed to write to disk (_sync_landing): a failed copy, so no source is removed."""

def _sync_landing(dest_path, base=None):
	"""For a move (SYNC_COPIES): the copy at dest_path on disk before its file is reported copied, so no source goes while the
	copy may be only in memory (a power cut then loses both), and a write error the filesystem reports is met here, on the
	descriptor of the copy's own file while its data is still dirty (a writeback error is kept only on the in-memory inode,
	and one evicted under memory pressure takes it along; syncfs before Linux 5.8 reports none). Raises _NotOnDisk, naming the
	landing and why, after removing the regular file there: its content cannot be trusted, and a rerun would take it for the
	finished copy (same size and time). Only the file the copy wrote is removed: the one at the name when the fsync began (one
	lstat right after the copy); a file put at the name since is someone else's. A block device named by -d is not fsynced
	(_fsync_file passes it): a move onto one writes it and keeps every source (213: a landing that is another kind of entry holds
	no source), so nothing can be lost. base: the door's result for the landing's directory (None: its path), every call
	relative to it."""
	target = _landing_path(dest_path)[0]
	base = os.path.dirname(target) if base is None else base
	name = os.path.basename(target)
	try:
		written = os.lstat(*_at(base, name), **_fd_of(base))
	except OSError:
		written = None
	try:
		_fsync_file(target, base, name)
	except OSError as e:
		try:
			now = os.lstat(*_at(base, name), **_fd_of(base))
			if written is not None and stat.S_ISREG(now.st_mode) and (now.st_dev, now.st_ino) == (written.st_dev, written.st_ino):
				os.remove(*_at(base, name), **_fd_of(base))
		except OSError:
			pass
		raise _NotOnDisk(f'{dest_path} cannot be written to disk ({_why(e)})')

def _flush_loop_device(loop_device, sysfs='/sys/block'):
	"""
	Flush what a loop device holds: the device, then the file behind it (from /sys, read while the device is attached),
	so an image's clone is on disk when the run ends. A global sync(2) would also write every other filesystem on the
	host. An OSError is ignored: a flush that cannot be made is what the detach would have gone without.
	"""
	paths = [loop_device]
	try:
		# As bytes: a name that is not valid in the locale is still the name (text mode raises UnicodeDecodeError).
		with open(os.path.join(sysfs, os.path.basename(loop_device), 'loop', 'backing_file'), 'rb') as f:
			paths.append(f.read().rstrip(b'\n'))
	except OSError:
		pass
	for path in paths:
		try:
			fd = os.open(path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
			try:
				os.fsync(fd)
			finally:
				os.close(fd)
		except OSError:
			pass

def detach_loop_device(loop_device, retries=5, delay=0.5):
	"""Flush a loop device and its backing file, then detach it, with retries for intermittent EBUSY."""
	if not loop_device:
		return True
	losetup = _bin('losetup')
	_flush_loop_device(loop_device)
	for attempt in range(1, retries + 1):
		if not os.path.exists(loop_device):
			print(f"Loop device {loop_device} already gone.")
			return True
		if attempt > 1:
			# Give udev / partition holders a moment to release after a busy failure.
			if 'udevadm' in _binPaths:
				multiCMD.run_commands([[_bin('udevadm'), 'settle', '--timeout=5']], timeout=10, max_threads=1, quiet=True)
			time.sleep(delay * attempt)
		task = multiCMD.run_commands([[losetup, '-d', loop_device]], timeout=COMMAND_TIMEOUT, max_threads=1, quiet=True, return_object=True)[0]
		if task.returncode == 0:
			if attempt > 1:
				print(f"Detached loop device {loop_device} on attempt {attempt}.")
			return True
		err = (task.stderr[-1].strip() if task.stderr else '') or f"return code {task.returncode}"
		eprint(f"Failed to detach {loop_device} (attempt {attempt}/{retries}): {err}")
	eprint(f"Giving up detaching loop device {loop_device}")
	return False

def clean_up(mount_points: list,loop_devices: list, delayed_commands: list = None):
	# clean up loop devices and mount points if we are using a image
	if delayed_commands is None:
		delayed_commands = []
	# Last mount first. A mount made after a bind of a live filesystem, with its directory on that filesystem, also lands
	# inside the bind's copy in another mount namespace (a PrivateTmp service's). Unmounted first, that bind's copy
	# keeps its child and stays: on Linux 3.10 (EL7) for good, with rmdir failing EBUSY and its device held.
	for mount_point in reversed(list(mount_points)):
		print(f"Unmounting {mount_point}")
		run_command_in_multicmd_with_path_check(["umount",mount_point],strict=False)
		if _is_mount_point(mount_point):
			# Still busy; retry once before leaving it for loop detach retries.
			time.sleep(0.5)
			run_command_in_multicmd_with_path_check(["umount",mount_point],strict=False)
		print(f"Removing mount point {mount_point}")
		try:
			os.rmdir(mount_point)
		except OSError as e:
			eprint(f"Failed to remove mount point {mount_point}: {e}")
	mount_points.clear()
	# Fold any delayed losetup --detach into loop_devices so they get flush/retry handling.
	remaining_commands = []
	for cmd in delayed_commands:
		if cmd and len(cmd) >= 3 and cmd[0] == 'losetup' and cmd[1] in ('-d', '--detach'):
			if cmd[2] not in loop_devices:
				loop_devices.append(cmd[2])
		else:
			remaining_commands.append(cmd)
	delayed_commands.clear()
	if remaining_commands:
		print("Running delayed commands")
		# Serial on purpose: write_partition_info queues e2fsck then tune2fs -U
		# as ordered pairs, and a parallel run would let tune2fs race the fsck.
		run_commands_in_multicmd_with_path_check(remaining_commands,strict=False,max_threads=1)
	for loop_device_dest in list(loop_devices):
		print(f"Removing loop device {loop_device_dest}")
		detach_loop_device(loop_device_dest)
	loop_devices.clear()

HASH_SIZE = 1<<24

def get_args(args = None):
	parser = argparse.ArgumentParser(description='Copy files from source to destination',
								  epilog=f'Found bins: {list(_binPaths.values())}\n Missing bins: {_binCalled - set(_binPaths.keys())}')
	parser.add_argument('-s', '--single_thread', action='store_true', help='Use serial processing')
	parser.add_argument('-j','-m','-t','--max_workers', type=int, default=None, help='Max workers for parallel processing. Default is 1 * CPU count, and at least 16 for -rm and -rf (a move, or a removal without -d), whose workers wait on their fsyncs. Use negative numbers to indicate {n} * CPU count, 0 means 1/2 CPU count.')
	batch_group = parser.add_mutually_exclusive_group()
	batch_group.add_argument('-b','--batch',action='store_true', help='Batch mode, process all files in one go',default=True)
	batch_group.add_argument('-nb','--no_batch','--sequential',action='store_false', dest='batch', help='Do not use batch mode', default=False)
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
	parser.add_argument('-do', '--directory_only', action='store_true', help='Only copy directory structure')
	parser.add_argument('-nds', '--no_directory_sync', action='store_true', help='Do not sync directory metadata, useful for verfication')
	parser.add_argument('-fh', '--full_hash', action='store_true', help='Checks the full hash of files')
	parser.add_argument('-hs', '--hash_size', type=int, default=1<<16, help='Hash size in bytes, default is 65536. This means hpcp will only check the last 64 KiB (about 1 page in a SSD) of the file.')
	parser.add_argument('-fpj', '--files_per_job', type=int, default=1, help='Base number of files per job, will be adjusted dynamically. Default is 1')
	parser.add_argument('-sfl','-lfl', '--source_file_list', type=str, help='Load source paths from a file, or use "-" to read them from stdin (read by default when no source is given and stdin is not a terminal: find src -type f | hpcp -d out). Each path lands in the destination as given, minus a leading / and any .. (cd src && find . | hpcp -d ../out to leave src out); a listed folder copies recursively. Paths are not expanded and are separated by newlines, or by NUL bytes when the data holds one (find -print0, fd -0). If --compare_file_list is specified, it will be used as source for compare')
	parser.add_argument('-fl','-tfl', '--target_file_list', type=str,help='Specify the file_list file to store list of files in src_path to. If --compare_file_list is specified, it will be used as targets for compare')
	parser.add_argument('-cfl', '--compare_file_list',action='store_true', help='Only compare file list. Use --file_list to specify a existing file list or specify the dest_path to compare src_path with. When not using with file_list, will compare hash.')
	parser.add_argument('-dfl', '--diff_file_list', type=str, nargs='?', const="auto",default=None, help="Implies --compare_file_list, specify a file name to store the diff file list to or omit the value to auto-determine.")
	parser.add_argument('-tdfl', '--tar_diff_file_list', action='store_true', help='Generate a tar compatible diff file list. ( update / new files only )')
	parser.add_argument('-nhfl', '--no_hash_file_list', action='store_true', help='Do not append hash to file list')
	parser.add_argument('-rm', '--remove', action='store_true', help='Remove all files and folders specified in src_path')
	parser.add_argument('-rf', '--remove_force', action='store_true', help='Remove all files without prompt')
	parser.add_argument('-rme', '--remove_extra', action='store_true', help='Remove the files and folders the sources do not copy, below where each directory source lands (all of out/ for src/, out/b for b); never a source, or what lies in or holds one')
	parser.add_argument('-rwloff','--do_not_remove_files_while_listing', action='store_true', help='With -rm and no destination: list the trees first, then remove the listed files in parallel jobs (default: remove while walking the trees).')
	parser.add_argument('-e', '--exclude', action='append', default=[], help='Exclude source files matching the pattern')
	parser.add_argument('-x', '--exclude_file', type=str, help='Exclude source files matching the pattern in the file')
	parser.add_argument('-lt', '--link_tracking', action='store_true', help='Also copy the target of a relative symlink when the destination does not have it, to the same place relative to the link (this can be outside the destination). Default: off, symlinks are copied as they are.')
	parser.add_argument('-nlt', '--no_link_tracking', action='store_true', help='Do not copy the files that symlinks point to (the default). Overrides -lt.')
	parser.add_argument('-V', '--version', action='version', version=f"%(prog)s {version} @ {COMMIT_DATE} with {('XXHash' if xxhash_available else 'Blake2b')}, multiCMD V{multiCMD.version}, and [ {', '.join(_binPaths.keys())} ]; High Performance CoPy (HPC coPy) by pan@zopyr.us")
	parser.add_argument('-pfl', '--parallel_file_listing', action='store_true', help='Use parallel processing for file listing')
	parser.add_argument('src_path', nargs='*', type=str, help='Source Path')
	parser.add_argument('-si','--src_image',action='append', type=str, help='Source Image, mount the image and copy the files from it.')
	parser.add_argument('-siff','--load_diff_image',action='append', type=str, help='Apply a patch image made by -diff to the target (-d / -di or a single path). Repeat for each -dis volume. Every path the patch touches is checked first; conflicts prompt before being overridden. A disk patch (made with -dd -diff) takes the target disk with -d.')
	parser.add_argument('-d','-C','--dest_path',action='append', type=str, help='Destination Path. Repeat it to spill over: each file goes to one destination, the first with room, not to all of them.')
	parser.add_argument('-rds','--random_dest_selection', action='store_true', help='Randomly select destination path from the list of destination paths instead of the first with room. Can speed up transfer if dests are on different devices. Warning: can cause unable to fit in big files as dests are filled up by smaller files.')
	parser.add_argument('-di','--dest_image', type=str, help='Base name for destination Image, create a image file and copy the files into it.')
	parser.add_argument('-dis','--dest_image_size', type=str, help='Destination Image Size, specify the size of the destination image to split into. Default is 0 (No split). Example: {10TiB} or {1G}', default='0')
	parser.add_argument('-diff', '--get_diff_image', type=str, nargs='?', const='auto', default=None, help='Generate a patch image that updates the base (-d / -di) to the source (src_path / -si): added and modified files in full plus a change list under .hpcp_diff/. Optionally name the output; default DIFF_<src>_TO_<base>_<time>.img. Honors -dis, -fh, -hs and -e. With -dd, compares whole GPT disks: the new disk as the path and the old disk with -d.')
	parser.add_argument('-fa', '--force_apply', action='store_true', help='With -siff, apply even when the target differs from the state the patch expects. Conflicts are still printed as warnings.')
	parser.add_argument('-dd', '--disk_dump', action='store_true', help='Disk to Disk mirror, use this if you are backuping / deploying an OS from / to a disk. \
					 Require 1 source, can be 1 src_path or 1 -si src_image, require 1 -di dest_image. Note: will only actually use dd if unable to mount / create a partition.')
	parser.add_argument('-ddr', '--dd_resize', action='append', type=str, help='Resize the destination image to the specified size with -dd. Applies to biggest partiton first. Specify multiple -ddr to resize subsequent sized partitions. Example: {100GiB} or {200G}')
	parser.add_argument('-nfp','--no_fs_param_mirror', action='store_true', help='Do not mirror source filesystem parameters (geometry, features) in -dd mode. Create destination filesystems with mkfs defaults, preserving only label and UUID.')
	parser.add_argument('-L','-rl','--rate_limit', type=str, default=None, help='Approximate a rate limit the copy speed in bytes/second. Example: 10M for 10 MB/s, 1Gi for 1 GiB/s. Default is 0: no rate limit.')
	parser.add_argument('-F','-frl','--file_rate_limit', type=str, default=None, help='Approximate a rate limit the copy speed in files/second. Example: 10K for 10000 files/s, 1Ki for 1024 files/s. Default is 0: no rate limit.')
	parser.add_argument('-tfs','--target_file_system', type=str, default=None, help='Specify the target file system type. Will abort if the target file system type does not match. Example: ext4, xfs, f2fs, ntfs, fat32, exfat. Default is None: do not check target file system type.')
	parser.add_argument('-ncd','--no_create_dir', action='store_true', help='Ignore any destination folder that does not already exist. ( Will still copy if dest is a file )')
	parser.add_argument('-co','--content_only', action='store_true', help='Content-only copy: do not sync mode, owner, or timestamps on files or directories. A destination file with other hard links is replaced, not written through. A FIFO, socket or device node is made as cp -a makes it, never read. Still create missing destination directories using filesystem-default permissions (so parent ACL/setgid/setuid inherit).')
	parser.add_argument('-wb', '--write_beside', '--write-beside', action='store_true', help='Write every rewrite to a new file beside the old one, write it to disk and rename it over the old one once whole: a source that fails, a full disk or a stop leave the old copy as it was. The file gets a new inode (a per-file Lustre or GPFS layout, chattr flags and a project id are not kept; NFS readers holding it open may get ESTALE). Needs a new name in the file\'s directory and room for a second copy: without either the old copy stays (exit 190). By default a rewrite is written in place, and only a file with other names, one that cannot be opened for writing, or a FIFO, device or link goes beside.')
	parser.add_argument('-rbw', '--read_before_write', '--read-before-write', action='store_true', help='Before a file at the destination is written in place, read its source to the end: a source that fails on read leaves the old copy as it is (exit 190). Costs a second read of each rewritten file. Does nothing with -wb, which writes nothing in place.')
	parser.add_argument('-ctl','--command_timeout_limit', type=int, default=0, help='Set the command timeout limit in seconds for external commands ( ex. cp / dd ). Default is 0: no timeout.')
	parser.add_argument('-enes','--exit_not_enough_space', action='store_true', help='Skip a source whose copy does not fit the destinations\' free space, counting only what the copy will write (files already there and the same are not counted, a rewrite only what it adds); the run exits 228. Without it, a copy that may not fit is tried, with a warning on ext2/3/4, XFS and tmpfs (a compressing filesystem can fit more than its free space).')
	try:
		args = parser.parse_intermixed_args(args)
	except Exception:
		# try to parse the arguments using parse_known_args
		args, unknown = parser.parse_known_args(args)
		# if there are unknown arguments, we will try to parse them again using parse_args
		if unknown:
			print(f"Warning: Unknown arguments, treating all as Source Path: {unknown!r}")
			args.src_path = args.src_path + unknown
	#print(f'Arguments: {vars(args)}')
	defualt_args_dict = vars(parser.parse_args([]))
	# format a long format argument of what the user supplied and echo it back
	startArgs = [f'> {sys.argv[0]}']
	for values in args.src_path:
		startArgs.append(f'\'{values}\'')
	for argumentName, value in vars(args).items():
		if value != defualt_args_dict[argumentName]:
			if argumentName == 'src_path':
				continue
			if isinstance(value, list):
				# skip positional arguments
				for v in value:
					startArgs.append(f'--{argumentName}=\'{v}\'')
			else:
				startArgs.append(f'--{argumentName}=\'{value}\'')
	print(' '.join(startArgs))
	return args

#%% ---- Main Function ----
def hpcp(src_path, dest_paths = [], single_thread = False, max_workers = None,
			verbose = False, directory_only = False,no_directory_sync = False, full_hash = False, files_per_job = 1, target_file_list = "",
			compare_file_list = False, diff_file_list = None, tar_diff_file_list = False, remove = False,remove_force = False, remove_extra = False, parallel_file_listing = False,
			exclude=None,exclude_file = None,dest_image = None,dest_image_size = '0', no_link_tracking = False,link_tracking = False,src_image = None,dd = False,dd_resize = 0,
			batch = False, append_hash_to_file_list = True, hash_size = ..., source_file_list = None, random_destination_selection = False, 
			bytes_rate_limit = None, files_rate_limit = None,target_file_system = None, no_create_dir = False, content_only = False, command_timeout_limit = 0,
			exit_not_enough_space = False,do_not_remove_files_while_listing = False,no_fs_param_mirror = False,
			get_diff_image = None, load_diff_image = None, force_apply = False, write_beside = False, read_before_write = False):
	global HASH_SIZE
	global RANDOM_DESTINATION_SELECTION
	global BYTES_RATE_LIMIT
	global FILES_RATE_LIMIT
	global COMMAND_TIMEOUT
	global NO_CREATE_DIR
	global CONTENT_ONLY
	global WRITE_BESIDE
	global READ_FIRST
	global MIRROR_FS_PARAMS
	global REMOVE_FILES_WHILE_LISTING
	NO_CREATE_DIR = no_create_dir
	CONTENT_ONLY = content_only
	WRITE_BESIDE = write_beside
	READ_FIRST = read_before_write
	MIRROR_FS_PARAMS = not no_fs_param_mirror
	remove = remove or remove_force
	if max_workers is None:
		# A move's worker waits on each file's fsync before its next copy: at least 16 workers keep the disk busy (the 9.85
		# review's -j 16 on 2- and 4-vCPU VMs: 2,000 files of 4 KiB in about half the time of -j 2 or 4). The fsyncs in flight are
		# the disk's need, not the CPUs': on 32 CPUs and a fast disk, 4 a CPU (128 workers) made a 2,000-file move 20% (3.14) to
		# 45% (3.6) slower than 32, and 3.6 to 3.8 start every worker of a pool at once (a 100-file move: 0.60 s against 0.32 s).
		max_workers = max(16, multiprocessing.cpu_count()) if remove else multiprocessing.cpu_count()
	if random_destination_selection:
		RANDOM_DESTINATION_SELECTION = True
		print("Random destination selection enabled.")
	else:
		RANDOM_DESTINATION_SELECTION = False
	if bytes_rate_limit:
		BYTES_RATE_LIMIT = format_bytes(bytes_rate_limit,to_int=True)
	if files_rate_limit:
		FILES_RATE_LIMIT = format_bytes(files_rate_limit,to_int=True)
	if hash_size != ...:
		try:
			HASH_SIZE = int(hash_size)
		except TypeError:
			print(f"Invalid hash size {hash_size}, using default hash size {HASH_SIZE}")
	if HASH_SIZE < 0:
		HASH_SIZE = 0
	if HASH_SIZE == 0:
		print("Warning: Hash size set to 0, will not check file content for skipping.")
	try:
		dest_image_size = format_bytes(dest_image_size,to_int=True)
	except Exception:
		print(f"Invalid destination image size {dest_image_size}, using default size 0")
		dest_image_size = 0
	COMMAND_TIMEOUT = command_timeout_limit if command_timeout_limit > 0 else 0
	REMOVE_FILES_WHILE_LISTING = not do_not_remove_files_while_listing
	print('-'*80)
	global _STDIN_HELD_LIST
	_STDIN_HELD_LIST = False
	src_paths = []
	src_image_roots = []
	mount_points = []
	loop_devices = []
	delayed_commands = []
	src_str = ''
	programStartTime = time.monotonic()
	exclude = format_exclude(exclude,exclude_file)
	if dest_image:
		dest_image = os.path.expanduser(dest_image)
	if max_workers == 0:
		max_workers = round(0.5 * multiprocessing.cpu_count())
	elif max_workers < 0:
		max_workers = round(- max_workers * multiprocessing.cpu_count())
	# This run's space-warning slots (_space_warning); a nested run (-dd) has its own, and gives the outer one its back. Made
	# with the stops blocked in the process that runs main(), as the pools' are (_pool): its lock is the run's first SemLock,
	# and under forkserver or spawn Python 3.6 and 3.7 start their semaphore tracker for it with nothing blocked, where an
	# early Ctrl-C printed the tracker's traceback (3.8's resource tracker blocks them itself). The tracker keeps them blocked.
	global _SPACE_WARNED
	mask = _block_stops() if os.getpid() == _MAIN_PID else None
	try:
		share = _space_share() if os.name == 'posix' else None
	finally:
		_restore_stops(mask)
	outer_space, _SPACE_WARNED = _SPACE_WARNED, share
	try:
		if get_diff_image or load_diff_image or force_apply:
			if os.name == 'nt':
				raise RuntimeError("Platform error: patch images are not supported on Windows, exiting")
			get_diff_image, src_path = _validate_diff_args(get_diff_image, load_diff_image, force_apply, src_path, src_image,
				dest_paths, dest_image, dd=dd, remove=remove, remove_extra=remove_extra, target_file_list=target_file_list,
				compare_file_list=compare_file_list, diff_file_list=diff_file_list, tar_diff_file_list=tar_diff_file_list,
				content_only=content_only)
			workers = 1 if single_thread else max_workers
			if load_diff_image:
				process_load_diff_image(load_diff_image, src_path, dest_paths, dest_image, mount_points, loop_devices,
					force=force_apply, max_workers=workers, verbose=verbose, dd=dd)
				raise RuntimeError("Exiting after applying patch image.")
			if dd:
				process_get_disk_diff_image(src_path, dest_paths, get_diff_image, mount_points, loop_devices,
					exclude=exclude, full_hash=full_hash, dest_image_size=dest_image_size, max_workers=workers,
					parallel_file_listing=parallel_file_listing, verbose=verbose)
				raise RuntimeError("Exiting after generating disk patch image.")
			process_get_diff_image(src_path, src_image, dest_paths, dest_image, get_diff_image, mount_points, loop_devices,
				exclude=exclude, full_hash=full_hash, dest_image_size=dest_image_size, max_workers=workers,
				parallel_file_listing=parallel_file_listing, verbose=verbose)
			raise RuntimeError("Exiting after generating patch image.")
		if dd:
			if os.name == 'nt':
				raise RuntimeError("Platform error: dd mode is not supported on Windows, exiting")
			# check if running as root
			if os.geteuid() != 0:
				eprint("Permission warning: dd mode likely requires root privileges, please expect errors. Continuing in 5 seconds...")
				time.sleep(5)
			print("dd mode enabled, performing Disk Dump Copy. Setting up the target ...")
			if dest_paths:
				if len(dest_paths) > 1:
					print("Destination path is not 1, taking the first destination path as image file.")
				dest_path = dest_paths[0]
			else:
				dest_path = dest_image
			src_path = src_path if src_path else src_image

			if dest_image_size:
				print(f"Applying dest_image_size {format_bytes(dest_image_size)}B to -dd destination image.")
			if link_tracking and not no_link_tracking:
				# Each partition is copied on its own: a tracked target could lie outside it (or outside the disk).
				print("Note: -lt is ignored with -dd: each partition's symlinks are copied as they are.", file=sys.stderr)
				
			if not dest_path and src_path:
				eprint(f"Destination path not specified, using {src_path[-1]} as destination")
				dest_path = src_path[-1]
				del src_path[-1]

			# check write permission on dest_path
			dest_path = os.path.expanduser(dest_path)
			if not os.access(os.path.dirname(os.path.abspath(dest_path)), os.W_OK):
				print(f"Destination path {dest_path} is not writable, continuing with high probability of failure.")
			dd_src = validate_dd_source_path(src_path,loop_devices = loop_devices)
			partition_infos, delayed_commands, dest_loop = create_dd_dest_part_table(dd_src,dd_resize=dd_resize,src_path=src_path, dest_path=dest_path, dest_image_size=dest_image_size)
			if not partition_infos:
				raise RuntimeError("Copy partition info error: Failed to create destination partition table, exiting.")
			
			# Reuse the loop created while writing the partition table; do not attach the same image twice.
			if dest_loop:
				dd_dest = dest_loop
				if dest_loop not in loop_devices:
					loop_devices.append(dest_loop)
					print(f"Reusing loop device {dest_loop} for DD destination {dest_path}")
			else:
				dd_dest = dest_path
				# check if dd_dest is a block device
				if not pathlib.Path(dd_dest).resolve().is_block_device():
					# check if dd_dest is a file
					if os.path.isfile(dd_dest):
						print(f"DD Destination {dd_dest} is a file, mounting as a loop device")
						dd_dest = create_loop_device(dd_dest)
						loop_devices.append(dd_dest)
			
			_ = partition_infos.pop(dd_src)
			# need to check if partion info is empty and fix partition table if necessary
			src_partition_paths = get_partitions(dd_src)
			dest_partition_paths = get_partitions(dd_dest)
			for partition in partition_infos:
				print(f"Copying partition {partition} from {dd_src} to {dd_dest}")
				# mount both the src and dest to a temporary folder
				src_mount_point = tempfile.mkdtemp()
				mount_points.append(src_mount_point)
				dest_mount_point = tempfile.mkdtemp()
				mount_points.append(dest_mount_point)
				src_partition_path = [path for path in src_partition_paths if path.endswith(partition)][0]
				dest_partition_path = [path for path in dest_partition_paths if path.endswith(partition)][0]
				print(f"Mounting {src_partition_path} at {src_mount_point}")
				run_command_in_multicmd_with_path_check(_dd_source_mount_command(src_partition_path, src_mount_point))
				# check if the mount is successful
				if not any(os.scandir(src_mount_point)) and not os.path.ismount(src_mount_point) and is_device(src_partition_path):
					# if the source is a device and currently mounted, try to find the current mount point, then try to bind mount it
					mtab = get_mount_table()
					if src_partition_path in mtab:
						src_partition_mount_path = mtab[src_partition_path][0]
						eprint(f"Error mounting {src_partition_path}, trying to use bind mount {src_partition_mount_path}")
						# try to bind mount
						run_command_in_multicmd_with_path_check(['mount','--bind',src_partition_mount_path,src_mount_point],strict=False)
				if not any(os.scandir(src_mount_point)) and not os.path.ismount(src_mount_point):
					eprint(f"Error mounting {src_partition_path}, using dd for copying.")
					dd_partition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest)
					continue
				print(f"Mounting {dest_partition_path} at {dest_mount_point}")
				run_command_in_multicmd_with_path_check(_rw_mount_command(dest_partition_path, dest_mount_point),strict=False)
				# check if the mount is successful
				if not any(os.scandir(dest_mount_point)) and not os.path.ismount(dest_mount_point) and is_device(dest_partition_path):
					mtab = get_mount_table()
					if dest_partition_path in mtab:
						dest_partition_mount_path = mtab[dest_partition_path][0]
						eprint(f"Error mounting {dest_partition_path}, trying to use bind mount {dest_partition_mount_path}")
						# try to bind mount
						run_command_in_multicmd_with_path_check(['mount','--bind',dest_partition_mount_path,dest_mount_point],strict=False)
				if not any(os.scandir(dest_mount_point)) and not os.path.ismount(dest_mount_point):
					eprint(f"Error mounting {dest_partition_path}, using dd for copying.")
					dd_partition(src_partition_path,dest_partition_path,partition,dd_src,dd_dest)
					continue
				# copy the partition files
				print(f"Copying partition {partition} files from {src_mount_point} to {dest_mount_point}")
				# The nested run judges (and clears) ERRORS: give it a clean list, so only this partition's copy decides
				# copy_rc, and put this run's own errors back afterwards.
				outer_errors = ERRORS[:]
				ERRORS.clear()
				try:
					copy_rc = hpcp([src_mount_point], dest_paths = [dest_mount_point], single_thread=single_thread, max_workers=max_workers,
						verbose=verbose, directory_only=directory_only,no_directory_sync=no_directory_sync,full_hash=full_hash, 
						files_per_job=files_per_job, parallel_file_listing=parallel_file_listing,exclude=exclude,no_link_tracking = True,
						batch=batch,append_hash_to_file_list = append_hash_to_file_list, hash_size = hash_size, source_file_list = source_file_list,
						random_destination_selection = random_destination_selection, bytes_rate_limit = bytes_rate_limit, files_rate_limit = files_rate_limit,
						target_file_system = target_file_system, no_create_dir = no_create_dir, content_only = content_only, command_timeout_limit = command_timeout_limit,
						exit_not_enough_space = exit_not_enough_space, write_beside = write_beside, read_before_write = read_before_write)
				finally:
					ERRORS[:0] = outer_errors
				if copy_rc:
					raise RuntimeError(f"Partition copy error: Copying partition {partition} failed with return code {copy_rc}.")
			# sort the output partitions
			#run_command_in_multicmd_with_path_check(f"sgdisk --sort {dest_path}")
			print(f"Done disk dumping {src_path} to {dest_path}.")
			raise RuntimeError("Exiting after dd mode.")
			
		# set max_workers to 61 on windows
		if os.name == 'nt':
			if dest_image or src_image:
				raise RuntimeError("Platform error: Destination / Source as a image is currently not supported on Windows, exiting")
			max_workers = min(max_workers,61)
			if max_workers == 61:
				eprint("Max workers set to 61 on Windows")
				eprint("This is because Windows has a limit of 64 threads per process and we need 3 threads for the main process")
				print("See https://bugs.python.org/issue26903")

		# if not dest_path:
		#     dest_path = src_path.pop()
		if src_image:
			if os.geteuid() != 0:
				eprint("Warning: Source image mount may require root privileges, please expect errors. continuing in 5 seconds...")
				time.sleep(5)
			src_images = expand_paths(src_image)
			src_str = mount_src_image(src_images,src_paths,mount_points,loop_devices)
			src_image_roots = list(src_paths)

		if source_file_list:
			src_paths.extend(load_source_list(source_file_list))
		elif not src_path and not sys.stdin.isatty():
			print("Reading source file list from stdin...")
			src_paths.extend(load_source_list('-'))
		named, held_back = _named_sources(src_path, not target_file_list)
		src_paths.extend(named)
		# A source that is not there stops -rme, and a move as a file that failed to copy does: see copy_failures.
		missing_sources = verify_src_path(src_paths, exclude)
		src_paths[:] = _without_covered_sources(src_paths)
		# Names the -dfl file, so only then (a list would make a name of every entry).
		if not src_str and diff_file_list == 'auto':
			src_str = 'source_list' if _has_listed(src_paths) else "-".join([os.path.basename(src) for src in src_paths])

		if target_file_list:
			store_file_list(target_file_list,src_paths,single_thread=single_thread, max_workers=max_workers,verbose=verbose, 
							files_per_job=files_per_job,compare_file_list=compare_file_list, remove_extra=remove_extra,
							parallel_file_listing=parallel_file_listing,exclude=exclude,diff_file_list=diff_file_list,tar_diff_file_list=tar_diff_file_list,src_str = src_str,
							append_hash=append_hash_to_file_list,full_hash=full_hash)
			raise RuntimeError("Exiting after storing file list.")
		
		# if dest_image:
		# 	dest , target_mount_point = get_dest_from_image(dest_image,mount_points,loop_devices)
		# 	dest_str = dest_image
		# else:
		# 	dest = get_dest_from_path(dest_path,src_paths,src_path,can_be_none=remove)
		# 	dest_str = dest

		dests, dest_folder_name, target_mount_point = get_dests(dest_paths,dest_image,mount_points,loop_devices,src_paths,src_path,can_be_none = remove)
		if held_back is not None and all(os.path.abspath(dest) != os.path.abspath(held_back) for dest in dests):
			# At the prompt the operator gave another destination: the last word was a source all along.
			found, missing = _present_sources([os.path.expanduser(held_back)], exclude)
			src_paths.extend(found)
			missing_sources.extend(missing)

		if compare_file_list or diff_file_list:
			if diff_file_list == 'auto':
				if not src_str:
					src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
				diff_file_list = f'DIFF_{src_str}_TO_{dest_folder_name}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
			process_compare_file_list(src_paths, dests, max_workers=max_workers,parallel_file_listing=parallel_file_listing,
							exclude=exclude,dest_image=dest_image,diff_file_list=diff_file_list,tar_diff_file_list=tar_diff_file_list,append_hash=append_hash_to_file_list,full_hash=full_hash,src_image_roots=src_image_roots)
			raise RuntimeError("Exiting after comparing file list.")

		# Nothing below names a path through the working directory: sources and destinations are resolved, and one that needed
		# it has already failed. Python 3.14 starts its workers through os.getcwd(), so a removed one would stop the first pool.
		try:
			os.getcwd()
		except FileNotFoundError:
			os.chdir(os.sep)
		# A move: the top directory of each -d path the run will make (the loop below makes one ending in a separator; the copy
		# makes the rest), looked for before anything is made. Its parent holds the new name, which only that parent's fsync
		# commits on vfat: _sync_moved_directories fsyncs it and every directory from the top down to the destination.
		made_tops = [top for top in (_first_missing(dest) for dest in dests if dest and dest != ...) if top] if remove and not dest_image else []

		dests_to_remove = []
		for dest in dests:
			try:
				if dest and dest != ... :
					if dest.endswith(os.path.sep) and (not (os.path.exists(dest) or os.path.ismount(dest))):
						if no_create_dir:
							eprint(f"Destination path does not exist: {dest} does not exist, skipping as --no_create_dir is specified, ignoring {dest}.")
							dests_to_remove.append(dest)
						else:
							_makedirs(dest, exist_ok=True)
					elif target_file_system:
						destfs = get_fs_type_from_path(dest)
						if destfs and destfs.lower() != target_file_system.lower():
							eprint(f"Destination fs type mismatch: Expected {target_file_system}, got {destfs}, ignoring {dest}.")
							dests_to_remove.append(dest)
			except FileExistsError :
				eprint(f"Destination path {dest} may be a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...")
			except OSError as e:
				# A destination that cannot be made (no permission, a read-only or full filesystem, a file in its path, a name the
				# filesystem refuses such as 'x.' on exFAT mounted iocharset=cp1251): that destination is out, and the rest go on.
				eprint(f"Directory sync error: cannot make {dest} ({e.strerror}), ignoring it")
				dests_to_remove.append(dest)
		if dests and dests_to_remove == dests:
			raise RuntimeError("No valid destination: Exiting due to no valid destination paths.")
		for dest in dests_to_remove:
			dests.remove(dest)
		if not dests:
			if dest_image and target_mount_point:
				# A -di image that does not exist yet: create it, with -rm too (a move into a new image).
				dests = create_image(dest_image,target_mount_point,loop_devices,src_paths,mount_points,max_workers=max_workers,parallel_file_listing=parallel_file_listing,exclude=exclude,dest_image_size = dest_image_size)
			elif dest_paths:
				# Destinations were asked for and none is usable (a -d glob that matches nothing): with -rm, going on
				# would delete the sources without copying them.
				raise RuntimeError(f"No valid destination: none of {dest_paths} gives a destination; nothing copied"
					f"{', sources not removed' if remove else ''}")
			elif remove:
				dests = ...
			else:
				dests = create_image(dest_image,target_mount_point,loop_devices,src_paths,mount_points,max_workers=max_workers,parallel_file_listing=parallel_file_listing,exclude=exclude,dest_image_size = dest_image_size)

		copy_failures = []
		if dests != ...:
			copy_exclude = exclude
			places = _RealPlaces()
			# Decided before the entries inside a destination go: what is left of the sources is still a list's.
			listed = _has_listed(src_paths)
			holding = left_out = None
			# -rme never removes a source: the ones left in place inside a destination too (_source_guard).
			named_sources = list(src_paths)
			if listed or not dest_image:
				src_paths[:], skipped, holding, left_out = _inside_destinations(src_paths, dests, places)
				if skipped:
					print(f"Skipped {skipped} list entries inside the destination {', '.join(dests)}: not copied or moved")
				if left_out:
					print(f"Left out {', '.join(left_out)}: a source that is the destination, or a link to it, is not copied or moved")
				if holding:
					copy_exclude = _exclude_roots(exclude, holding)
					print(f"A source holds the destination ({', '.join(sorted(holding))}): the destination is left out of its copy")
				if not src_paths:
					print('Nothing to copy: every source lies inside the destination')
					return get_rc_from_error()
			# The remove gate below judges the errors recorded from here on. A listing cached earlier (create_image's size
			# scan, or a library caller's) would not record its errors again: list afresh.
			_get_file_list_cache.clear()
			# What each source is before the copy reads it: a move removes nothing below a source replaced since.
			source_ids = _source_identities(src_paths)
			# -rme cleans the destinations the copy writes: pinned now, checked before -rme (_checked_destination_roots).
			pinned_dest_roots = _destination_roots(dests) if remove_extra else None
			copy_error_start = len(ERRORS)
			# What the operator named is followed and written through, as cp follows its destination; below it nothing is.
			global NAMED_DESTS, RUN_ID, NAMED_IDS, SYNC_COPIES, _BAR
			NAMED_DESTS, RUN_ID = frozenset(os.path.abspath(dest) for dest in dests), uuid.uuid4().hex
			# Each named destination that is a directory now is pinned, as named: a process that opens it later and finds it
			# leading elsewhere writes nothing there (_dest_directory).
			NAMED_IDS = _named_ids(NAMED_DESTS)
			SYNC_COPIES = bool(remove)
			# -v turns the progress bars off (suppress_all_output)
			_BAR = not verbose
			# Refuses a clash before anything is copied or removed.
			dropped, replaced, in_place = _settle_destinations(src_paths, dests, dest_image, directory_only, copy_exclude, 1 if single_thread else max_workers,
				parallel_file_listing, remove=remove, places=places)
			# Between the steps of a run, a stop not acted on yet is (_check_stop): one recorded inside a pool, or one whose
			# KeyboardInterrupt Python dropped. Nothing after it runs: no copy, no -rme, no removal.
			_check_stop()
			if replaced:
				print(f"Merged directories: {replaced} file(s) replaced by a later source")
			total_file_list, total_sym_links, dir_records = process_copy(src_paths, dests, single_thread=single_thread, max_workers=max_workers,
																			verbose=verbose, directory_only=directory_only,no_directory_sync=no_directory_sync,
																			full_hash=full_hash, files_per_job=files_per_job, parallel_file_listing=parallel_file_listing,
																			exclude=copy_exclude,dest_image=dest_image,batch = batch,exit_not_enough_space = exit_not_enough_space,dropped = dropped,in_place = in_place)
			_check_stop()
			if verbose:
				# sort file list and sym links
				for file in natural_sort(total_file_list):
					print(f"Copied File: {file}")
				for link in total_sym_links:
					print(f"Link: {link} -> {total_sym_links[link]}")
			# A move's link directories: a link has no data to fsync, and is made after every file, so no file's fsync commits it.
			link_dirs = set() if SYNC_COPIES else None
			link_targets = create_sym_links(total_sym_links,exclude=exclude,link_tracking=link_tracking and not no_link_tracking,link_dirs=link_dirs)
			_check_stop()
			if not remove:
				# The move's backstop, at the end of a copy: two sources that landed in one file, where the destination took their
				# names for one and no check on names could say so (an 8.3 alias, a legacy name read as UTF-8). The landing is left
				# as it is: it holds one source whole, or after two cps wrote it at once (a parallel copy) a mix of both; the other content
				# has no name of its own there. A move asks before removing.
				# A file whose copy failed has said so: at none of its landings it says nothing more, and a clash is not said
				# to follow a copy of it.
				failed = {error.source for error in ERRORS[copy_error_start:] if isinstance(error, _CopyFailure)}
				for path, why in sorted(_not_held(_Joined(total_file_list, total_sym_links), src_paths, dests, bool(dest_image),
						1 if single_thread else max_workers, move=False).items()):
					if path not in failed:
						eprint(f'Destination clash: copied {path}, but {why}')
					elif not why.startswith('it is at none of its landings'):
						eprint(f'Destination clash: {path} failed to copy, and {why}')
				_check_stop()
			# A copy that reported errors is incomplete: removing the sources (a move) would lose what was not copied,
			# and an unlisted source subtree would make its destination counterpart look extra.
			copy_failures = [error for error in ERRORS[copy_error_start:] if _is_failure(error)]
			# A list entry that is gone has nothing to lose, and a list can be older than the tree: the rest of a move goes on.
			copy_failures += [src for src in missing_sources if not isinstance(src, _ListedSource)]
			extra_failures = len(copy_failures) + sum(isinstance(src, _ListedSource) for src in missing_sources)
			if remove_extra and extra_failures:
				eprint(f'Remove skipped: not removing extra files from {dests}: the copy reported {extra_failures} error(s); '
					'fix them (or exclude those paths with -e) and re-run')
			elif remove_extra and (holding or left_out):
				# What it left out of its copy lies in the destination, and looks extra there.
				print('-rme checked nothing: a source is or holds the destination, so all that is in it belongs to that source. '
					'To mirror a directory, name a destination outside it')
			elif remove_extra and not _directory_sources(_src_to_dest_map(src_paths, dests, bool(dest_image))):
				# Only a directory can say what is extra below it: a named file, a link or a list of files says nothing of the rest.
				if listed:
					print('-rme checked nothing: the list names no directories, and only below those can a list judge extras. '
						'To mirror a directory, list the directory itself: printf "src\\n" | hpcp -d out -rme')
				else:
					print('-rme checked nothing: every source is a named file or a link, which says nothing of what else belongs in the destination. '
						'To mirror a directory, name it: hpcp src/ -d out -rme')
			elif remove_extra:
				print('-'*80)
				dest_roots = _checked_destination_roots(dests, pinned_dest_roots)
				# The directories the copy covers, from its listings, before remove_extra_files clears them.
				covered = _covered_directories(src_paths, dests, bool(dest_image), exclude, max_workers, parallel_file_listing)
				remove_extra_files(total_file_list, dests,max_workers,verbose,files_per_job,single_thread,exclude=exclude,parallel_file_listing=parallel_file_listing,src_paths=src_paths,sym_links=total_sym_links,
					dest_image=bool(dest_image),keep=link_targets,dest_roots=dest_roots,listed=listed,named=named_sources)
				print('-'*80)
				print("Removing extra directories...")
				remove_extra_dirs(src_paths, dests,exclude=exclude,dest_image=bool(dest_image),keep=link_targets,dest_roots=dest_roots,named=named_sources,covered=covered)
			# Each synced directory's times, exact mode and flags, once what goes in it is in: its files, its links and -rme's
			# removals (ran or skipped); in a move, each fsynced through the same descriptor, and then every other directory the
			# move wrote into. Before a move's removal, which a directory either could not set or write to disk stops, as a
			# failed copy does (9.86 counted a directory it could not set the same way).
			_check_stop()
			finish_start = len(ERRORS)
			finish_directories(dir_records, 1 if single_thread else max_workers, sync=SYNC_COPIES)
			if SYNC_COPIES:
				_check_stop()
				synced = _sync_moved_directories(total_file_list, link_dirs, src_paths, dests, bool(dest_image), dir_records,
					1 if single_thread else max_workers, made_tops)
				_check_stop()
				_flush_drives(total_file_list, set(synced).union(os.path.abspath(record[1]) for record in dir_records), src_paths,
					dests, bool(dest_image))
			copy_failures += [error for error in ERRORS[finish_start:] if _is_failure(error)]
			_check_stop()
			print('-'*80)
			if len(src_paths) > 1:
				print("Overall Summary:")
				print(f"Number of files / links: {len(total_file_list)}")
				print(f"Number of links: {len(total_sym_links)}")
				print(f"Total time taken: {time.monotonic()-programStartTime:0.4f} seconds")
		removal_ended = False
		try:
			if remove and copy_failures:
				eprint(f'Remove skipped: not removing {_sources_text(src_paths)}: the copy reported {len(copy_failures)} error(s); '
					'fix them (or exclude those paths with -e) and re-run')
			elif remove and dests != ... and replaced:
				# Removing the earlier directory would destroy the only copy of each file the later one replaced.
				# -rme ran before this, and never removes a source (_source_guard).
				eprint(f"Remove skipped: not removing {_sources_text(src_paths)}: {replaced} file(s) in merged directories "
					'were replaced by a later source, and the earlier source holds their only copy; every source stays')
			elif remove and dests != ...:
				# A move: only what the copy listed goes (see remove_moved_sources). Each copy was written to disk by its worker
				# (_sync_landing), and one that could not be is a copy failure, which stopped this branch above.
				_check_stop()
				remove_moved_sources(src_paths, total_file_list, total_sym_links, single_thread=single_thread, max_workers=max_workers,
							verbose=verbose, files_per_job=files_per_job, remove_force=remove_force, exclude=exclude, root_ids=source_ids,
							dests=dests, dest_image=bool(dest_image))
			elif remove:
				_check_stop()
				process_remove(src_paths,single_thread=single_thread, max_workers=max_workers,verbose=verbose,
							files_per_job=files_per_job, remove_force=remove_force,exclude=exclude,batch=batch,parallel_file_listing=parallel_file_listing)
			removal_ended = True
		finally:
			if SYNC_COPIES:
				# A non-root move's directories whose exact mode takes their owner's search (0607): set after the removal step
				# that looked below them. After a stop or an error in it, only those whose source directory it removed: no rerun
				# reaches them; the others keep owner read and search for the rerun, whose folder sync looks below them (a rerun
				# over a 0607 one can fail: docs/limitations.md).
				_finish_late(dir_records if removal_ended else [record for record in dir_records if not os.path.lexists(record[0])])
	except KeyboardInterrupt:
		# The stop is acted on: later checks (_check_stop) change nothing.
		global _STOP_ACTED
		_STOP_ACTED = True
		eprint("Keyboard Interrupt: Signal received, cleaning up and exiting.")
	except RuntimeError as e:
		if str(e):
			eprint(e)
		else:
			eprint("Runtime Error: Exception raised during processing, exiting.")
	except Exception as e:
		# also print the traceback line number
		import traceback
		track = traceback.format_exc()
		eprint(f"General Exception: {e} \n{track}")
	finally:
		NAMED_DESTS, RUN_ID, NAMED_IDS, SYNC_COPIES, _BAR = frozenset(), None, {}, False, False
		# what the door held for the run (_dest_directory): its workers' went with them
		_close_cursors()
		_SPACE_WARNED = outer_space
		clean_up(mount_points,loop_devices,delayed_commands)
		get_partition_details.cache_clear()
		get_partition_infos.cache_clear()
		hash_file.cache_clear()
		_get_file_list_cache.clear()
	_repeat_space_warnings()
	return get_rc_from_error()
		
def hpcp_gui():
	import tkinter as tk
	from tkinter import filedialog, messagebox
	global root
	root = tk.Tk()
	root.title("High Performance CoPy")

	src_entry = tk.Entry(root)
	src_entry.grid(row=0, column=1)
	tk.Label(root, text="Source path").grid(row=0)

	src_browse_button = tk.Button(root, text="Browse", command=lambda: src_entry.insert(0, filedialog.askdirectory()))
	src_browse_button.grid(row=0, column=2)

	dest_entry = tk.Entry(root)
	dest_entry.grid(row=1, column=1)
	tk.Label(root, text="Destination path").grid(row=1)

	dest_browse_button = tk.Button(root, text="Browse", command=lambda: dest_entry.insert(0, filedialog.askdirectory()))
	dest_browse_button.grid(row=1, column=2)

	var_max_workers = tk.IntVar(value=multiprocessing.cpu_count())
	tk.Entry(root, textvariable=var_max_workers).grid(row=2, column=1)
	tk.Label(root, text="Max Workers").grid(row=2)

	var_single_thread = tk.BooleanVar()
	tk.Checkbutton(root, text="Single Thread", variable=var_single_thread).grid(row=3, column=0)

	var_parallel_file_listing = tk.BooleanVar()
	tk.Checkbutton(root, text="Parallel File Listing", variable=var_parallel_file_listing).grid(row=3, column=1)

	var_directory_only = tk.BooleanVar()
	tk.Checkbutton(root, text="Directory Only", variable=var_directory_only).grid(row=4, column=0)

	var_full_hash = tk.BooleanVar()
	tk.Checkbutton(root, text="Full Hash", variable=var_full_hash).grid(row=4, column=1)

	var_compare_file_list = tk.BooleanVar()
	tk.Checkbutton(root, text="Compare File List", variable=var_compare_file_list).grid(row=5, column=0)

	var_verbose = tk.BooleanVar()
	tk.Checkbutton(root, text="Verbose", variable=var_verbose).grid(row=5, column=1)

	var_files_per_job = tk.IntVar(value=1)
	tk.Entry(root, textvariable=var_files_per_job).grid(row=6, column=1)
	tk.Label(root, text="Files per Job").grid(row=6)

	var_file_list = tk.StringVar()
	# file list
	tk.Label(root, text="Save / Load File List .txt",).grid(row=7, column=0, sticky='w')
	tk.Entry(root, textvariable=var_file_list).grid(row=7, column=1)
	tk.Button(root, text="Browse", command=lambda: var_file_list.set(filedialog.askopenfilename(filetypes=(("Text Files", "*.txt"),("All Files", "*.*"))))).grid(row=7, column=2)



	var_remove = tk.BooleanVar()
	tk.Checkbutton(root, text="Delete Mode", variable=var_remove).grid(row=8, column=0)

	var_remove_extra = tk.BooleanVar()
	tk.Checkbutton(root, text="Delete Extra In Dest", variable=var_remove_extra).grid(row=8, column=1)





	def run_hpcp():
		try:
			root.withdraw()
			print(f"Source path: {src_entry.get()}")
			print(f"Destination path: {dest_entry.get()}")
			# we print out any arguments that are not default
			if var_single_thread.get():
				print(f"Single Thread: {var_single_thread.get()}")
			if var_max_workers.get() != multiprocessing.cpu_count():
				print(f"Max Workers: {var_max_workers.get()}")
			if var_verbose.get():
				print(f"Verbose: {var_verbose.get()}")
			if var_directory_only.get():
				print(f"Directory Only: {var_directory_only.get()}")
			if var_full_hash.get():
				print(f"Full Hash: {var_full_hash.get()}")
			if var_files_per_job.get() != 1:
				print(f"Files per Job: {var_files_per_job.get()}")
			if var_file_list.get():
				print(f"File List .txt: {var_file_list.get()}")
			if var_compare_file_list.get():
				print(f"Compare File List: {var_compare_file_list.get()}")
			if var_remove.get():
				print(f"Remove: {var_remove.get()}")
			if var_remove_extra.get():
				print(f"Remove Extra: {var_remove_extra.get()}")
			if var_parallel_file_listing.get(): 
				print(f"Parallel File Listing: {var_parallel_file_listing.get()}")

			# dest_paths is a list: a bare str would be taken one character per destination.
			dest = dest_entry.get()
			hpcp([src_entry.get()], [dest] if dest else [], single_thread=var_single_thread.get(), max_workers=var_max_workers.get(),
				verbose=var_verbose.get(), directory_only=var_directory_only.get(), full_hash=var_full_hash.get(), files_per_job=var_files_per_job.get(),
				target_file_list=var_file_list.get(), compare_file_list=var_compare_file_list.get(), remove=var_remove.get(), remove_extra=var_remove_extra.get(),
				parallel_file_listing=var_parallel_file_listing.get())
			# messagebox.showinfo("Success", "File copy completed successfully")
			root.deiconify()
		except Exception as e:
			messagebox.showerror("Error", str(e))

	tk.Button(root, text="Start Copy", command=run_hpcp).grid(row=13)
	root.mainloop()

#%% ---- CLI ----
# Python 3.6's stdout, replaced by _escape_unencodable_stdout: kept, since collected it would close the buffer.
_STRICT_STDOUT = None

def _escape_unencodable_stdout():
	"""Paths with bytes that are not UTF-8 arrive surrogate-escaped; under a UTF-8 locale stdout is strict, and one
	print of such a name would end the run partway. Show them escaped instead (stderr already does). The same for
	anything else the encoding cannot show: Python 3.6's stdout in the C locale is ASCII with surrogateescape, and the
	progress bar's first block character ended the copy."""
	global _STRICT_STDOUT
	try:
		sys.stdout.reconfigure(errors='backslashreplace')
		return
	except ValueError:
		return
	except AttributeError:
		pass
	# Python 3.6 has no reconfigure: the same buffer through a new wrapper, which holds no text of its own.
	# Only a genuine TextIOWrapper (a tee or logging proxy in its place is left alone: it would be bypassed).
	import io
	stdout = sys.stdout
	if type(stdout) is not io.TextIOWrapper or stdout.errors == 'backslashreplace':
		return
	try:
		stdout.flush()
		sys.stdout = io.TextIOWrapper(stdout.buffer, encoding=stdout.encoding, errors='backslashreplace',
			line_buffering=stdout.line_buffering, write_through=True)
	except (AttributeError, ValueError, OSError):
		return
	_STRICT_STDOUT = stdout

def _end_now(rc):
	"""End a run that was stopped, or that left a pool behind (_ABANDONED), without the interpreter's exit: that joins Python's
	pool threads, and one may be stuck for good (on a lock a killed worker held, on a result it half sent, on an lstat of a
	dead NFS mount). hpcp registers no atexit handler of its own. Of multiprocessing's exit work, what leaves something
	behind is done here: its named semaphores (forkserver and spawn; else the resource tracker removes them and warns about
	each) and its temporary directory (a fork server's socket). SIGTERM and SIGHUP end by their signal, Ctrl-C with 130, a
	pool left without a stop with rc."""
	with contextlib.suppress(Exception):
		sys.stdout.flush()
		sys.stderr.flush()
	with contextlib.suppress(Exception):
		# multiCMD's exit hook, which os._exit skips: a command of this process still running is stopped, as at exit (its
		# approved limitations leave one 'running until kill_all() or the program's exit stops it').
		multiCMD.kill_all()
	with contextlib.suppress(Exception):
		# The finalizers of exit priority 0: the semaphores' unlink. Not a queue's close (10): a pool's manager thread may be
		# closing that queue itself (from 3.9 it is told to end at the pool's exit), and the two raced to close one pipe (EBADF,
		# a traceback from the queue's feeder thread: 6 of 389 stops with a stuck worker on 3.14, 0 of 240 without the close).
		# os._exit closes every descriptor anyway.
		registry = multiprocessing.util._finalizer_registry
		for key in sorted((key for key in list(registry) if key[0] == 0), reverse=True):
			finalizer = registry.get(key)
			if finalizer is not None:
				finalizer()
	with contextlib.suppress(Exception):
		tempdir = multiprocessing.current_process()._config.get('tempdir')
		if tempdir:
			shutil.rmtree(tempdir, ignore_errors=True)
	if _STOPPED is not None and _STOPPED != getattr(signal, 'SIGINT', None):
		# Cleaned up: end by the signal, as 9.84 died by it, so timeout, a shell and systemd see what they sent.
		_end_by(_STOPPED)
	os._exit(ERROR_TO_RETURNCODE_TABLE['Keyboard Interrupt'] if _STOPPED is not None else rc)

def main():
	global RANDOM_DESTINATION_SELECTION, _MAIN_PID
	# A stop runs Ctrl-C's cleanup here too: a serial run copies in this process, and hpcp() cleans up mounts in a finally.
	# Installed first: a stop while the arguments are read ends the run quietly, as the signal asked (below).
	_MAIN_PID = os.getpid()
	rtnCode = 0
	try:
		# In the try: a stop landing while the later stops go in raises from the first one's handler (_stop), inside
		# signal.signal, and ends the run quietly as any stop before hpcp() does.
		_install_stops()
		_escape_unencodable_stdout()
		args = get_args()
		# after get_args: -h and -V print only what they are asked for
		_warn_bad_overrides()
		# we run gui if the current platform is windows and src_path is not specified
		if os.name == 'nt' and len(args.src_path) == 0:
			hpcp_gui()
			return
		_register_worker_start()
		rtnCode = hpcp(args.src_path, dest_paths = args.dest_path, single_thread = args.single_thread, max_workers = args.max_workers, verbose = args.verbose,
			 directory_only =  args.directory_only, no_directory_sync = args.no_directory_sync,full_hash = args.full_hash, files_per_job = args.files_per_job,
			 target_file_list = args.target_file_list, compare_file_list = args.compare_file_list , diff_file_list = args.diff_file_list, tar_diff_file_list = args.tar_diff_file_list,remove = args.remove, remove_force =args.remove_force,
			 remove_extra = args.remove_extra, parallel_file_listing = args.parallel_file_listing,exclude = args.exclude,exclude_file = args.exclude_file,
			 dest_image = args.dest_image,dest_image_size=args.dest_image_size,no_link_tracking = args.no_link_tracking,link_tracking = args.link_tracking,src_image = args.src_image,dd=args.disk_dump,
			 dd_resize=args.dd_resize,batch=args.batch,append_hash_to_file_list=not args.no_hash_file_list, hash_size=args.hash_size,source_file_list=args.source_file_list,
			 random_destination_selection = args.random_dest_selection,bytes_rate_limit = args.rate_limit,files_rate_limit = args.file_rate_limit,
			 target_file_system = args.target_file_system, no_create_dir = args.no_create_dir, content_only = args.content_only, command_timeout_limit = args.command_timeout_limit,
			 exit_not_enough_space = args.exit_not_enough_space,do_not_remove_files_while_listing = args.do_not_remove_files_while_listing,
			 no_fs_param_mirror = args.no_fs_param_mirror, get_diff_image = args.get_diff_image,
			 load_diff_image = args.load_diff_image, force_apply = args.force_apply, write_beside = args.write_beside,
			 read_before_write = args.read_before_write)
	except KeyboardInterrupt:
		# A stop before hpcp() took it (the arguments, hpcp()'s first lines): nothing was started, nothing to clean up.
		rtnCode = ERROR_TO_RETURNCODE_TABLE['Keyboard Interrupt']
	if _STOPPED is not None:
		# A worker forked after the stop was passed on (a stop while the pool was starting) ends too.
		_pass_stop_on(_STOPPED)
	if _STOPPED is not None or _ABANDONED:
		_end_now(rtnCode)
	if rtnCode:
		exit(rtnCode)
		

if __name__ == '__main__':
	main()
