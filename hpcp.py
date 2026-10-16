#!/usr/bin/env python3
# /// script
# requires-python = ">=3.6"
# dependencies = [
#     "xxhash",
#     "multiCMD>1.19",
# ]
# ///
import argparse
import concurrent.futures
import contextlib
import errno
import fnmatch
import functools
import glob
import hashlib
import multiprocessing
import os
import pathlib
import platform
import random
import re
import shutil
import stat
import struct
import sys
import tempfile
import threading
import time
import uuid
from collections import deque, namedtuple
from concurrent.futures import ProcessPoolExecutor
from math import log
from pprint import pformat

try:
	import multiCMD  # type: ignore
	if tuple(int(x) for x in multiCMD.version.split('.')) < (1, 54):
		raise ImportError('multiCMD too old')
except (ImportError, AttributeError, ValueError):
	import sys,types,base64,lzma   # noqa: E401
	multiCMD = types.ModuleType("multiCMD")
	sys.modules["multiCMD"] = multiCMD
	_SRC_B85=r'''
rAT%)=o\O*k16n/!WXAE'EA+5s6&_Ti3^3S2I-:pLdFUh,;[5]%l=/aQgL]knEMJj2Td\)1[`bH(.0)JfPPs3eIh)]U<,WTZEDeSe:(YNb%HJ57qq1/\
*gL=8;jBHJ\fYg4:j%to1R4$=%aO[S5W>p`gA_=$'[uT/?.1N_&PE1HaE>4:M&BCBVcr3$)%gS.F#Y0U;'G@Yf#\iaYMX8n@Cj7`rmp%aCrJu%B[E*^L
S#9/4@nY7\_*&('7?"ObNK^I$nStm_t)5D!<$XBk]A8FSldVFqkipZ;EWnA;)9(JK#j9Pg8;XEEFcR&MXS$i8K]2@Z)d(Fc2uC*D(R5&XBjOabb#nGQM
s'jk<mB2kgSK<iH7Up,F9L!`X'-4q[s(,,!OKs+R;A4P#:D(9F&6"H-::L31I>l>F7FcpCR'TuK\)3iV_%ms=;+MJU^li"a#5qt/XYPkN5YYCn?/L"Y9
`YKd2^$XX`_0C@\.Y8GnU'3R!cE6c><Gj@T/l.dA@#Y@'89c>!>R>8f@b[+0An>r^6A/gX)W)CDc1L_86SU+RdgW->rUPk!WadE/L9>fri=%uV;mun'5
2_H#a9V4FZFU@>WEd%CE3T`13F,*D5jnV7@&%/X_r0mN$m(uMs3]J)]nP>G)=>E$<Ka)@W:H=qDF!e(2DrId6E+Z"C:Y+AtXI::L\,=9=c.FfG`o.dSo
%uH7$0.2oW90b^E7BeXP1$YdbC<7@KF!8Hc.fK488]7m6(LL7^F94hZMZCP#G-g$#t$[![OrtHQ.dNq2M"K`io4/Aaq:)NY%oXKqdal!if3T"J\jYUMa
LjW./0YE:)=T0`+ld;^/R@33e)T89fst(Z>>?q\sFNgYoIo'<_JQBY'\f>H/hcSh+VW"%i$#@LOBfpHSrTj,-6r$,u]&;aj^I,O#&Fr=lD>U%4ZA3-7?
qi(sCNJI;W70*G&[-fJ9+\1WLo/`7\E(]EH.DE,H6&c&Tr=.K$ssRlF@&DY@bmMPH0Y\$SeWq'fLJGZ'PU2eN-Cr\e-k7O!^?2<R_a8MG,LUBC%a2C,q
SAMc+^H?IF$4Sa9OX)PksrN1<9etSPYK$Th^bjGkW8712[%<ZYdkA$<J;>\N>_k[_Yp6\(PVQP`uX(m+Qa5')Bs"9*sVJUue<mZRI4gd@bSTc.*l_DB:
]h'7/kZ",I.#!O.G)U3M5tT8.`JJD";p:m$m%nY8Iu$MC$2!"*5s!;nLQfud%'YF6[\%AYF<WT5BTWhm=(+nL!;kS<56m%$-bf]iiT:%>pacJ=:42g^I
R+Y?iX4m8HhP"G_nY\R!!DRtrG_6TnSfK5oDGg`ZTcb$b%ENX*M&/WqEj[%>M)d;2@YIgMt?*i-Mm0O:Sn_4&B!*f;qPpgi8XGMl1/_'N4:Xt=aVTUNG
`XH^:h&2_+Jn[T$(QZ@rEcHd^Y1^4]"P@mI,e?W\Lc1f\2P?r8F+VVKP$b-c\gj-P.;DV:K\VXJ.THKuQV#A/,P5:_2uYQr(g!?>BcXk=3_V)KEm)WFZ
6]UA<LG68SQB"@T&F8Ju&GE8C\P8GfqgJK&s=c?N=Jo!d/EGRWuPqt+D1K)qNo96LCM'Hpjj2q-O@p*23uq3Yb[X8bg%b:%5'%3)%l=a-bnWI3`1HLd^
X44'I,mF6(A]'0%1Z)gSI_[G31^mBAJep8*SGBp5@L-?A=K>T+:rAjmV<L;IH7B2Gj@"M-P-012P3@)(,cYp7Sd[c:>U<E8FT@NILI:=]1%L8X;6(^TX
[;7o$RigVuVS_RuedEj>b-nAJ)PNLgi/SJd]"p9:&jj@r6I/6Ym\3DPWgarahNTc&=8F`aY7+t$M#L^0TRq>h]-D_SWGTHf>T@V)j@@VLUZQeI1urGt(
6Srf[E@/sX8Q,L^S_kbFm;c']h-,gXt(I)(Y'?=:37U^n]Zo%ggJt9%t\;t=C&jX-c@bf@!fUXYqJL2:<ODD;,u6l'0*]THKO%I#FjPk^Eq<e;5.B![t
EPS/frTV`*sO"Q9Toi!C=lo=J%!*b'D6'cP)NHZ95o'3G``H9,pK22:16I6FFe>Kr!AbD00R)ID1;-*8d@e-Z$sQRn"U2fcM($\H/MG!;2<A;-*_^ap!
V2AHQVu,`Qp%/K.(t+>mkgEfl>O2T#/lYDeXpi`!+k(*(!a/`M6l$Za&*8!VWS3Yod4'<G7-?&6E@ZGK;oWDJ$3Cq5>i`k6T#F,1]<^n/418o(<1"76;
eSY`kiRVEN_=WAE.eB@;*L.I)H.t6=7UgI!";T+cDPJ2:LVpAmW$)>`\,4/1'"m:"NL*jD8S0hPPVMAp7$m*7(8oAab=4LNkXZ%k%+r3\;[3NYB*Hh+'
ZS1EDot3U`.]4S.;4h#:f2R0-PlOet!e[Lj^gmi"k\!^[:;QPkX/`9(#/\nZhWf,B;Pl"eI)Y5j+1u1haGpaE2]@S0$Z??!2&;C^.u,'N6at@K5Dk)ab
R[K2qM8Y,r5PnRhN/bH!1AC'*lBmo,D>2+pnl(5Z,$TUo%?05,;Np:AXsSb?3tgd6F1Jc/a"GKN;^lp?b?qG+<'<TE/8))BSMD$*^Y(D-_7?$6m0R$aF
&oP4_Vb=#p#Y:;C"#$[$9iD:5Z)DF^mBDG$drZnXgKd'.1eUeW-j-F*J@<UC;&G$o;(kN\?t@n!F+DC!tc^cmoYn`aI>RM+*%![dX-oCraelJ:#u>KI`
R1*K&j#a30,J\g$i/+"mlg>i\Et!Lo-jp^-sFb<?k3g';e-<Q'X<b\J>3aU+Yo'[gs`WiPh#S0L9Wa3BOQl:'6Q(q6^EA4=ru:!*u1O+`:%,rZn-eT@)
>(=LMO[rll<X[mt#l?h<9k/'9`-o<RI1ZsAa`&+)H/s_00b??4kWYG8Q`_'1Z"TnTmBU[gYQg5G<P@T^1)OP<d4/fa8$"bFt\!u5?UHUD%T*MRe7f\;]
n"qjD#b/a(mXn!Af5A?e[2<R>i\j4qr%34mNT`87!eMsJ!kP_A0,/W^V#e@9q.[kO7=OlDEsut+#*PZeT;)Z?>C@("'e'GpN49hF`>k?JR(]cVhjDDs/
GF12JGj4h.*_)ni=>cre2:J=_QeN*iRHt>&2!p!\oQPsf?M`Tj!!cg)fb;&BW[mKVp4"OIT)*fQp_[L`a%qH.n,tK'Tqujp4hTS`]Q*GK#VnlWBBoK.b
pFeJn\["::I_K?&;:ORA$#Sj#:Cc2Dc;dIu[I^mQt.?F)NSLL#N_IV0d?S#F*L#jWE?a21/f7d25hsKWV%!WpTfT9kn:QU0-^[1Pnl8*%8nu[N'0C[hj
j2<WE!k[2ac*4V2#/3CWPqB7TTj/R9E&*nTiaMnL]NN):0Ep2hU3Vl5O-;*/M21"SB92=gJQEEY@6+L[,.'csN+nrjhC08[ItlWB5GkQ'(WXal4P6V<U
C;Akf&<Z[%5=pRn0ai-DcOtfkK1+R&j-qrRkn75KPL0D0l+VsYr`/&f!M[JOHY]/t$%,&:\ofnq&LilJ3_<!E+On50^\u!1=baF%01^Q)6,MG*GjKYV1
/G22K$s:^s#c/$]"LQVl+CiQ>CAZU#'[1]n$.H#sB4\VdFE'iJ01X?ghjOU8gRt65?_9BkbAfU(r@qZpna3Mo$$qBU<noX.'s-*b<=Y`4D]r*?JnV'^\
*c$aq2-dB*$@P^@haCA]+44$N6fs/FJgFcr_8J?7!+2C)r-S"r)X_t5c)sB^gosL$Fr_hdgTOK48NJ,4^>;Xc/3i\qdAuC5!#Rp[o7eVG%@Y9ZT*Jdm*
7Gfo,+93XV#EmTj.F/,Hu@['80_'C*kJ:!;6b)ntAM8[o<HYC`frVE1k3G,W+jSU6A,sYsF.:R6PH:C;$7PSdlK%Io4CinX&2qkZ"[HKRnNa81_FF6>$
B^N*RF5:-@7Q2*Zr@>B2:+M:[GXe)(O(]Nn(+."f-`Fd^%:$5UqK5;75Vm^&KuRH-dnU9^r&`82`%HfH*HETY%b,H6NGR4/#)<^219m6H/9j^fpp.QK#
U/@lk>i/M$D`fmDg)<dn1bS!TS=\`cD6^iG=Ud>+/[7aYS`K2U&;.GMN"7+?GmEeWbM/g8I)1ZZ,bG4;4r8Y((;gK(#^fm@VUb5;P]bYO_VG=/[b40@=
Siqf\B+>@XZLOh)kpj"RRR1`#%,]9Z>:^R':[mI`H]j]#2^u%Z4O3\!qs5'`H%:1#(mt)b/@.EPj$er<GgbLnQ:7pBN"D@J0Zb9+o_Li94-b?-U:hl)Z
DW%*<;)mlj--r>KU?Rf'>nro98``\QFV9Oar6OrL%X%F@I/"cXj's./a%&Jl*"=Ig<7TEAs+u3g-A>HTQ((b-nD6D9R--TlXqFt8$\#ERFTRA<ZAG$rN
6)]rSc>eG]XtheijutKop0Ap,@;8RKB$gZ!u:(pTi&1,F+"pU9Z-U.P/d,`Pr,A8UAj@-tV;'fIR?p0Bk\VFL0bNc[7,,c#ktE!^;`]HlIg(in59L-G8
0YT!Oef;?NY4^L-G&9*SVO(CFRGC-f.:@Dco1$U9i9?_$n!$RLme+:Yi%@*^]l<_%7&?+ODLe`1`5WR0*fKh4eb?OG9!^$YrIVi)ZXmGQ`)34^`'UpOD
%VXWV#]A1CjJ;Olg?QgY+ML=DBRU!;G;]6U=PP9HRgiud[%_"3NbCI?SP;+PFObE&K@>GHH`Vk`GVR-5;\,X8+''7Ih]@>2Z.i$]kjg:+T/oDoRV:!F8
[N8c$Of/,r*"kMF<6o=7b(B=]2meGVM4sh;<ZG"-.=$nn3,j6\b\aRCG;#>cUj6=9'a];N-?6=Rn_UWZ\;dQ%KHmb^g4rBB$Y_`o6`.'%BrtG?=RJ)OX
CV2Qb6"]GH<s>@aHWkjWSY_t6F'HCg2%nlQ9]!'g)^BFn@$6'R2,7:*p7aRLnn7A8EK:oF\8*J/XoEP2j%d+S'\4YdGhDYs#U9Y`T;!Wh]cWX[k0;O;m
<tEGf_2<:,$0kH?U8H#5sO@lk%IcnYN!08lpr`G_8"4[\<+[cUBAa_<lT8(b0>Z\D.KWgg"'$]uVL>R_RKd\WWn?E6+-p^I'[cF6%?Am2Ldt_8M6$2m1
<=8r&$L<DSVRBKhuJZcnM3fmH-Ud;M-1HB1#Og"gP\nt2FDTLej+AF%IK8p0fO.<C,?AS.8g8Ig]eJS78HM@anCMiN#3Z?;0gf&2an:Q_Sj)Rs75+FK[
kY)'\^cpi.B9(BQ+kS`Yl3*L2[3G(Mf41]L-%ceZ9OF(n;"R]r-bQ#[A6<iX\=o((YX\$Cj5_D9U,)PfA(.DdI"Ud:/4ijPs2csf(8@(a-,n+"/Oa)!2
`kO4SIt3/GU]1^0a@]q6/*:;V%%hn*\j@kh)\7R3rs,YW(X1>"e<)M7\P"@3UEDpRF@[IDJ,QH8q]#::?l"WDn=WZ!JYi8<rP_cH=bVqHV,)NlJNmG9]
!?=4ZIL//-hL[?"*RF.Y:6_f";hdg_hU^1nrBi>%#TNj+@)@a\X6']=.hY?gk6EEP\onA(et'E04cL$("JJj(\7nDKq2sTW?(HtY7%0@k>M"38XY/jBE
+KnTg(!VkY'TJ:3=C]ZI^Gm+=qg!"2\8t?UrHUr\=G"BX-AV2,`Sg!3#FP;<]OL=e71a%\#LDIglGL8k"YURpBg+3FjY570uYs4;)qjK`BiaAu`.Z9%q
/kY3f"GeQ5>,ZfhQ^<.7lFI'8('-@kLdQ(E\f6!D&FNG?7'\`/Ggqb<8KNB*YB-jO2:+>Z,ZlQf<]O9if9/\2c(-E:HiBUHK_V\WenLuBS5@qcP%\(.i
GS!X+MWAj(-o6YP$Y_GIgS&Qp&)b_Z]Q?cg!rk)indZbgTQc\TdFU\W9GA:(WYA_>s+KeBrEa@h/9)so]:iSXH7LTirQ\t$lO^/n=9dQ#5,O7B0aIrH8
PR5S8TFPn;J`NI`WB.6M6iS[adcM=R%uRm(HC*7;1)n/NWR4'f!RYDM5to;p0SDKG45d5r8hPh4$!+W[0/E[,q,U=?9?<tBYXJdY0lWl1AIkFD,2V%MH
fE8<"'^!b[M1,GKJF]Z`Mp_-oQ3,SOtlob"k\sH_05i2)',Lhq4'Ri;N\ek/.ZpJ%CG6.c'KC_Gu\J[5mJK[S.;2+K`qAjf#fX$bp4H$VR0a)\5D91?m
K0]-:m[BA.Mria&@WGH(e.)WAGJ$7+Ed@!QoX>HAQ`I,J6*i9t>6Qi8#I[oDsD11\<W>'YrZJ%Jm:moQZJM8=lPtmfNBHq("/ugT\b14fl029dWD)0*"
]n?BC`"N,-KQ;m#\,qJgNVBqt%\N_\5P&`,AOVrumG#esrN/JA,ZH@t;R)`Ga2*=+no'ojdlZ(bCB)9CP*bX^SCIMKe3mo[U(L5b:mGbg#6Ck\QI)S?+
fg'7[VhnX]W-1<?Xl%IU\Tnq/WmdoSf5ocfDDin=rr<_g^5POI'EOdZWBq/$HKm*4VkrTE0Q#'TU>KsTbGj1(\ig0"$[,kb;ke2P&0;V!!oXjXD(uPG$
53kB:9:&;grj#&4($o91HXCs1&Pf^4S-^N!2B*II5lY5ql#n?/dFq^gU$;8P4s!*KR,fhHg_Oj3Y:,m!-XRF/>()cPl/W`Vq&k\I,;!nm21K3N;&C!a#
of,)FeAm_dfeB3XIoV"ECqKB8jc.1)uph`6eT*)l7qGl(1O?dpEFf)pSkl/7k=DPR8W+*Aa`t3Qpq-:3V^0\WV,kB;2h?"X*Q!L]#W!sJOI*9HoUjTb.
87P!)`6917s99c:@77^OWVJE'-ALW2r]rAY]f5U8$mYFIp."K%u!!RU@#7E0"'7NbEURr4t&u<f:oH:`U[Z[hF:u,7Suoh^H?/iH_g/1:5UL,35eS_PL
kB+W@JG<L/j;8B.&-X*]Eb4h1d[e5$jV.SQe*i?Rd0j#*WU84]Bo(F&qMj^.PAB5"-%`CHL7WuFW^b4WC'V+fI)OJWIrZYYoF+EGhPB`h8IM]EPn-[g]
p3^1;Z*2gVMgUoHbIQ/@[%Ihh)Wi^uU-YOWG=t"";BSs9Jd<:oN\l@*,6?r>Q2R:TQbTI%LI7h1;4XqipD/f'4'ar.(<!9tJ%pc4G?KprCNH:X'Ks&PX
]WlA(Q`\N[L?J>[e5dX$hLCB]<WS/Z!AD?325jspm+k<:=c%brA.IMQ'"u)p=MLV)#(L?sc>^?dF5\tqm3F8u>)qm@fi^VD.W$dqC<1G_&spE(Ypp$62
BV$YnoVrX/m,M5jmE&).X#g)RcNoJWkOGbEXN5m4eINF8=>RC"88Z]GLP)Tl]iuQgH_$#^IA8pV'H8iJrSVtP'O)24W^^",Mq':9$AL2d[[VPZu9(l7A
o$TllE?IX)>Hhp^.SO<uJ00-ET8rT5?TC\+uVB/5^C0<L-4N)8kcC&q]7'?nV&Zc!eT9jPJp$+SJ'J\g\QWSr"E_,GU4Hm?PO2;,SAbMcbg`$V5[W=KR
#oA"h&:=d*!cT-'s(,o5EIe.lK07-VT`lG8PTRSLmKfd@;"9hA/giY[/9O;)]Wm+7rBrJWWaggufC>iItS/[J'MCZEp@^BsC/&0niO@SYJI*^Iq?SR>d
`!FCO[df<2GS+;e"3KS'kT:SEkB\s6e;abj!gSO:Hm+fB`-"mMC,n95q,T1pG;D:?aAr6s+q"H%0*P1T'bb++r:Q*Pp?Yf4"nL@(,UkD[m]h)!3BF=_3
Gp@YeU80?F_T#B`!R)0N0d_c7+UT;:qN?8sNVu%R-`%JeQ86DGJ"hu30rg?Xr.S\hnUSs'2,bAN?^:t4dV';DFs*S-3Q*(320c+,U1)2p&t5=6@>u&lq
tV!J<s.Z7"!>>q:i1'VFBiL.4K5m*FVA.q0If,E^.tLg+ujV\,HkmD"N8ei!n?!T"B]]FFhj6H,<U2k2"!G(FN+2he&W.01lC,XOTd"UEQ)t4NBO"D&P
E$[]qdiBmdiag9LQC'-3ID`Z9saqNLpieJ>1e_hHi6VKlZ"n<,O6C*b>f]qa1T>'OkBQBoe+'!>r"3Xq-4iS-Y(4n?=;pJD[qRTBeF!#<nY42ra<(GB7
A:;h5C:H?>cuES/</Wsfib(\8)g*?ZPh7PESdcNRpK/m.N@F[l7+Fiqd-mnm9lnFsSMn>`(/U?60+^>CrBkX%sL.gL8@eT3akC$t<^r3F2;no$0Wc*>G
+bkoOO5B..9>dZhZRqQjl9^lVLi4nl1U?\=XYuAl7)Z-JeR?7n%(d0nsdbC:be8]MMZRO:D2+Ps6R!F%N]Esam5"efB$B,4pbAg+_6:[Vp[dV2c=Rede
V/)G"r5K(00(uha![b;lP1f&;&:M*._`3=YT<3K%an:e'[)8HTrCH!FPLV]c3h']Nm!MPQ'\!B0X`n!#O:Y4u_LXTMnJaKFhY/Q#5@Tp05ZgLX7/_N2(
iGc35*?_";Di[O!Kdl"i'L!,%r4\ViD\L#cV5d!TZe,a`?-H%i1ni=FYspP>l=Ir7"cJS[32TG#$V;[d0](\#_^<g4K'qphF#iT#7'hIfWPT35!2,)\j
I(6TYXa"&rJ:Y7qujk$j*bq[Q@VY)1=J^M#u6P![>F"G,6>o"gp'^.-9;2:m#0*%s&.>.A`/sj&Jm.gGPiN&L?H*Qbu$u1A_l<k"*p_AVJ[!(q*TM#T[
C1#;Qo!STgZe#cW<e!?Q__[(h!>p,O1/;$uBns#gaQZB,>[a-8Z]hmG`IGdLeda$a>G2(b=m!`Pd0M/$0G_PuYt(N:V(kqMFI.Hh3cLY!`E)jO1d_IB0
UeA"#jJPD3u/nLMb?;@/?*kYO1-pt0'!1,,T(a_J7<kZkr)\IL,nK9#82TO/$f-r0;0HDBXOk@;gT^e(Ykdu<=;>K!_Jnt\;1"p!(^tdaOoA80gN>qMZ
Tj[D'[t%AdFP]"'l8]0P)Kk"X;gM7N)O(gZ@I>:'I!ou+Aa[+?7dg&lZ'5N0p*u$&8:dYdTqnQN;Xg*qm\Z=+kW7P(<(/S%S'B!VO20Bh]'n)qpd+9p(
UI3QpV:p((WWpZ.BAa*eb/D%g)uRqMkGP+Nj-)^35p%ABu95Yo$C;Rf-nQuiRl89j3EhXU9i(]XhWqR@J?;\b$N_s>7nG_&9(X_&X<nf[0=\2Bg-O()^
q?t?^BFfAlK'A$/1%I?Nhh-"\d<-QJ/dr#if+belhu+RKM5b^m&(tJ37$[:aX_IRt`l'`]MI-0hl-6jtP.RgfdnXq$<!I>*FrrX<uNL#E1(#1>[7Xh`=
ndMF>_e%A-bir"9S&JLqX$'u:OQB),Mql)CftN6$]aKP9d]_'J=T3<[#"+)CdoR%m<=8)grIYc.@]]kf!K!1.S/q#h&X(2.1"ld'(_Wq.V0^MR#fN.KF
p^ahf^a6pCCGBbP<'+ZrYa$<1Q(r_A.8+mCAjZJ%TVdbN,``@S/RdLL4k2<QXm*LAf;AJo314sCF1cV%NMe/nt>h>hB;=r(7F,5Q3b.n08@O-TRFtQM6
@T3iHT_P1nr'>ksA[^i5?l<`5_g<4gK'CBToH\8N`a]\S(#2)ciN.bcWh[FJdbN5@X]OTA;O_t.[_F&e",IMZ]-K;&U/9Onm1q<HHWQ#QR`]s-<aG/&:
c9Hl9;[L*O(2P/rnLtbrEMbuI^BZH+Ql5,llmIkO@(0$e>*g?#6T<?c1JlhH;rJF7PGmp<MJbX1jB"_XBp@WW/e=m\S&Z7<Aj(^Dk#(Vk8^p6l;GS_gg
sa:H6jY#=U!4b4GNl\=5'C&8(U+e%[Kj`"IHg;etLEM%P;".@bE+nWd:P`eBhQ-b0CET"-*Foe5o($mhn\;CJt98n>rte^40GCecKlI3r9m=[$F%3Ghe
)RkgseDgUA#"8-/_C(0cnE^0Mojjde[j7?)dTS#)N@l#;M_/^t_2R%<kj>u-:iQe3;lXtZ`)C]!1E;Xr.uG8'n[%1L+"r&;uc;LUr7^nnmn2Opt3W%MJ
ACC*>.//"sZ^'@PdnJ"'#2!7@??M$h#.=n.?6qp>=ZPLpd&ZP62a:YM>jrJt`$blT$k!b09*talW?Oqc`W#(SbJN;JoMGk%$1]3#%748!U@,a0'WrNV]
-rgQ/9K,#u6OR'sF+7K]ej9c]oO4B&OE"ne+W'EZgF^Fe97'PK8Y)%gNKft_p+9NLal;8NbKK(+->N%kpG1K66a?[CKM>^rmIPoGHX`./rHbaVHn`>JL
KD6TI#[d6XUf.JZFc<Tp8/kW^!cKuP$=2^$f=`F[GR4kBi.$"7MW"Se!&CYNQ>J>EBaSR7Ymf"+^GQ._sSqiGG2`dTHJJ]GEDuS225RaQpo<=&8:#bR3
OlEKSe?j%EcsXA_B`Xlka?.Q?7CqRimEWM3`QF*SK^L?abKk@:pZMM^W4e]Kr2B0;OImroS>\'D7,SC\nab`0+2D7mn%K#r[RWJqdod*n!;9%%JFY_o4
Z<Xai.P53eVc^b*%A$LNkdlJ$@9Vr'(iF3hHP[Kl'hq;Klse4CjM"YDJ\eZMBFXi%!C+or`JcrjMQ8`?/[:-hVsTqN+:=E%j'WY)>S#s/56oeq,*Z*Tg
p$6]?`7(HUJjtd#'^UOB#hINQ!SHfY$Y6aI+^acu5O7e2GR,D\CQ_#N'EH61klGhgZ:jd21kkY.ME^IaqV.m,VSVNI^(WnPh(\aJKed';!/,0Y`HNX'_
H4Pf_s,HtMZX5.,hO.mqE/`muT2;HWM7bWYho;$[ek"f$8/F1T4UOT:;%'l>OXe;_bt&U.]FeH92(#8F1DM8WMToAbJ->nrKZE(CeWDadgeS4\IOH%ri
Qm1'DhMkF,Y]N6oP@i&T(l[3rC"ipaC(VKlSUaX7bQ9D+IM;b@gCR2\1a^MLlW[D''mU>s$=sDQH?O+>FJhpIXha2P^Wk;$1YBs34S^Js2.r$RjWiSch
,'K1(#cnN&[K%[Ng6a:aBDtVb(b)4?FiU`0VOpV2\YE()s)EU_Z"j>PG<2]!d-'6V*-A_W7Mb'%0@;mBs5jE5QI!q*^l>7!bDI+H]Z2H&"/&R"N".C\]
`Ng)S#sJc4HjCDA]F;mkN2iY-N2QG8mpZ"2W#Vn]7R1.%)$RLTC@>co9pLD$]QDfu2=P77iK8R%#CYgWI3B)Gu+AXL=9h<cjE4Qs=%l`Z#Fbo]$U@IFE
NJWH/DH.En@)\^96Z0Z-U]=5gb(K,k9/4BK"gufKnN-`U6m%0Uj$G4itO:?@6GY1(AG[?0Y^Lm%pRu't!]'@LiOU^/W0m?@BqI+L0(`=C4@bMqtLPq#D
Z\e;BoL@4:F:o!@ds8Kms,Hd2Fo!*:XOsrhA@!<sd7sHh"d",XDg/S?Xfsog8PrUs\/3^7A"Q(^aJHb*hS0Dr%K=6`nrrGF4]l`+9&HqU8&*80,l'BO0
p\,MppC'aS1@EgnqM+Q!%FmN[5*]LNbO;lgYXnABbUOZ;C5^B-p>HZ6EkP!>j7&9>?&Ridna_?TQtjMM)fQaXU]ePY;Q=r/^cOd9mPW[$0(0o]?+WmKS
s^5b:i^npN^!Y=CRH*m0=k-)Ag8URX`kcDQbZi\>2Z)6Q;LJc;h<#H^*8bkuk/&_G(d/r#65+N-i%`":P1V9cD!8_qi8l^3M(VZ[9*RoK&=1WRe26*\a
9Y/jlT8["PKp6al/oJ'A`]:YSgq0\'4(bD77In1Y#-%`h_:JNZ<:=cS_Imk?V?087de-_.*cDk[%1g=)1F?p@aCg>nQ/RT6o'C9j;"WS-mj[$\'u<'H%
1gJ%E6dJ;t\q)=8ho&?YQc$:&)grJ0/Gu*$Us!lQikR7+ih`U>B!l7TpGU887:M5/GUn*$]m%P#)8Ou_K[5GD*Hb3jXTb,c1QF!=SLf;Y5^"8"O;a"+;
N#BP!&c_\N]VoJ,+>4+b/0&6S&lQ0+dPln:1_kRfPA'1:c^)s(L#O#%7dEDsKfG$3BV78u7KsiUrR@Z`"'(a1p'(9Q,mOTIl<+r*lho,4J_"Idc(^mM6
JKqP&NL_C>R,gm*=B1uQbm38LXLNE<@3Q"J]9TT8ZaE&llsThXPX?ZDXk\7+*?,6MA/6'2eXlKean"7GiM2<+0;[7Q6`(Y;:Z3,d6?>1n)kk`0W'W?<*
o_S6"Ug#_sJKt1!?:M!G3YL]2<1LAX5m:TTZjgojr4%(Y?,o]@BD=(Z$6#/IJL%!rL"A)]K1;aEbFtAV.3qe*kt,pO"8s_Yjq8`[FgBg]$BlfD,l:LO7
YQ]hu[$?>5NGoEr1_IT55<iWi+Y0q`5MEn!b>*BsB_32o\O02((GL<]lIE4M1%F_7G;lR5;r*o/bhSNb'[/r2qrH5\kt_@).tg)a]V&eC\sF,bUq\8q%
DI(+HX_Hrr>8I\p?Hs`::D#k3k3TWr$\M[5#NL?Lod"j[V_1(^>q?$^u0/tt]'?)J-Zl`2NqEnc96!JnX;Rb1e*t6i5.?U9OE3*Im?<%cXK4+IqbI;+(
'jWfb_dFp-A1OHMMHlK-C<ZTpURO_(q],9o.u+_Q)D+1eb5KVD:pQ4,F)Y!gM0q]nr<jpIln-//+#7*X)q%T$j:#9bW2b072.2B%TU"NNWg)\aOiV<3-
5g):4UtjBMrhZDhs$_7$cO2=lj+I$Q2cQqnJ,p'rGjG131gbB#YV,\\JfW)l"C$1A6HBu:Ak-<c?D2u(ITWUqVOT3)<[kZ"?MA>4UI2$8Zfi9km5TYgM
&D4fC%!jSo7ONBb9a`gg/(OpW/4K*Te>MAjHBA2E(/"8]iEd\%Jhie<dK!Y<-SVVr'3qNltGM]PO55oqP"tHPNGBT=XqkIDLY!fFotYCW9*dOocklGG+
2&(,$JmSI;q%%Qnbse75NJXg)lSOTNI=14$c1i5#=R20"OWF*"mZ)*MBe?u"gm+0SE*IAH.fg;2/_@trUn>OGfW_N!RopM\"!aV-,cOef6]R+Zk&0o]T
,_7d^j;RI,F'm#1ibEshu<@AM`$@^$um,nGWIn[U.gI<eUqhpRS"[HY<C#acPgT4L6mCKG1oTF<,=LmH_&fu)b?":[Q=e^L@h<d.ei>-YEI'srg0uKhS
jMJMRn8AUSs!-%27,7W<B$pHViuF`)bE3<PYn<.n(es;#4(39OYk)/reD3tFCe+TV^+?VVcMjrK^g@d;D+L=O$[gbhLU+E.>%$0fpJ9l*(]h?0M.jXIQ
GP*;N9e%O2FHXL^-+%\_"B%bI0`0I!cKF2-K0jDn]&<:+KtB6Z@DqY/RJ4lGbQ!#)BNVV+FiS1mcS89]e]AiI9Z4K@%Ro,E6AR<E>u&HlIY*I-$Tq5l'
Lb:X]'A([dFG2oJ..XL7e%8UMp69*<1/CgMEAZ"eM9LW*2!HkL-H^1c=?r!,$^(WoF%QdAm;Zoo66K2j*2rWBSH`C&b(J#aYAJ1<6Zcj)pe(@QkVN4OC
@?W_,^9lPlVaEXl^b3K_BD\n"gdFfj*h!.M5u93a]DVbHT2Q^\qI`5FRY"SIVD7CcjVP!MC&f<g%>auZoR(f0a55M5ijJX8ng3oJs^V_%7eKiAp76kT2
8.siq%]2@fEF4j[sA+i&L.](C`OZna'ln1j-I$E^`Ef?_OM)0ljKlHfFfJZOs2XGY[\MPR<oMiU>WrCtuelQYu9).SSikU.]%(HD#_iX\4`0i*U_On>4
?9Uj$D.ph(-8DQCApYd(l&<+j>u_mf9.pA)h")XL%`"FO5!n[QVo@3Cag36RiSJ6IY4:M>U/AUO&p5t5J!-DAh'S4<&RK*S!tp5#*eqB<ha=gC(79fDS
_UUdlZ*;jM=3c&_#Ck02ujc1[35!9c.*Is+,I'*kW[bD7Jt(i1u3SX1oNR;/&9CDPk)%G,gB&q-5deBnn?uT"b6%5P1UYY[Cr`Cp_osW)R^'5g>"L3KL
Ylk%YC=dZ:_mV+H!S$%F&%-hh9dXbU'LH?XpS(NG(g/dNYn&<:&3AhC.,(n5Q.SL=0cia4\s>DeIY0;aq2Kg/k%6Ui-&!aSG'j'>h&?hND%@oa4n])86
tB!n;nZCPhhJ$k7,d(sZWW=!uqM4^24,1[ck_&?f7X"/c(5<G23>BZ_A>8L?M-lp]3mPlCs5q73%\ihPRq;VA[pP_H_O!gN^qhVEf.^bW%k7CKV#_a_>
^6-Z]?20Q41%fGWW_GuS3J,UYK25:en28`X$&.3D1>0D.[[0!`k(dKl;V8edUE'!5j73.;14#iZI@K:eHE;p/:NMD2F-aaaF3,,mtbI7)PW=;(tT3BO8
9jO^"U8S[lB]M0Fe*cp9p:+#V'T^mMaSYp5<Q:a=_$7ZeRj%n'AD9)&hJsL3.sl)e*Y@57]n:5hr7ZU-jO[CeUAD.VLd=nRL(\;j`BNZXQk!I1P6,,t'
Qt=P9e4iP,<uq`YMPE$qd=kD=KL='3,oJjH@Tn&E(ff,RQ1ZNW1-G=opV7SUZn)ZJ$hemQ8N]rkD:ZnZU`qC8>aA(VCgdVPJ-PAJhE,Wk>Q7(Z@Gq6>a
WJNq%$;LYa7LZpRXYWGQ:SbGmpUG%,0[0gVYV?aXJGIAZmBLU`iEsk[,qeTb`\[_j,l(%cH74I0"V!A2\D@;%(hc+,6g2G-qkc/P;<g)Ibal15d)6ePJ
m-Wd:@r.&Y<Z@67O;^h;[nR.s4f*:VYG9tB6KF)V)T^nBJn0JQYo<#2Qshbn;I7hQu?[q(q0?H.md-Y3Jkb\>i@=9(9(Id#ng(V@tr:"2V&4=,`5pIcO
5H:D,]_AePATuZ#EqAUupq=@m)P(SZj^*^dZ\cls=3p'#G@qO?Bh(#(7O31Apb;87&!lF"t62C<^^(blCIfF,>Ua.Dl4-Fl3S6Uo8UWrs&Am5JZ(tE\`
r(`W'63M`9&J4"-Bs%4%Z\JCf<\WtQBFZ!We(f84K&u`$O-C,6q-3YKZ'6dU[2(&OeG06FB!5=??,lB`>JpM<+o&RCp&_+),3oOZLiT(@bK]/3pODUe.
`[M!<Np'dY=`D.mMX?&6gWZGZ!&S-_+TcR>Q-ZX"P2/rKhm[:L+'jAluK^EbDngnEW2_$8c@r9k'cIL%e+483TXm4<]i+E/102e8lRV7oPiM0H'.`Sf!
&[E%1CGHI72c@d*NcJAE$4EgfQRKZP(eRs7`Y;(jPb0.@3hT-3sA&AW6h!KOb,Nd!2Y@ku8A+G3sL/C,%7hI-YmoACH)Gm@agLCj[padjZ5X[Fk/S*<"
X9!r;'o6SmRh^Po+8WKN<OkkQn'=C'rmU/49T>4TEj.4HAJi[q$hdB=Va1q0H\BN$+6+4eC78(WiU<:;>ceBOC0Jr#Y$]pqPK=/6NI+6K%<U'[+nUb8"
%jWb/#&JMTZ=IXEo3iq>Td8W^=Z7"oFWa)B*2Sm'48/5QtMh*7c+F>o0`rtef`&Q'-\ca%M?n(!pP7mEG?koSfj!H%.K1#_10\9HQfT31^iu4/)l<DcL
1MFI$J[Au`HDu_[#15`Dfh<4a\h1keGT\K;DO4][..]tqYo:`L(;]\QP%&ZD;4.QQJlf6QSGNAm38k./C8i&UBf`mL,YDs#^"LYrhUB_dDo4dAHBoF?Q
9iR3BJD1/B/I`gQ]"GeBgsu8^_#HAe9]Z`6[JnL6'V/tbH<K#!BP&F;b"=5hX,<mn(Y`>$,TKrqZd_j9(!;.f57Zp8o%pm%Js$t]lq!]$f^o@X'YqV>@
;LkNST`nO\sg$5lYd`Xk1l(^4rJYE.3U5;e/=eEf-APD[q5%C\)mZSbT2`c*_Of1sfYYohGe(7F0YZrgn5[VaW<h;4X[WqYL((J#m\!D&S<UB&TfZM"(
gd7[mk\/Eh)l.0;7nP2]U5**3S^nGbW.`YWeM+T4jr&`UOJ_pW!``B2q;30$D_!F%9<JeX/XEXkK3c%W+!"VRSY4.aH=2%9PDaD$2["jl(eFr4+aZtC%
O]LhbK)@pJ!Cd\@KkRU<6BR,I+X(_:s6M5RieHSu4A*F-bSttC_a=_B.%9#JlSihjtE_nRSm)k(Kk[8/cU[F$[YE9Z1QSi!>FN6neC\LWL\:WcuqPMR-
gqH+K4g/BXJR('aH'Lu$ZYaO_-1GKbj1d(mX%O=Moneo5Hq%fBV;q#;#S-sld7om>DEeR[C&7)d$*-L0,Rg.i1Mm9*`X[j0@O2eNW\RAqPQBs>5A/7)o
HDsrfQa!a:?@R;.P!]2,]>(PKLG0UH%Zj\V>uj(i]A`nKJ[pU(gbDor+SD[>ca3a.WatkheNpKW6`e7nZQ#M;*T,9V((5DU?cZZ1JIo`SuQTVM_i\-E.
?LKIVn_M$G_n4J)7mD>%6e0$(_cDq4Y23f)?K4SG&?.FaQm&-k#"c**U8"=V,Z;BB:t**p(=E62PGn3To!B>.IB-T>7k`13I>.YX6K9n+9/$a?U)??WK
%&(\A,#Hq4JrJ;.2Hn#"Xu@&[T(Ng*umO`-dh(\(C0dF#fSG@iJK0jLhZs0kut\&50YqC"WcOe'/V`uhB)>Oj1&+1$%>bU;8pYi3o0!5VJ?6/5s1B8Dp
F]84<Y\%^<3HdZ==VkK"?EpOWI&PLQ0QYRH^?T47jH9%,:8Z-h%h%Cg'X094q;0!,$AaHk^c@\'RbY:_57R7p*D=&>:'7XUpo<>VD4c\A?F1lgUnf8@C
6#MAiPJ.c4p5l`XkQ!KqESq]CoE.I/3Wc\?SEb+J3@_=s"'E#SAVLXR;o,2a+lpg-2Q,jUBpF.Ya6aZ4On@8p9(SYt/*^)5MQje:`9ZIW[1e!(mgi#mC
R0ll=/-"61`@TO#lgfjEs6Y.0XWd4,Rg1dKs"$1.tN,emV';`#sL'CK7d9?(h\Geq%>+Z]Dqp3`9UO'`;@Y_!!1M[^Zbe"FZdWIZ,C;P!WW3#!!HG.'''
	exec(lzma.decompress(base64.a85decode(_SRC_B85)).decode("utf-8"), multiCMD.__dict__)
try:
	import xxhash  # type: ignore
	hasher = xxhash.xxh64()
	xxhash_available = True
except ImportError:
	import hashlib
	hasher = hashlib.blake2b()
	xxhash_available = False

version = '9.73'
__version__ = version
COMMIT_DATE = '2026-10-16'

MAGIC_NUMBER = 1.61803398875
RANDOM_DESTINATION_SELECTION = False

BYTES_RATE_LIMIT = 0
FILES_RATE_LIMIT = 0

COMMAND_TIMEOUT = 0
NO_CREATE_DIR = False
CONTENT_ONLY = False
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
	'FS param build warning': 0,
	'FS param warning': 0,
}
RETURNCODE_TO_ERROR_TABLE = {v: k for k, v in ERROR_TO_RETURNCODE_TABLE.items()}

REMOVE_FILES_WHILE_LISTING = True

# set limit to 1m files
SINGLE_PASS_FILE_LIMIT = 1_000_000

#%% ---- Helper Functions ----
def eprint(*args, **kwargs):
	global ERRORS
	try:
		if 'file' in kwargs:
			print(*args, **kwargs)
		else:
			print(*args, file=sys.stderr, **kwargs)
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
			self.last_call_args = callArgs
			multiCMD.print_progress_bar(*callArgs)
	def stop(self):
		self.stop_flag = True
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
@functools.lru_cache(maxsize=None)
def check_path(program_name):
	"""
	Check if the given program is in the system path.

	Args:
		program_name (str): The name of the program to check.

	Returns:
		bool: True if the program is found, False otherwise.
	"""
	#global __configs_from_file
	global _binPaths
	config_key = f'_{program_name}Path'
	program_path = (
		#__configs_from_file.get(config_key) or
		os.environ.get(program_name.upper() + '_PATH') or
		globals().get(config_key) or
		shutil.which(program_name)
	)
	if program_path:
		_binPaths[program_name] = program_path
		return True
	return False

_binCalled = {'lsblk', 'losetup', 'sgdisk', 'blkid', 'umount', 'mount','dd','cp', 'xcopy',
			  'truncate',
			  'mkfs', 'mkfs.btrfs', 'mkfs.xfs', 'mkfs.f2fs', 'mkfs.ntfs', 'mkfs.vfat', 'mkfs.exfat', 'mkfs.hfsplus',
			  'mkudffs', 'mkfs.jfs', 'mkfs.reiserfs', 'newfs', 'mkfs.bfs', 'mkfs.minix', 'mkswap',
			  'e2fsck', 'btrfs', 'xfs_repair', 'xfs_info', 'fsck.f2fs', 'dump.f2fs', 'ntfsfix', 'ntfsinfo',
			  'fsck.fat', 'fsck.exfat', 'dump.exfat', 'fsck.hfsplus',
			  'fsck.hfs', 'fsck.jfs', 'fsck.reiserfs', 'fsck.ufs', 'fsck.minix',
			  'tune2fs', 'dumpe2fs', 'xfs_admin', 'exfatlabel', 'udflabel', 'udfinfo', 'jfs_tune',
			  'reiserfstune', 'debugreiserfs', 'swaplabel', 'udevadm', 'e2label', 'fatlabel', 'ntfslabel', 'btrfstune', 'f2fslabel'}
[check_path(program) for program in _binCalled]

def run_command_in_multicmd_with_path_check(command, timeout=...,max_threads=1,quiet=False,dry_run=False,strict=False):
	"""
	Run a command in multiCMD with path check.

	Args:
		command (str): The command to run.
		timeout (int, optional): The timeout value in seconds. Defaults to COMMAND_TIMEOUT.
		max_threads (int, optional): The maximum number of threads to use. Defaults to 1.
		quiet (bool, optional): Whether to suppress the output. Defaults to False.
		dry_run (bool, optional): Whether to perform a dry run. Defaults to False.
		strict (bool, optional): Whether to exit if the command fails to find the bin. Defaults to True.

	Returns:
		list: The output of the command.
	"""
	return run_commands_in_multicmd_with_path_check([command], timeout=timeout, max_threads=max_threads, quiet=quiet, dry_run=dry_run, strict=strict)[0]

def run_commands_in_multicmd_with_path_check(commands, timeout=...,max_threads=1,quiet=False,dry_run=False,strict=False):
	"""
	Run commands in multiCMD with path check.

	Args:
		commands (list): The commands to run.
		timeout (int, optional): The timeout value in seconds. Defaults to COMMAND_TIMEOUT.
		max_threads (int, optional): The maximum number of threads to use. Defaults to 1.
		quiet (bool, optional): Whether to suppress the output. Defaults to False.
		dry_run (bool, optional): Whether to perform a dry run. Defaults to False.
		strict (bool, optional): Whether to exit if the command fails to find the bin. Defaults to True.

	Returns:
		list: The outputs of the commands.
	"""
	global _binPaths
	global COMMAND_TIMEOUT
	global ERRORS
	if timeout is ...:
		timeout = COMMAND_TIMEOUT
	newCommands = []
	for command in commands:
		# Check the path of the command
		if isinstance(command, str):
			command = command.split()
		if not command:
			eprint("Command is empty: The input command is empty")
			continue
		if not isinstance(command[0],str):
			command[0] = str(command[0])
		if command[0] not in _binPaths and not check_path(command[0]):
			eprint(f"Command not found: Command '{command[0]}' not found. Please consider installing it then retry.")
			if strict: 
				raise RuntimeError(f"Command not found: Command '{command[0]}' not found. Please consider installing it then retry.")
		newCommands.append(command)
	# Run the command
	tasks = multiCMD.run_commands(newCommands, timeout=timeout, max_threads=max_threads, quiet=quiet, dry_run=dry_run,return_object=True)
	for task in tasks:
		error = ''
		if task.returncode == 124 and task.stderr[-1].strip().startswith('Timeout!'):
			error = f"Command timeout: Command '{task.command}' timed out after {timeout} seconds."
		elif task.returncode == 137 and task.stderr[-1].strip().startswith('Ctrl C detected, Emergency Stop!'):
			error = f"Keyboard Interrupt: Command '{task.command}' was interrupted by user."
		elif task.returncode != 0:
			program = ''
			cmd = getattr(task, 'command', None)
			if isinstance(cmd, (list, tuple)) and cmd:
				program = os.path.basename(str(cmd[0]))
			elif isinstance(cmd, str) and cmd.strip():
				program = os.path.basename(cmd.split()[0])
			# e2fsck encodes "errors corrected" as bits 0 and 1 (exit 1, 2, or 3).
			# A live clone's journal replay is exit 1; that is success, not a
			# failed delayed UUID rewrite.
			if program == 'e2fsck' and task.returncode in (1, 2, 3):
				error = ''
			else:
				error = f"Task return code error: Command '{task.command}' failed with return code {task.returncode}."
		if error:
			if not quiet:
				eprint(error)
			else:
				ERRORS.append(error)
			if strict:
				raise RuntimeError(error)
	return [task.stdout for task in tasks]

def get_free_space_bytes(path):
	if not os.path.exists(path):
		path = get_last_existing_parent(path)
	stat = os.statvfs(path)
	return stat.f_bavail * stat.f_frsize  # available blocks * fragment size

@functools.lru_cache(maxsize=None)
def cp_supports_sparse():
	"""
	Whether the cp found on PATH accepts --sparse (GNU coreutils does, BSD / macOS cp does not).
	"""
	if os.name != 'posix':
		return False
	try:
		task = multiCMD.run_commands([[_binPaths.get('cp', 'cp'), '--version']], timeout=10, quiet=True, return_object=True)[0]
		return task.returncode == 0 and 'GNU' in '\n'.join(task.stdout)
	except Exception:
		return False

def get_file_size(path):
	"""
	Get the on-disk (allocated) size of a file, falling back to the apparent size.
	Use os.path.getsize() where the apparent size is needed, e.g. for content comparison.
	"""
	try:
		st = os.stat(path,follow_symlinks=False)
		# stat_result is a tuple: `'st_blocks' in st` would compare against field values, not names
		if hasattr(st, 'st_rsize'):
			realSize = st.st_rsize
		elif hasattr(st, 'st_blocks'):
			realSize = st.st_blocks * 512
		else:
			realSize = st.st_size
	except Exception:
		try:
			realSize = os.path.getsize(path)
		except Exception:
			realSize = 0
	return realSize

def get_timestamp_precision(num):
	"""
	Get the precision of a timestamp.

	Args:
		num (float): The number to get the precision of.

	Returns:
		int: The precision of the number. Possible precisions are 2, 1,  0.01, 0.0000001 , 1e-9

	Examples:
		>>> get_timestamp_precision(10)
		0
		>>> get_timestamp_precision(10.0)
		2
		>>> get_timestamp_precision(10.5)
		0.01
	"""
	if isinstance(num, int):
		return 0
	elif isinstance(num, float):
		if num / 2 == int(num / 2):
			return 2
		if num == int(num):
			return 1
		if num * 100 == int(num * 100):
			return 0.01
		if num * 1e7 == int(num * 1e7):
			return 0.0000001
		return 1e-9
	return 0

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
			fs_type = run_command_in_multicmd_with_path_check(["df", "-PT", path],quiet=True)[-1].split()[1]
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

def _on_worker_start():
	parent = os.getppid()
	def watch_parent():
		# every 0.5s, check if parent still exists
		while True:
			if os.getppid() != parent:
				os._exit(1)
			time.sleep(0.5)
	t = threading.Thread(target=watch_parent, daemon=True)
	t.start()

_worker_initialized = False
def _init_worker():
	"""Run _on_worker_start() once per worker process.
	Used instead of ProcessPoolExecutor's initializer parameter for Python 3.6 compatibility."""
	global _worker_initialized
	if not _worker_initialized:
		_on_worker_start()
		_worker_initialized = True

# The run settings hpcp() keeps in module globals. A pool worker that did not fork from the parent
# (spawn, forkserver) would otherwise run with the import-time defaults.
_TASK_SETTINGS = ('CONTENT_ONLY', 'NO_CREATE_DIR', 'RANDOM_DESTINATION_SELECTION', 'HASH_SIZE', 'COMMAND_TIMEOUT',
	'BYTES_RATE_LIMIT', 'FILES_RATE_LIMIT', 'MIRROR_FS_PARAMS', 'REMOVE_FILES_WHILE_LISTING')

def _submit_task(executor, func, *args, **kwargs):
	"""Submit func to a process pool with this process's run settings; collect it with _task_result."""
	settings = {name: globals()[name] for name in _TASK_SETTINGS}
	return executor.submit(_run_task, settings, func, *args, **kwargs)

def _run_task(settings, func, *args, **kwargs):
	"""Worker side of _submit_task: (result, the errors func recorded, None), or (None, errors, crash text) when it raised.

	ERRORS is process-local, so the errors go back through the future; a reused worker never carries them into its next task.
	"""
	_init_worker()
	globals().update(settings)
	error_start = len(ERRORS)
	try:
		return func(*args, **kwargs), ERRORS[error_start:], None
	except Exception as e:
		return None, ERRORS[error_start:], f'{type(e).__name__}: {e}'
	finally:
		del ERRORS[error_start:]

def _task_result(future, default, crash_prefix):
	"""A _submit_task future's result in the parent. The errors the task recorded join ERRORS (the worker printed them);
	a task that raised, or whose worker died (BrokenProcessPool), is recorded under crash_prefix and gives default.
	Either way the other tasks go on (a broken pool then fails the next submit)."""
	try:
		result, errors, crash = future.result()
	except Exception as e:
		result, errors, crash = None, [], f'{type(e).__name__}: {e}'
	ERRORS.extend(errors)
	if crash is not None:
		eprint(f'{crash_prefix}: {crash}')
		return default
	return result

def _raise_task_failures(future):
	"""A _submit_task future's result, raising its recorded failures (and a crash) as one RuntimeError, as
	_call_raising_recorded_failures does in-process."""
	result, errors, crash = future.result()
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

def _excluded_or_below(path, exclude):
	"""True when path, or a directory above it, matches exclude: the listing prunes an excluded directory with
	everything below it, so a path reached another way (a symlink's target) is checked against its parents too."""
	return bool(exclude) and any(is_excluded(part, exclude) for part in _path_and_parents(os.path.normpath(os.path.abspath(path))))

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
	partitioned md array built on the disk): a row is a partition of the disk only when its TYPE is part and its
	parent (PKNAME) is the disk itself (its KNAME). -P prints KEY="value" pairs, so the disk's empty PKNAME cannot
	shift the columns.
	"""
	names = columns.split(',')
	lines = run_command_in_multicmd_with_path_check(['lsblk', '-p', '-nb', '-P', '-o', columns + ',KNAME,PKNAME,TYPE', disk], strict=True)
	rows = [_lsblk_pairs(line) for line in lines if line.strip()]
	if not rows:
		raise RuntimeError(f'Disk error: lsblk printed nothing for {disk}')
	disk_row = rows[0]
	partitions = [row for row in rows[1:] if row.get('TYPE') == 'part' and row.get('PKNAME') == disk_row.get('KNAME')]
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
	fs_type = run_command_in_multicmd_with_path_check(["lsblk", '-nbl', '-o', 'FSTYPE', path],strict=True)[0].strip()
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
		resolved = [_binPaths.get(command[0], command[0])] + list(command[1:])
		tasks = multiCMD.run_commands([resolved], timeout=COMMAND_TIMEOUT, max_threads=1, return_object=True)
		if not tasks:
			return 1, ''
		rc = getattr(tasks[0], 'returncode', 1)
		stderr = getattr(tasks[0], 'stderr', None) or []
		return rc, (stderr[-1].strip() if stderr else '')

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

def _ext_mkfs_task_text(task):
	"""Join a multiCMD Task's stdout and stderr into one searchable string."""
	chunks = []
	for attr in ('stderr', 'stdout'):
		val = getattr(task, attr, None) or []
		if isinstance(val, str):
			chunks.append(val)
		else:
			chunks.extend(str(line) for line in val)
	return '\n'.join(chunks)

def _ext_mkfs_probe_features(names):
	"""Ask local mke2fs whether each -O feature name parses. Cache the answers.

	Uses `mke2fs -n` against a nonexistent path so nothing is created. An
	unknown name fails immediately with 'Invalid filesystem option set'; a
	known name gets past option parsing and then fails on the dummy path,
	which still means the name is known. Probe failures fail open (known)
	so a missing mkfs does not strip a usable -O list.
	"""
	todo = [n for n in dict.fromkeys(names) if n and n not in _EXT_MKFS_FEATURE_KNOWN]
	if not todo:
		return
	try:
		mkfs = _binPaths.get('mkfs', 'mkfs')
		commands = [
			[mkfs, '-t', 'ext4', '-n', '-F', '-O', name, '/hpcp-ext-feature-probe', '256']
			for name in todo
		]
		tasks = multiCMD.run_commands(
			commands, timeout=COMMAND_TIMEOUT, max_threads=min(8, len(commands)),
			return_object=True, quiet=True)
		if not tasks:
			for name in todo:
				_EXT_MKFS_FEATURE_KNOWN.setdefault(name, True)
			return
		for name, task in zip(todo, tasks):
			text = _ext_mkfs_task_text(task)
			_EXT_MKFS_FEATURE_KNOWN[name] = 'Invalid filesystem option set' not in text
		for name in todo[len(tasks):]:
			_EXT_MKFS_FEATURE_KNOWN.setdefault(name, True)
	except Exception:
		for name in todo:
			_EXT_MKFS_FEATURE_KNOWN.setdefault(name, True)

def _ext_mkfs_knows_feature(name):
	"""True if local mke2fs parses this -O feature name.

	One unknown name, including a negation of a feature this e2fsprogs has
	never heard of, makes mke2fs reject the entire -O list (e2fsprogs 1.45:
	^orphan_file, ^fast_commit). Callers drop those names and keep the rest.
	"""
	if name not in _EXT_MKFS_FEATURE_KNOWN:
		_ext_mkfs_probe_features([name])
	return _EXT_MKFS_FEATURE_KNOWN.get(name, True)

def _probe_ext(device):
	"""Read ext2/3/4 geometry and features from dumpe2fs."""
	params = {}
	for line in run_command_in_multicmd_with_path_check(['dumpe2fs', '-h', device], quiet=True):
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
			if _ext_mkfs_knows_feature(name):
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

def _probe_xfs(device):
	"""
	Read xfs geometry and features from xfs_info.

	xfs_info groups values into sections (meta-data / data / naming / log /
	realtime) and reuses key names across them - `bsize` is the block size under
	`data` but the directory block size under `naming` - so values are bucketed
	by section rather than flattened.
	"""
	params = {}
	section = ''
	for line in run_command_in_multicmd_with_path_check(['xfs_info', device], quiet=True):
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
	for line in run_command_in_multicmd_with_path_check(['btrfs', 'inspect-internal', 'dump-super', device], quiet=True):
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
	for line in run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'export', device], quiet=True):
		key, sep, value = line.partition('=')
		if sep and key.strip() == 'VERSION':
			version = value.strip().upper()
			if version.startswith('FAT') and version[3:].isdigit():
				params['fat_bits'] = int(version[3:])
	for line in run_command_in_multicmd_with_path_check(['fsck.fat', '-nv', device], quiet=True):
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
	for line in run_command_in_multicmd_with_path_check(['ntfsinfo', '-m', device], quiet=True):
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
	for line in run_command_in_multicmd_with_path_check(['dump.exfat', device], quiet=True):
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
	for line in run_command_in_multicmd_with_path_check(['dump.f2fs', '-d', '1', device], quiet=True):
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
	for line in run_command_in_multicmd_with_path_check(['udfinfo', device], quiet=True):
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
	for line in run_command_in_multicmd_with_path_check(['debugreiserfs', device], quiet=True):
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
	The partitions are lsblk's TYPE part rows whose parent is the disk, with full paths (see _lsblk_disk_rows): a
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
	result = run_command_in_multicmd_with_path_check(["sgdisk", '--info='+partition, device],strict=True)
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
	partitions = run_command_in_multicmd_with_path_check(["sgdisk", '--print', device],strict=True)
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
			start_sector = int(run_command_in_multicmd_with_path_check(["sgdisk", '--first-aligned-in-largest', image])[0].strip())
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
def create_sym_links(symLinks,exclude=None,no_link_tracking=False):
	"""Recreate symLinks ({src link: [dest paths]}). Returns, with link tracking, a (destination path, source path) pair
	for each relative link's target that exists at the source (copied now or already there, or excluded): -rme keeps
	those destination paths, except where the copy itself covers them (see _kept_paths)."""
	global RANDOM_DESTINATION_SELECTION
	tracked_targets = set()
	if len(symLinks) == 0:
		return tracked_targets
	nestedSymLinks = {}
	counter = 0
	print(f"\nFound Symbolic Links:   {len(symLinks)}")
	if no_link_tracking:
		print("Skipping copying file as no_link_tracking ...\n")
	#print(symLinks)
	start_time = time.monotonic()
	for src, dests in symLinks.items():
		try:
			src = os.path.normpath(src)
			dests = [os.path.normpath(d) for d in dests]
			if exclude and is_excluded(src,exclude):
				print(f"\n{pformat(src)} is excluded, skipping...")
				continue
			for d in dests:
				if os.path.islink(d):
					os.remove(d)
			# A real file or directory already at the link's path (in any dest: a file lives in one) keeps its place.
			occupied = next((d for d in dests if os.path.exists(d)), None)
			if occupied is not None:
				# An error, not a note: the destination does not match the source, and a move (-rm) must not go on.
				eprint(f"Failed to create symbolic link: {pformat(occupied)} is a {'directory' if os.path.isdir(occupied) else 'file'} "
					f"in the destination, skipping the link {pformat(src)}")
				continue
			# Determine if the link is a absolute link or relative link
			linkedTargetFile = os.readlink(src)
			dest = ''
			# The link's target at the source when tracking applies to it (relative, link tracking on); keep_target: its
			# destination copy belongs to the copy (the target exists at the source), and -rme keeps it (_kept_paths).
			sourceLinkedFile = None
			keep_target = False
			if not os.path.isabs(linkedTargetFile):
				# Files are spread over the dests: a relative link goes where its target is, so that it resolves.
				destLinkedFiles = [os.path.join(os.path.dirname(d), linkedTargetFile) for d in dests]
				dest = next((d for d, destLinkedFile in zip(dests, destLinkedFiles) if os.path.exists(destLinkedFile)), '')
				if not no_link_tracking:
					# The target by the link's own spelling with '..' resolved, so -e patterns match it and the tracked tree;
					# used only when the kernel resolves the link to that same file (no '..' after a symlinked directory).
					linked = os.path.join(os.path.dirname(src), linkedTargetFile)
					sourceLinkedFile = os.path.normpath(linked)
					note = None
					if os.pardir in linkedTargetFile.split(os.sep) and os.path.realpath(sourceLinkedFile) != os.path.realpath(linked):
						note = f"\n{pformat(src)} -> {pformat(linkedTargetFile)} goes through a symlinked directory: its target is not tracked"
					elif _excluded_or_below(sourceLinkedFile, exclude):
						# Not copied, but an earlier copy of it in the destination is kept, as for any excluded path.
						keep_target = True
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
					os.symlink(linkedTargetFile, dest, target_is_directory=os.path.isdir(linkedTargetFile))
					if keep_target:
						tracked_targets.add((os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(dest)), linkedTargetFile)), sourceLinkedFile))
					break
				except Exception:
					# Notes: only 'Failed to create symbolic link' below, when every destination failed, is the error.
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


	endTime = time.monotonic()
	print(f"Time taken:             {endTime-start_time:0.4f} seconds")
	if len(nestedSymLinks) > 0:
		print(f"\nNested Symbolic Links:   {len(nestedSymLinks)}")
		tracked_targets |= create_sym_links(nestedSymLinks,exclude=exclude,no_link_tracking=no_link_tracking)
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
	hasher = xxhash.xxh64() if xxhash_available else hashlib.blake2b()
	with open(path, 'rb') as f:
		if not full_hash:
			# Only hash the last hash_size bytes
			#f.seek(-min(1<<16,size), os.SEEK_END)
			f.seek(-min(HASH_SIZE,size), os.SEEK_END)
		for chunk in iter(lambda: f.read(4096), b''):
			hasher.update(chunk)
	return hasher.hexdigest()

def is_file_identical(src_path, dest_path,src_size,full_hash=False,compare_metadata=False):
	dst_size = os.path.getsize(dest_path)
	# try to find the mtime are different
	if src_size != dst_size:
		return False
	try:
		src_mtime = os.path.getmtime(src_path)
		dst_mtime = os.path.getmtime(dest_path)
		precision = max(get_timestamp_precision(src_mtime), get_timestamp_precision(dst_mtime))
		if abs(src_mtime - dst_mtime) > precision * 2:
			return False
	except Exception:
		pass
	if hash_file(src_path,src_size,full_hash) != hash_file(dest_path,dst_size,full_hash):
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

def _get_file_list_cache_key(path, exclude, append_hash, full_hash):
	exclude_key = frozenset(exclude) if isinstance(exclude, (list, set)) else exclude
	return (path, exclude_key, append_hash, full_hash)

def get_file_list(path, exclude=None, append_hash=False, full_hash=False, max_workers=56, drop_cache=False, parallel_file_listing=True, remove_files_while_listing=False, return_removed_count=False):
	"""Unified file list getter with a single cache. Dispatches to get_file_list_serial or get_file_list_parallel based on serial.

	With return_removed_count, returns a 5-tuple whose last element is the number of
	files/links unlinked during the scan (always 0 unless remove_files_while_listing).
	"""
	cache_key = _get_file_list_cache_key(path, exclude, append_hash, full_hash)
	if drop_cache or remove_files_while_listing:
		# In deletion mode the scan itself is what unlinks the files, so a cached list
		# would silently skip the removal. Always drop it and walk the tree for real.
		_get_file_list_cache.pop(cache_key, None)
	if cache_key in _get_file_list_cache:
		result = _get_file_list_cache[cache_key]
		return (*result, 0) if return_removed_count else result
	start_time = time.monotonic()
	print(f"Getting file list for {path}")
	if remove_files_while_listing:
		print("Note: removing files while listing")
	if parallel_file_listing:
		file_list, links, size, folders, removed_count = get_file_list_parallel(path, max_workers=max_workers, exclude=exclude, append_hash=append_hash, full_hash=full_hash, remove_files_while_listing=remove_files_while_listing)
	else:
		file_list, links, size, folders, removed_count = get_file_list_serial(path, exclude=exclude, append_hash=append_hash, full_hash=full_hash, remove_files_while_listing=remove_files_while_listing)
	result = (file_list, links, size, folders)
	if remove_files_while_listing:
		print(f"Removed: {removed_count} files/links with size of {format_bytes(size,to_str=True)}B")
		# Do not cache the result if we are in deletion mode
	else:
		_get_file_list_cache[cache_key] = result
	print(f"Time taken to get file list: {time.monotonic() - start_time:0.4f} seconds")
	return (*result, removed_count) if return_removed_count else result

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

def _remove_file_or_link(path):
	try:
		os.remove(path)
		return True
	except Exception as exc:
		eprint(f'Remove process exception:\n Deleting {path} generated an exception: {exc}')
		return False

def _scan_dir(root):
	"""os.scandir(root)'s entries. A directory that cannot be opened or read on (EACCES, EIO) is recorded as a
	'Listing error' and gives the entries read so far, so the rest of the tree is still listed."""
	try:
		with os.scandir(root) as entries:
			yield from entries
	except OSError as e:
		eprint(f'Listing error: {root}: {e}')

def get_file_list_serial(root,exclude=None,append_hash=False,full_hash=False,recurse=True,_rate_limiter=None,respect_file_rate_limit=True,remove_files_while_listing=False):
	# skip if path is longer than 4096 characters
	if len(root) > 4096:
		return frozenset() ,frozenset(),0,frozenset(),0
	#print(f'Getting file list for {root}')
	global FILES_RATE_LIMIT
	if _rate_limiter is None and respect_file_rate_limit and FILES_RATE_LIMIT > 0:
		_rate_limiter = {'start_time': time.monotonic(), 'count': 0, 'limit': FILES_RATE_LIMIT}
	if exclude and is_excluded(root,exclude):
		return frozenset() ,frozenset(),0,frozenset(),0
	if os.path.islink(root):
		_file_list_rate_limit(_rate_limiter)
		removed_count = 0
		if remove_files_while_listing:
			removed_count = int(_remove_file_or_link(root))
			return frozenset() ,frozenset(),0,frozenset(),removed_count
		return frozenset() ,frozenset([get_file_repr(root,append_hash,full_hash)]),0,frozenset(),0
	if os.path.isfile(root):
		_file_list_rate_limit(_rate_limiter)
		realSize = get_file_size(root)
		removed_count = 0
		if remove_files_while_listing:
			removed_count = int(_remove_file_or_link(root))
			return frozenset() ,frozenset(), realSize,frozenset(),removed_count
		return frozenset([get_file_repr(root,append_hash,full_hash)]) ,frozenset(), realSize,frozenset(),0
	file_list = set()
	links = set()
	folders = set()
	size = 0
	removed_count = 0
	iteration = 0
	start_time = time.monotonic()
	globalStartTIme = start_time
	if os.path.isdir(root):
		folders.add(root)
		for entry in _scan_dir(root):
			# update the progress bar every 0.5 seconds
			currentTime = time.monotonic()
			if currentTime - start_time > 0.5:
				start_time = currentTime
				# use the time passed as the iteration number
				iteration = int(currentTime - globalStartTIme)
				suffix = ''
				if file_list:
					suffix += f'Files: {format_bytes(len(file_list),use_1024_bytes=False,to_str=True)} '
				if removed_count:
					suffix += f'Removed: {format_bytes(removed_count,use_1024_bytes=False,to_str=True)} '
				if links:
					suffix += f'Links: {format_bytes(len(links),use_1024_bytes=False,to_str=True)} '
				if folders:
					suffix += f'Folders: {format_bytes(len(folders),use_1024_bytes=False,to_str=True)} '
				if size:
					suffix += f'Size: {format_bytes(size)}B '
				if _rate_limiter:
					suffix += f'| Overall: {format_bytes(_rate_limiter["count"],use_1024_bytes=False,to_str=True)} Files'
				# if the root is longer than 50 characters, we only show the last 50 characters
				multiCMD.print_progress_bar(iteration=iteration, total=0, prefix=f'{root}'[-50:], suffix=suffix)
			if exclude and is_excluded(entry.path,exclude):
				continue
			try:
				if entry.is_symlink():
					_file_list_rate_limit(_rate_limiter)
					realSize = get_file_size(entry.path)
					size += realSize
					if remove_files_while_listing:
						removed_count += int(_remove_file_or_link(entry.path))
					else:
						links.add(get_file_repr(entry.path,append_hash,full_hash))
				elif entry.is_file(follow_symlinks=False):
					_file_list_rate_limit(_rate_limiter)
					realSize = get_file_size(entry.path)
					size += realSize
					if remove_files_while_listing:
						removed_count += int(_remove_file_or_link(entry.path))
					else:
						file_list.add(get_file_repr(entry.path,append_hash,full_hash))
				elif entry.is_dir(follow_symlinks=False):
					if recurse:
						dir_files, dir_links, dir_size, dir_folders, dir_removed_count = get_file_list_serial(entry.path,exclude=exclude,append_hash=append_hash,full_hash=full_hash,_rate_limiter=_rate_limiter,respect_file_rate_limit=respect_file_rate_limit,remove_files_while_listing=remove_files_while_listing)
						file_list.update(dir_files)
						links.update(dir_links)
						size += dir_size
						folders.update(dir_folders)
						removed_count += dir_removed_count
					else:
						folders.add(entry.path)
			except OSError as e:
				# One entry that cannot be read (its hash, a link target) is recorded; the others are still listed.
				eprint(f'Listing error: {entry.path}: {e}')
		#multiCMD.print_progress_bar(iteration=iteration, total=iteration, prefix=f'{root}', suffix=f'Files: {format_bytes(len(file_list),use_1024_bytes=False,to_str=True)} Links: {format_bytes(len(links),use_1024_bytes=False,to_str=True)} Folders: {format_bytes(len(folders),use_1024_bytes=False,to_str=True)} Size: {format_bytes(size)}B')
	else:
		eprint(f'Source path type error: {root} is not a file or directory')
		return frozenset([root]) ,frozenset(), 0,frozenset(),0
	return frozenset(file_list), frozenset(links) , size, frozenset(folders - set(['.', '..'])),removed_count

def get_file_list_parallel(path,max_workers=56,exclude=None,append_hash=False,full_hash=False,remove_files_while_listing=False):
	global FILES_RATE_LIMIT
	# skip if path is longer than 4096 characters
	if len(path) > 4096:
		return frozenset() ,frozenset(),0,frozenset(),0
	if exclude and is_excluded(path,exclude):
		return frozenset() ,frozenset(),0,frozenset(),0
	if os.path.islink(path):
		removed_count = 0
		if remove_files_while_listing:
			removed_count = int(_remove_file_or_link(path))
			return frozenset() ,frozenset(),0,frozenset(),removed_count
		return frozenset() ,frozenset([get_file_repr(path,append_hash,full_hash)]),0,frozenset(),0
	if os.path.isfile(path):
		realSize = get_file_size(path)
		removed_count = 0
		if remove_files_while_listing:
			removed_count = int(_remove_file_or_link(path))
			return frozenset() ,frozenset(), realSize,frozenset(),removed_count
		return frozenset([get_file_repr(path,append_hash,full_hash)]) ,frozenset(), realSize,frozenset(),0
	#print(f'Getting file list for {path}')
	if os.path.isdir(path):
		file_list, link_list , size, folder_list, removed_count = get_file_list_serial(
			path,
			exclude=exclude,
			append_hash=append_hash,
			full_hash=full_hash,
			recurse=False,
			respect_file_rate_limit=False,
			remove_files_while_listing=remove_files_while_listing
		)
		file_list = set(file_list)
		link_list = set(link_list)
		folder_list = set(folder_list)
		folders_to_expand = deque(folder_list - set([path]))
		enqueued_folders = set(folders_to_expand)
		futures = {}
		listing_start_time = time.monotonic()
		listed_items = len(file_list) + len(link_list) + removed_count
		with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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
						remove_files_while_listing=remove_files_while_listing
					)] = folder
					under_rate_limit = (
						FILES_RATE_LIMIT <= 0
						or listed_items <= FILES_RATE_LIMIT * (time.monotonic() - listing_start_time)
					)
				if not futures:
					if folders_to_expand and FILES_RATE_LIMIT > 0:
						time.sleep(0.05)
					continue
				done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
				for doneTask in done:
					folder_path = futures.pop(doneTask)
					dir_file_list, dir_link_list, dir_size, dir_folder_list, dir_removed_count = _task_result(
						doneTask, (frozenset(), frozenset(), 0, frozenset(), 0), 'Listing error')
					file_list.update(dir_file_list)
					link_list.update(dir_link_list)
					size += dir_size
					removed_count += dir_removed_count
					listed_items += len(dir_file_list) + len(dir_link_list) + dir_removed_count
					for sub_folder in dir_folder_list:
						if sub_folder != folder_path and sub_folder not in enqueued_folders:
							folders_to_expand.append(sub_folder)
							enqueued_folders.add(sub_folder)
					folder_list.update(dir_folder_list)
			
	else:
		eprint(f'Source path type error: {path} is not a file or directory')
		return frozenset([path]) ,frozenset(), 0,frozenset(),0
	return frozenset(file_list), frozenset(link_list) ,size, frozenset(folder_list - set(['.', '..'])),removed_count


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
					if exclude:
						# The listing pass may have left protected entries behind.
						# Recheck exclusions during cleanup, including empty directories.
						with os.scandir(path) as entries:
							children = [entry.path for entry in entries]
						total_size += delete_file_bulk(children, exclude=exclude)[0]
						try:
							os.rmdir(path)
						except OSError as exc:
							if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
								raise
					else:
						shutil.rmtree(path)
				else:
					os.remove(path)
			except Exception as exc:
				eprint(f'Remove process exception:\n Deleting {path} generated an exception: {exc}')
	endTime = time.monotonic()
	return total_size, endTime - start_time

def delete_file_list_parallel(file_list, max_workers, verbose=False,files_per_job=1,init_size=0):
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
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while file_list_iterator or futures:
			# counter = 0
			under_limit = apb.under_rate_limit()
			while file_list_iterator and len(futures) < max_scheduled_jobs and under_limit:
				delete_files = []
				try:
					for _ in range(round(files_per_job)):
						delete_files.append(next(file_list_iterator))
					future = _submit_task(executor, delete_file_bulk, delete_files)
					futures[future] = delete_files
					# counter += 1
				except StopIteration:
					if delete_files:
						future = _submit_task(executor, delete_file_bulk, delete_files)
						futures[future] = delete_files
					file_list_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED, timeout=1)

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

def delete_files_parallel(paths, max_workers, verbose=False,files_per_job=1,exclude=None,batch=False,parallel_file_listing=False):
	global REMOVE_FILES_WHILE_LISTING
	if not batch and isinstance(paths, str):
		paths = [paths]
	all_files = set()
	init_size_all = 0
	removed_while_listing = 0
	for path in paths:
		file_list, links,init_size, _, removed_count = get_file_list(path, max_workers=max_workers, exclude=exclude, parallel_file_listing=parallel_file_listing,remove_files_while_listing=REMOVE_FILES_WHILE_LISTING,return_removed_count=True)
		all_files.update(set(file_list) | set(links))
		init_size_all += init_size
		removed_while_listing += removed_count
	# Entries unlinked during the scan never reach all_files, so count them separately.
	total_files = len(all_files) + removed_while_listing
	print(f"Number of files: {total_files}")
	print(f'Initial estimated size: {format_bytes(init_size_all)}B')
	if not all_files:
		# Nothing left to delete in parallel: either the tree was empty or the scan
		# already removed everything. Only the directory structure remains.
		return removed_while_listing + 1 , delete_file_bulk(paths, exclude=exclude)[0]
	delete_counter, delete_size_counter = delete_file_list_parallel(all_files, max_workers, verbose,files_per_job,init_size=init_size_all)
	print("Removing directory structures....")
	delete_size_counter += delete_file_bulk(paths, exclude=exclude)[0]
	print(f"Initial estimated size: {format_bytes(init_size_all)}B, Final size: {format_bytes(delete_size_counter)}B")
	return delete_counter + removed_while_listing + 1, delete_size_counter

#%% ---- Copy Files ----
def _chown_like_cp(path, st):
	"""Give path st's owner and group, as cp -a does: without privileges another user's ownership cannot be kept,
	so keep at least the group (one the user belongs to) and treat the rest as no error. Root failures raise.
	Returns (owner kept, group kept), as path has them now (it may have had the right owner already)."""
	try:
		os.chown(path, st.st_uid, st.st_gid)
		return True, True
	except PermissionError:
		if os.geteuid() == 0:
			raise
		try:
			os.chown(path, -1, st.st_gid)
		except PermissionError:
			pass
	now = os.stat(path)
	return now.st_uid == st.st_uid, now.st_gid == st.st_gid

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
	try:
		src_size = get_file_size(src_path)
		copiedSize = 0
		for dest in dest_paths:
			if os.path.exists(dest) and (not os.path.islink(src_path)) and is_file_identical(src_path, dest,os.path.getsize(src_path),full_hash):
				if not CONTENT_ONLY:
					st = os.stat(src_path,follow_symlinks=False)
					_remove_stale_xattrs(src_path, dest)
					shutil.copystat(src_path, dest,follow_symlinks=False)
					if os.name == 'posix':
						owner_kept, group_kept = _chown_like_cp(dest, st)
						#also copy the modes (as cp -a: no set-user-ID / set-group-ID bit for an owner / group not kept)
						os.chmod(dest, st.st_mode & ~((0 if owner_kept else stat.S_ISUID) | (0 if group_kept else stat.S_ISGID)))
					os.utime(dest, (st.st_atime, st.st_mtime),follow_symlinks=False)
				endTime = time.monotonic()
				return 0, endTime - start_time , symLinks #, task_to_run
		bak_dest_paths = dest_paths.copy()
		using_bak_paths = False
		if os.path.islink(src_path):
			symLinks[src_path] = dest_paths
		else:
			copied = False
			copy_error_start = len(ERRORS)
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
				try:
					try:
						# if the parent path for the file does not exist, create it
						if not NO_CREATE_DIR:
							os.makedirs(os.path.dirname(dest_path), exist_ok=True)
						if os.name == 'posix':
							cp_command = ["cp", "-f" if CONTENT_ONLY else "-af"]
							if cp_supports_sparse():
								cp_command.append("--sparse=always")
							run_command_in_multicmd_with_path_check(cp_command + [src_path, dest_path],quiet=True,strict=True)
							copiedSize = get_file_size(dest_path)
						else:
							if CONTENT_ONLY:
								shutil.copy(src_path, dest_path, follow_symlinks=False)
							else:
								shutil.copy2(src_path, dest_path, follow_symlinks=False)
							#shutil.copystat(src_path, dest_path)
						copied = True
						break
					except Exception as e:
						if os.path.exists(dest_path):
							os.remove(dest_path)
						if using_bak_paths:
							if not dest_paths and not verbose:
								import traceback
								print(f'Error copying {src_path} to {dest_path}, {e}')
								print(traceback.format_exc())
								print(f'Trying to copy from {src_path} to {dest_path} without sparse')
							if os.name == 'posix':
								cp_flags = "-f" if CONTENT_ONLY else "-af"
								run_command_in_multicmd_with_path_check(["cp", cp_flags, src_path, dest_path],quiet=True,strict=True)
								copiedSize = get_file_size(dest_path)
								#task_to_run = ["cp", "-af", src_path, dest_path]
							elif os.name == 'nt':
								run_command_in_multicmd_with_path_check(["xcopy", "/I", "/E", "/Y", "/c", "/q", "/k", "/r", "/h", "/x", src_path, dest_path],quiet=True,strict=True)
								#task_to_run = ["xcopy", "/I", "/E", "/Y", "/c", "/q", "/k", "/r", "/h", "/x", src_path, dest_path]
							copied = True
							break
						elif verbose:
							eprint(f'Retrying with a different destination path in {dest_paths}')
				except Exception as e:
					if using_bak_paths and not dest_paths:
						import traceback
						eprint(f'Copy failed: ERROR copying {src_path} to {dest_path}, No more destination paths to try: {e}')
						print(traceback.format_exc())
						return 0, time.monotonic() - start_time, symLinks #, task_to_run
					elif verbose:
						print(f'Re-Retrying with a different destination path in {dest_paths}')
					if os.path.exists(dest_path):
						os.remove(dest_path)
				if not copied and not using_bak_paths and not dest_paths:
					if verbose:
						print(f'Gracious copy pass failed. Force trying all destination paths for {src_path}')
					using_bak_paths = True
					dest_paths = bak_dest_paths
			if not copied:
				eprint(f'Copy failed: FAILED to copy {src_path} to {dest_paths}')
				return 0, time.monotonic() - start_time, symLinks
			# A successful retry resolves this file's failed attempts. Preserve
			# errors from earlier files and from path validation above.
			del ERRORS[copy_error_start:]
			if verbose:
				print(f'Copied {src_path} to {dest_path}')
				print(f'Estimated remaining size: {format_bytes(dest_free_space - copiedSize)}B')
	except Exception as e:
		eprint(f'Copy process crashed: Fatal Error copying {src_path} to {inDests}: {e}')
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
	file_list_iterator = iter(file_list)
	start_time = time.monotonic()
	lastRefreshTime = start_time
	total_files = len(file_list)
	futures = {}
	files_per_job = max(1,files_per_job)
	symLinks = {}
	for link in links:
		srcRelativePath = os.path.relpath(link, src_path)
		symLinks[link] = [os.path.join(dest_path,srcRelativePath) for dest_path in dest_paths]
	if len(file_list) == 0:
		return 0, 0, symLinks , frozenset()
	print(f"Processing {len(file_list)} files with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=total_files,total_size=estimated_size,last_num_job_for_stats=max(1,max_workers//10),process_word='Copied',suppress_all_output=verbose,bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	last_transition_time = time.monotonic()
	last_under_limit = True
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
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

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED,timeout=1)

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
	for src_path, dest_paths, file_list in jobs:
		for f in file_list:
			yield (src_path, dest_paths, f)

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
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
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

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED,timeout=1)

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

def copy_files_parallel_batch(jobs, max_workers, full_hash=False, verbose=False,files_per_job=1,parallel_file_listing=False,exclude=None,
							  exit_not_enough_space=False):
	newJobs = []
	total_symLinks = {}
	total_files = []
	total_item_count = 0
	total_size_count = 0
	total_link_count = 0
	total_folder_count = 0
	for src_path, dest_paths in jobs:
		dest_paths = _validate_paths(src_path, dest_paths)
		if dest_paths is None:
			continue
		if exclude and is_excluded(src_path,exclude):
			continue
		file_list , links,init_size,folders  = get_file_list(src_path, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)
		if exit_not_enough_space:
			# check if there is enough space on all dest paths
			dest_free_space = sum(get_free_space_bytes(dest_path) for dest_path in dest_paths)
			if dest_free_space < init_size:
				eprint(f'Not enough space: free space on {dest_paths} too small, skipping copy {src_path}')
				print(f'Free space: {format_bytes(dest_free_space)}B, Required: {format_bytes(init_size)}B')
				continue
		newJobs.append((src_path, dest_paths, file_list))
		total_item_count += len(file_list)
		total_size_count += init_size
		total_link_count += len(links)
		total_folder_count += len(folders)
		total_files.extend(file_list)
		for link in links:
			srcRelativePath = os.path.relpath(link, src_path)
			total_symLinks[link] = [os.path.join(dest_path,srcRelativePath) for dest_path in dest_paths]

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
	for file in file_list:
		srcRelativePath = os.path.relpath(file, src_path)
		size, cpTime ,rtnSymLinks = copy_file(file, [os.path.join(dest_path, srcRelativePath) for dest_path in dest_paths],full_hash = full_hash, verbose=verbose)
		#update_progress_bar(copy_counter, copy_size_counter, total_files, start_time)
		apb.update(num_files=1,cpSize=size,cpTime=cpTime,files_per_job=1)
		links.update(rtnSymLinks)
		apb.rate_limit()
	symLinks = {}
	for link in links:
		srcRelativePath = os.path.relpath(link, src_path)
		symLinks[link] = [os.path.join(dest_path, srcRelativePath) for dest_path in dest_paths]
	endTime = time.monotonic()
	apb.stop()
	_print_operation_summary(apb, endTime - start_time, show_bandwidth=True)
	return apb.item_counter, apb.size_counter , symLinks , frozenset(file_list)

def _link_dests(links, src_path, dest_paths):
	"""{link: [where it goes in each dest]} for the symlinks a listing of src_path found."""
	return {link: [os.path.join(dest_path, os.path.relpath(link, src_path)) for dest_path in dest_paths] for link in links}

def copy_files_serial_batch(jobs, full_hash=False, verbose=False,exclude=None,exit_not_enough_space=False):
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
		dest_paths = _validate_paths(src_path, dest_paths)
		if dest_paths is None:
			continue
		if exclude and is_excluded(src_path,exclude):
			continue
		if not os.path.isdir(src_path):
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
		if exit_not_enough_space:
			# check if there is enough space on all dest paths
			dest_free_space = sum(get_free_space_bytes(dest_path) for dest_path in dest_paths)
			if dest_free_space < init_size:
				eprint(f'Not enough space: free space on {dest_paths} too small, skipping copy {src_path}')
				print(f'Free space: {format_bytes(dest_free_space)}B, Required: {format_bytes(init_size)}B')
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
		for file in file_list:
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
			  parallel_file_listing=False,exclude=None, exit_not_enough_space=False):
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
		self.exit_not_enough_space = exit_not_enough_space
	def add_dir_sync(self, src, dests):
		self.dir_sync_job.append((src, dests))
	def add_copy(self, src, dests):
		self.copy_job.append((src, dests))
	def process(self):
		if self.dir_sync_job:
			start_time = time.monotonic()
			if self.max_workers == 1:
				self.total_sym_links.update(sync_directories_serial_batch(self.dir_sync_job,exclude=self.exclude))
			else:
				self.total_sym_links.update(sync_directories_parallel_batch(self.dir_sync_job, max_workers=self.max_workers,verbose=self.verbose,exclude=self.exclude,parallel_file_listing=self.parallel_file_listing))
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
				copy_counter, copy_size_counter , rtnSymLinks , file_list = copy_files_serial_batch(self.copy_job, full_hash = self.full_hash,verbose=self.verbose,exclude=self.exclude,exit_not_enough_space=self.exit_not_enough_space)
				#total_file_list.update(trim_paths(file_list,src))
				#self.total_file_list.update(trim_paths(rtnSymLinks.keys(),src))
			else:
				copy_counter, copy_size_counter , rtnSymLinks , file_list = copy_files_parallel_batch(self.copy_job, max_workers=self.max_workers,full_hash = self.full_hash,verbose=self.verbose,files_per_job=self.files_per_job,parallel_file_listing=self.parallel_file_listing,exclude=self.exclude,exit_not_enough_space=self.exit_not_enough_space)
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
		return self.total_file_list, self.total_sym_links
	def clear(self,clear_pending_jobs=False):
		self.copy_counter = 0
		self.copy_size_counter = 0
		self.total_sym_links = {}
		self.total_file_list = []
		if clear_pending_jobs:
			self.dir_sync_job = []
			self.copy_job = []


#%% ---- Copy Directories ----
def sync_directory_metadata(src_path, dest_paths):
	global CONTENT_ONLY
	dest_paths = _validate_paths(src_path, dest_paths)
	if dest_paths is None:
		return 0, 0 , set(), frozenset()
	start_time = time.monotonic()
	if os.path.islink(src_path):
		return 0,time.monotonic()-start_time,{src_path: dest_paths}
	if not os.path.lexists(src_path):
		# A live tree: the directory went away after it was listed. The destination keeps what it has.
		eprint(f'Directory sync error: {src_path} no longer exists')
		return 1,time.monotonic()-start_time,frozenset()
	if not os.path.isdir(src_path):
		return copy_file(src_path, dest_paths)
	# Create the directory if it does not exist
	try:
		st = os.stat(src_path)
	except OSError as e:
		# A live tree: the directory went away (or became unreadable) after it was listed.
		eprint(f'Directory sync error: {src_path}: {e}')
		return 1,time.monotonic()-start_time,frozenset()
	for dest_path in dest_paths:
		# One directory that cannot be created or synced is recorded; the others (and the other dests) go on.
		try:
			try:
				if not (os.path.exists(dest_path) or os.path.ismount(dest_path)):
					os.makedirs(dest_path, exist_ok=True)
			except FileExistsError:
				print(f"Destination path {dest_path} maybe a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...", file=sys.stderr)
			# Sync the metadata
			if not CONTENT_ONLY:
				_remove_stale_xattrs(src_path, dest_path)
				shutil.copystat(src_path, dest_path)
				if os.name == 'posix':
					owner_kept, group_kept = _chown_like_cp(dest_path, st)
					# copystat set the mode before the chown: no set-user-ID / set-group-ID bit for an owner / group not kept.
					lost = (0 if owner_kept else stat.S_ISUID) | (0 if group_kept else stat.S_ISGID)
					if st.st_mode & lost:
						os.chmod(dest_path, stat.S_IMODE(st.st_mode) & ~lost)
				os.utime(dest_path, (st.st_atime, st.st_mtime))
		except OSError as e:
			eprint(f'Directory sync error: {src_path} to {dest_path}: {e}')
	return 1,time.monotonic()-start_time,frozenset()

def sync_directory_metadata_bulk(src_paths, dest_paths,src_path):
	total_count = 0
	total_time = 0
	symLinks = {}
	for src in src_paths:
		source_relative_path = os.path.relpath(src, src_path)
		dests = [os.path.join(dest_path, source_relative_path) for dest_path in dest_paths]
		count , cpTime , rtnSymLinks = sync_directory_metadata(src, dests)
		total_count += count
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_count , total_time, symLinks

def sync_directory_metadata_bulk_batch(jobs):
	total_count = 0
	total_time = 0
	symLinks = {}
	for (src_path,folder,dest_paths) in jobs:
		source_relative_path = os.path.relpath(folder, src_path)
		dests = [os.path.join(dest_path, source_relative_path) for dest_path in dest_paths]
		count , cpTime , rtnSymLinks = sync_directory_metadata(folder, dests)
		total_count += count
		total_time += cpTime
		symLinks.update(rtnSymLinks)
	return total_count , total_time, symLinks

def sync_directories_parallel(src, dests, max_workers, verbose=False,folder_per_job=64,exclude=None,parallel_file_listing=False):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.1
	dests = _validate_paths(src, dests)
	if dests is None:
		return 0, 0, set(), frozenset()
	symLinks = {}
	if exclude and is_excluded(src,exclude):
		return 0, 0 , set(), frozenset()
	# 
	if not os.path.isdir(src):
		_, _ , symLinks = copy_file(src, dests)
		return symLinks
	sync_directory_metadata(src, dests)
	_,_,_,folders  = get_file_list(src, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)
	folder_list_iterator = iter(folders)
	futures = {}
	start_time = time.monotonic()
	last_refresh_time = start_time
	max_workers = max(2,int(max_workers / 4))
	folder_per_job = max(1,folder_per_job)
	num_folders_copied_this_job = 0

	print(f"Syncing Dir from {src} to {dests} with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=len(folders),total_size=len(folders),suppress_all_output=verbose,process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		while folder_list_iterator or futures:
			# counter = 0
			while folder_list_iterator and len(futures) <  max_scheduled_jobs and apb.under_rate_limit():
				src_folders = []
				try:
					for _ in range(folder_per_job):
						src_folder = next(folder_list_iterator)
						src_folders.append(src_folder)
					future = _submit_task(executor, sync_directory_metadata_bulk, src_folders, dests,src)
					futures[future] = src_folders
					# counter += 1
				except StopIteration:
					if src_folders:
						future = _submit_task(executor, sync_directory_metadata_bulk, src_folders, dests,src)
						futures[future] = src_folders
					folder_list_iterator = None

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED,timeout=1)
			time_spent_this_iter = 0
			for future in done:
				src_folders = futures.pop(future)
				num_folders_copied_this_job = len(src_folders)
				cpSize, cpTime, rtnSymLinks = _task_result(future, (0, 0, {}), 'Copy process crashed')
				time_spent_this_iter += cpTime
				apb.update(num_files=num_folders_copied_this_job, cpSize=cpSize, cpTime=cpTime , files_per_job=folder_per_job)
				symLinks.update(rtnSymLinks)
			apb.scheduled_jobs = len(futures)
			apb.print_progress()
			if done:
				if verbose:
						print(f'\nAverage cptime is {time_spent_this_iter / len(done):0.2f} for {len(done)} jobs with {num_folders_copied_this_job} folders each')

				if folder_list_iterator and num_folders_copied_this_job == folder_per_job and (time_spent_this_iter / len(done) > 5 or time.monotonic() - last_refresh_time > 5):
					folder_per_job //= MAGIC_NUMBER
					folder_per_job = round(folder_per_job)
					if verbose:
						print(f'\nCompletion time is long, changing folders per job to {folder_per_job}')
				elif folder_list_iterator and num_folders_copied_this_job == folder_per_job and time_spent_this_iter / len(done) < 1:
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
	return symLinks

def sync_directories_parallel_batch(jobs, max_workers, verbose=False,folder_per_job=64,exclude=None,parallel_file_listing=False):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	if FILES_RATE_LIMIT or BYTES_RATE_LIMIT:
		max_scheduled_jobs = max_workers
	else:
		max_scheduled_jobs = max_workers * 1.1
	symLinks = {}
	allFolderToSync = []
	for src, dests in jobs:
		dests = _validate_paths(src, dests)
		if dests is None:
			continue
		if exclude and is_excluded(src,exclude):
			continue
		if not os.path.isdir(src):
			_, _ , sl = copy_file(src, dests)
			symLinks.update(sl)
			continue
		sync_directory_metadata(src, dests)
		_,_,_,folders  = get_file_list(src, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)
		allFolderToSync.extend([(src,folder,dests) for folder in folders])
	if not allFolderToSync:
		return symLinks
	print(f"Syncing Dir for {len(allFolderToSync)} folders with {max_workers} workers")
	apb = Adaptive_Progress_Bar(total_count=len(allFolderToSync),total_size=len(allFolderToSync),suppress_all_output=verbose,process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	job_iterator = iter(allFolderToSync)
	futures = {}
	start_time = time.monotonic()
	last_refresh_time = start_time
	max_workers = max(2,int(max_workers / 4))
	folder_per_job = max(1,folder_per_job)
	num_folders_copied_this_job = 0
	
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
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

			done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED,timeout=1)
			time_spent_this_iter = 0
			for future in done:
				currentJobs = futures.pop(future)
				num_folders_copied_this_job = len(currentJobs)
				cpSize, cpTime, rtnSymLinks = _task_result(future, (0, 0, {}), 'Copy process crashed')
				time_spent_this_iter += cpTime
				apb.update(num_files=num_folders_copied_this_job, cpSize=cpSize, cpTime=cpTime , files_per_job=folder_per_job)
				symLinks.update(rtnSymLinks)
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
	return symLinks

def sync_directories_serial(src, dests,exclude=None):
	"""
	Synchronizes directories from source to destination in a single thread.

	Args:
		src (str): The source directory path.
		dest (str): The destination directory path.
		exclude (list, optional): A list of patterns to exclude from synchronization.

	Returns:
		tuple: A tuple containing:
			- int: Number of directories synced (always 0 in this implementation).
			- int: Total size of directories synced in bytes (always 0 in this implementation).
			- set: Set of synchronized directories (always empty in this implementation).
			- frozenset: Set of symbolic links (always empty in this implementation).

	Notes:
		- If the source or destination path length exceeds 4096 characters, the function will skip synchronization.
		- If the source path is excluded based on the exclude patterns, the function will skip synchronization.
		- The function prints progress information to the terminal, including the number of directories synced and the speed of synchronization.
	"""
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	dests = _validate_paths(src, dests)
	if dests is None:
		return 0, 0, set(), frozenset()
	if exclude and is_excluded(src,exclude):
		return 0, 0 , set(), frozenset()
	symLinks = {}
	if not os.path.isdir(src):
		_, _ , symLinks = copy_file(src, dests)
		return symLinks
	#sync_directory_metadata(src, dest)
	_,_,_,folders = get_file_list(src,exclude=exclude,parallel_file_listing=False)
	print(f"Syncing Dir from {src} to {dests} in single thread")
	apb = Adaptive_Progress_Bar(total_count=len(folders),total_size=len(folders),process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	for folder in folders:
		source_relative_path = os.path.relpath(folder, src)
		count , cpTime , _ = sync_directory_metadata(folder, [os.path.join(dest, source_relative_path) for dest in dests])
		apb.update(num_files=1, cpSize=count, cpTime=cpTime , files_per_job=1)
		apb.rate_limit()
	apb.stop()
	return symLinks

def sync_directories_serial_batch(jobs,exclude=None):
	global FILES_RATE_LIMIT
	global BYTES_RATE_LIMIT
	newJobs = []
	# skip if src path or dest path is longer than 4096 characters
	symLinks = {}
	totalFolderCount = 0
	srcFolders = []
	for src,dests in jobs:
		dests = _validate_paths(src, dests)
		if dests is None:
			continue
		if exclude and is_excluded(src,exclude):
			continue
		if not os.path.isdir(src):
			_, _ , sls = copy_file(src, dests)
			symLinks.update(sls)
			continue
		newJobs.append((src,dests))
		_,_,_,folders = get_file_list(src,exclude=exclude,parallel_file_listing=False)
		srcFolders.append(folders)
		totalFolderCount += len(folders)
	if totalFolderCount == 0:
		return symLinks
	print(f"Syncing Dir from {len(newJobs)} in single thread")
	apb = Adaptive_Progress_Bar(total_count=totalFolderCount,total_size=totalFolderCount,process_word='Synced',bytes_rate_limit=BYTES_RATE_LIMIT,files_rate_limit=FILES_RATE_LIMIT)
	for (src, dests), folders in zip(newJobs, srcFolders):
		for folder in folders:
			source_relative_path = os.path.relpath(folder, src)
			count , cpTime , _ = sync_directory_metadata(folder, [os.path.join(dest, source_relative_path) for dest in dests])
			apb.update(num_files=1, cpSize=count, cpTime=cpTime , files_per_job=1)
			apb.rate_limit()
	apb.stop()
	return symLinks

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

def _mtime_ns_close(a_ns, b_ns):
	"""Same tolerance as is_file_identical(): 2 x the coarser timestamp precision."""
	precision = max(get_timestamp_precision(a_ns / 1e9), get_timestamp_precision(b_ns / 1e9))
	return abs(a_ns - b_ns) <= precision * 2 * 1e9

_TZ_STEP_NS = 15 * 60 * 10**9
_TZ_MAX_NS = 24 * 3600 * 10**9

def _tz_shift_close(a_ns, b_ns):
	"""True when a_ns and b_ns are a whole number of 15-minute steps apart (at most 24 hours), within the usual
	timestamp tolerance: one FAT timestamp read under two timezone settings (sys_tz, tz=, time_offset=)."""
	shift = int(round((a_ns - b_ns) / _TZ_STEP_NS)) * _TZ_STEP_NS
	return 0 < abs(shift) <= _TZ_MAX_NS and _mtime_ns_close(a_ns, b_ns + shift)

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
	if _mtime_ns_close(info_a['mtime_ns'], info_b['mtime_ns']):
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

	shutil.copystat() copies and overwrites xattrs but never removes one. Non-portable,
	filesystem-internal names are never removed. Skipped under CONTENT_ONLY; unsupported,
	forbidden or already-gone removals are ignored.
	"""
	if CONTENT_ONLY or not hasattr(os, 'listxattr'):
		return
	try:
		src_names = set(os.listxattr(src, follow_symlinks=False))
		dest_names = os.listxattr(dest, follow_symlinks=False)
	except OSError:
		return
	for name in dest_names:
		if name in src_names or not _is_portable_xattr(name):
			continue
		try:
			os.removexattr(dest, name, follow_symlinks=False)
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

def write_changes(target, rows):
	"""Write rows to target in changes.tsv format (also the -dfl output format)."""
	with open(target, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
		f.write(f'{CHANGES_HEADER}\t{CHANGES_VERSION}\n')
		f.write('#op\tpath\told\tnew\n')
		for row in sort_change_rows(rows):
			f.write(f'{row.op}\t{_escape_field(row.path)}\t{row.old}\t{row.new}\n')

def read_changes(source):
	"""Read a changes.tsv file into ChangeRows. Raises RuntimeError('Patch format error: ...')."""
	rows = []
	with open(source, 'r', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
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
	with open(target, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
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
	with open(source, 'r', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
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
	files, links, _, folders = get_file_list(root, exclude=None if post_filter else exclude, max_workers=max_workers,
		drop_cache=True, parallel_file_listing=parallel_file_listing)
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
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = []
		for index in range(0, len(paths), chunk_size):
			chunk = paths[index:index + chunk_size]
			futures.append(executor.submit(_states_chunk, chunk, full_hash, read_xattrs, HASH_SIZE,
				frozenset(path for path in chunk if path in marked),
				frozenset(path for path in chunk if path in stat_paths) if stats is not None else None))
		for future in concurrent.futures.as_completed(futures):
			chunk_states, chunk_errors, chunk_stats = future.result()
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
			with open(diff_file_list, 'w', encoding='utf-8', errors='surrogateescape', newline='\n') as f:
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

def _parse_file_list_lines(lines, live_keys=()):
	"""Parse -fl style lines into {key: value}.

	value is the hash for files, the link target for symlinks, None when the list holds
	neither. A line ending in '/' is a directory. A line ending in ':<digest>' is a file, whatever
	algorithm made the digest, and in a hashed list so is 'path:' (made with -hs 0); only a digest of
	this system's algorithm is kept as the value, another one or none is None (not compared).
	A list is hashed when a line has a digest, or when every file and link line has a ':' (a -hs 0
	list; an unhashed list has plain paths). In a hashed list the remaining lines are symlinks
	'path:target': the longest live key followed by ':' is the path, else the text before the first ':'.
	"""
	hex_re = _hash_hex_re()
	lines = [line for line in lines if line]
	entry_lines = [line for line in lines if not line.endswith('/')]
	hashed = (any(':' in line and _LIST_DIGEST_RE.fullmatch(line.rpartition(':')[2]) for line in entry_lines)
		or (bool(entry_lines) and all(':' in line for line in entry_lines)))
	live_keys = set(live_keys)
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
			shutil.rmtree(path)
		else:
			os.unlink(path)
	except FileNotFoundError:
		pass

def _place_symlink(target, dest):
	if os.path.lexists(dest):
		_remove_entry(dest)
	os.makedirs(os.path.dirname(dest), exist_ok=True)
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
		os.makedirs(diff_dir, exist_ok=True)
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
			os.makedirs(_key_path(volume_root, key), exist_ok=True)
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
		rows = read_changes(os.path.join(ordered[0][0], DIFF_DIR_NAME, changes_file))
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

def _discard_temp(path):
	"""Remove a temporary entry that was not renamed into place; a missing one is fine."""
	try:
		os.unlink(path)
	except FileNotFoundError:
		pass
	except OSError as e:
		print(f"Warning: cannot remove temporary file {path}: {e}", file=sys.stderr)

def _fat_ioctl_numbers(machine=None):
	"""(FAT_IOCTL_GET_ATTRIBUTES, FAT_IOCTL_SET_ATTRIBUTES) for machine (platform.machine() by default):
	_IOR('r', 0x10, __u32) and _IOW('r', 0x11, __u32). alpha, mips, powerpc and sparc put _IOC_READ (2) and
	_IOC_WRITE (4) at bit 29, and parisc swaps the generic read and write bits: on all of them the two numbers come
	out the other way round."""
	machine = (platform.machine() if machine is None else machine).lower()
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
	temporary file never outlives this call.
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
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		futures = {_submit_task(executor, copy, source, [tmp], full_hash, verbose): (row, source, dest, tmp)
			for row, source, dest, tmp in jobs}
		for future in concurrent.futures.as_completed(futures):
			_finish_replace(futures[future], target_root, functools.partial(_raise_task_failures, future), metadata)

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

def _remove_tree_by_path(top, top_dev, mount_points):
	"""The path-based walk of _remove_tree_around_mounts, for platforms without directory descriptors (open to a
	swap of a directory for a symlink between the check and the use). top is a real path; returns (kept, errors)."""
	kept, errors = [], []
	# (directory, its entries already handled): a directory is pushed again after its children, then removed.
	stack = [(top, False)]
	while stack:
		directory, done = stack.pop()
		if done:
			try:
				os.rmdir(directory)
			except FileNotFoundError:
				pass
			except OSError as e:
				# Not empty: a kept mount below it, or an entry whose removal already failed (recorded).
				if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
					errors.append(f'{os.path.relpath(directory, top)}: {e.strerror}')
			continue
		stack.append((directory, True))
		try:
			names = os.listdir(directory)
		except OSError as e:
			errors.append(f'{os.path.relpath(directory, top)}: {e.strerror}')
			continue
		for name in names:
			child = os.path.join(directory, name)
			try:
				info = os.lstat(child)
			except FileNotFoundError:
				continue
			except OSError as e:
				errors.append(f'{os.path.relpath(child, top)}: {e.strerror}')
				continue
			if child in mount_points or (stat.S_ISDIR(info.st_mode) and info.st_dev != top_dev):
				kept.append(os.path.relpath(child, top))
			elif stat.S_ISDIR(info.st_mode):
				stack.append((child, False))
			else:
				try:
					os.unlink(child)
				except FileNotFoundError:
					pass
				except OSError as e:
					errors.append(f'{os.path.relpath(child, top)}: {e.strerror}')
	return sorted(kept), errors

def _open_below(root_real, rel, flags):
	"""A descriptor of root_real/rel reached one component at a time from a descriptor of root_real, each opened with
	flags (O_NOFOLLOW | O_DIRECTORY): a component that is a symlink or no directory raises OSError (ELOOP, ENOTDIR)."""
	fd = os.open(root_real, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_CLOEXEC', 0))
	try:
		for part in ([] if rel in ('', '.') else rel.split(os.sep)):
			next_fd = os.open(part, flags, dir_fd=fd)
			os.close(fd)
			fd = next_fd
	except BaseException:
		os.close(fd)
		raise
	return fd

def _remove_tree_around_mounts(path, mount_points=None, target_root=None):
	"""Remove the directory tree at path as shutil.rmtree would, without entering a mount.

	An entry that is a mount point in mountinfo (any entry: bind mounts and hidden mounts too), or a directory whose
	st_dev differs from path's (a mount made after mountinfo was read, or a btrfs subvolume), is left in place with
	everything below it, and so are the directories above it; path itself too when it is a mount point in mountinfo or
	its st_dev differs from its parent's. Symlinks are unlinked, never
	followed, and that holds for path itself: only its parent is resolved, so a path that is a symlink (or not a
	directory) removes nothing and returns an error. The walk is iterative and keeps going after an error. Returns (the entries left in place as mount
	points, relative to path; the errors met, as text).

	The walk goes by directory descriptors, as shutil.rmtree does: every directory is opened with O_NOFOLLOW
	relative to its parent (path itself included) and checked against the lstat taken before descending, and entries are read and removed
	relative to the descriptor, so a directory swapped for a symlink during the walk is refused, never followed
	(this runs as root on trees other users may write to). Where the platform lacks the descriptor functions it
	falls back to a walk by path, which does not close that race.

	target_root: the tree the removal must stay in (the apply's target; '/' when None). The top's parent is resolved once and must lie inside it; with the descriptor functions it is then reached from a descriptor of target_root one component at a time with O_NOFOLLOW, and the top is examined, entered and removed relative to its parent's descriptor. A directory above the top swapped for a symlink after the apply checked the path therefore refuses the removal instead of acting outside the target (this runs as root on trees other users may write to).
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
	if not _DIR_FD_WALK:
		try:
			top_info = os.lstat(top)
		except FileNotFoundError:
			return [], []
		except OSError as e:
			return [], [f'.: {e.strerror}']
		if not stat.S_ISDIR(top_info.st_mode):
			return [], ['.: not a directory (a symlink?)']
		if top in mount_points:
			return ['.'], []
		try:
			parent_dev = os.lstat(parent_real).st_dev
		except OSError as e:
			return [], [f'..: {e.strerror}']
		if parent_dev != top_info.st_dev:
			return ['.'], []
		return _remove_tree_by_path(top, top_info.st_dev, mount_points)
	kept, errors = [], []
	flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0)
	try:
		parent_fd = _open_below(root_real, os.path.relpath(parent_real, root_real), flags)
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
		def open_dir(name, parent_fd, rel, expected):
			"""A descriptor for the directory, or None (an error recorded unless it vanished)."""
			try:
				fd = os.open(name, flags, dir_fd=parent_fd)
			except FileNotFoundError:
				return None
			except OSError as e:
				errors.append(f'{rel}: {e.strerror}')
				return None
			try:
				info = os.fstat(fd)
				if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
					return fd
				errors.append(f'{rel}: changed while being removed; not entered')
			except OSError as e:
				errors.append(f'{rel}: {e.strerror}')
			os.close(fd)
			return None

		# One frame per open directory: [fd, name in its parent, real path, subdirectories still to enter].
		stack = []

		def enter(name, parent_fd, real, expected):
			fd = open_dir(name, parent_fd, os.path.relpath(real, top), expected)
			if fd is None:
				return
			frame = [fd, name, real, []]
			stack.append(frame)
			try:
				names = os.listdir(fd)
			except OSError as e:
				errors.append(f'{os.path.relpath(real, top)}: {e.strerror}')
				return
			for child_name in names:
				child = os.path.join(real, child_name)
				rel = os.path.relpath(child, top)
				try:
					info = os.stat(child_name, dir_fd=fd, follow_symlinks=False)
				except FileNotFoundError:
					continue
				except OSError as e:
					errors.append(f'{rel}: {e.strerror}')
					continue
				if child in mount_points or (stat.S_ISDIR(info.st_mode) and info.st_dev != top_dev):
					kept.append(rel)
				elif stat.S_ISDIR(info.st_mode):
					frame[3].append((child_name, child, info))
				else:
					try:
						os.unlink(child_name, dir_fd=fd)
					except FileNotFoundError:
						pass
					except OSError as e:
						errors.append(f'{rel}: {e.strerror}')

		try:
			enter(name, parent_fd, top, top_info)
			while stack:
				frame = stack[-1]
				if frame[3]:
					child_name, child, info = frame[3].pop()
					enter(child_name, frame[0], child, info)
					continue
				stack.pop()
				os.close(frame[0])
				try:
					if stack:
						os.rmdir(frame[1], dir_fd=stack[-1][0])
					else:
						os.rmdir(name, dir_fd=parent_fd)
				except FileNotFoundError:
					pass
				except OSError as e:
					# Not empty: a kept mount below it, or an entry whose removal already failed (recorded).
					if e.errno not in (errno.ENOTEMPTY, errno.EEXIST):
						errors.append(f'{os.path.relpath(frame[2], top)}: {e.strerror}')
		finally:
			for frame in stack:
				os.close(frame[0])
		return sorted(kept), errors
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
			os.makedirs(path, mode=0o700 if metadata else 0o777, exist_ok=True)
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
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-s', 'TYPE', '-o', 'value', device], quiet=True)
	except Exception:
		output = []
	finally:
		del ERRORS[error_count:]
	return next((line.strip() for line in output if line.strip()), '')

def _ro_mount_options(device):
	"""Read-only mount options that also work for cloned (duplicate-UUID) xfs and dirty ext3/4 journals."""
	return {'xfs': 'ro,nouuid,norecovery', 'ext3': 'ro,noload', 'ext4': 'ro,noload'}.get(_probe_fs_type(device), 'ro')

def _rw_mount_command(device, mount_point):
	"""The read-write mount command for device: a clone of a mounted xfs (a golden image, an image of a live node's
	disk) carries the same UUID, which xfs refuses to mount twice without nouuid."""
	if _probe_fs_type(device) == 'xfs':
		return ['mount', '-o', 'nouuid', device, mount_point]
	return ['mount', device, mount_point]

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
	return apply_patch_tree(volume_roots, target_root, force=force, max_workers=max_workers, verbose=verbose,
		volume_names=volume_names, store_note=store_note)

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
	"""Wait until udev has handled its queued events (quietly; nothing when udevadm is missing)."""
	if check_path('udevadm'):
		error_count = len(ERRORS)
		try:
			run_command_in_multicmd_with_path_check(['udevadm', 'settle'], quiet=True)
		finally:
			del ERRORS[error_count:]

def _settle_partitions(disk, numbers, timeout=10, settle=True):
	"""{number: device} once udev is idle and every partition in numbers has its device node.

	udev re-reads a whole disk's partition table after the disk is closed from writing (a new
	loop device, an MBR or GPT write), removing and re-adding the partition nodes meanwhile.
	settle False skips the udev settle (the caller has just settled).
	Raises RuntimeError('Disk layout mismatch: ...') when a node is still missing after timeout seconds.
	"""
	if settle:
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
	"""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'udev', device], quiet=True)
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
	if len(label.encode('utf-8', 'surrogateescape')) >= _UDEV_ENC_CUT_BYTES or _UDEV_PARTIAL_ESCAPE_RE.search(label):
		raw_label = _probe_raw_label(device)
		label = _udev_decode(label) if raw_label is None else raw_label
	else:
		label = _udev_decode(label)
	return values.get('ID_FS_TYPE', ''), _udev_decode(uuid), label

def _probe_raw_label(device):
	"""device's label from `blkid -p -o value -s LABEL`, which prints it raw (None when it prints nothing).

	multiCMD writes each output byte that is not UTF-8 as \\xNN (0x80 and above); those become the bytes again, kept
	with surrogateescape. A backslash in the label itself is printed as it is and stays: _udev_decode would read a
	label's own '\\x41' as 'A'.
	"""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'value', '-s', 'LABEL', device], quiet=True)
	finally:
		del ERRORS[error_count:]
	if not output:
		return None
	data = output[0].rstrip('\r\n').encode('utf-8', 'surrogateescape')
	data = _BACKSLASHREPLACE_RE.sub(lambda match: bytes([int(match.group(1), 16)]), data)
	return data.decode('utf-8', 'surrogateescape')

def _partition_table_type(device):
	"""Partition table type of device from a low-level `blkid -p` probe ('gpt', 'dos', ...; '' when it finds
	none, which includes an ambivalent probe: see _no_partition_table)."""
	error_count = len(ERRORS)
	try:
		output = run_command_in_multicmd_with_path_check(['blkid', '-p', '-o', 'value', '-s', 'PTTYPE', device], quiet=True)
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
	udev is settled once, the GPT is read with sgdisk alone, then each partition is probed once on its settled
	device (_probe_identity); fs_params False skips the mkfs-parameter probes ('fs_params' is then '': a verify read
	compares identity values only).
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
	# A node udev is still re-adding would be probed as empty: settle once, then wait for every node.
	# sgdisk --print only reads the disk, so it triggers no udev re-read: this one settle before it, plus the node wait
	# after it, is enough.
	_udev_settle()
	infos = get_partition_infos(device, identity=False)
	devices = _settle_partitions(device, [int(key) for key, info in infos.items() if 'disk_identifier' not in info], settle=False)
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

def _fs_identity_commands(fs_type, device, uuid=None, label=None):
	"""argv lists that set a filesystem's UUID and / or label on an unmounted device.

	uuid / label None leave the value alone; label '' clears it. Returns None when fs_type
	cannot have a requested value set without recreating it. dd mode's delayed UUID
	commands (write_partition_info) come from this table too.
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
			commands.append(['fatlabel', '-i', device, uuid.replace('-', '').upper()])
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
		os.makedirs(diff_dir, exist_ok=True)
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
			os.makedirs(os.path.dirname(dest), exist_ok=True)
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
			os.makedirs(root, exist_ok=True)
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
		output = run_command_in_multicmd_with_path_check(['losetup', '-j', image], quiet=True)
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
	layout = read_disk_layout(disk, name=target, pttype=pttype)
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
		for plan in order:
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
def remove_extra_dirs(src_paths, dests,exclude=None,dest_image=False,keep=()):
	"""
	Removes extra directories in destination that are not present in the source paths.

	A destination directory is kept when its source counterpart exists: dirname(src) + its path below dest, or with
	dest_image (process_copy's -di layout: each source at dest + its absolute path) when '/' + its path below dest is
	a source, a parent of one, or below one and still there.
	Excluded directories and mount points, with everything below them, are left alone, and so are symlinks and the
	link targets create_sym_links tracked (keep, see _kept_paths); a directory that still holds entries (excluded
	files, say) is kept.

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
	kept = _kept_paths(keep, mapping)
	# leave out excluded source paths (a filtered copy: the caller's list is not touched)
	src_paths = [src_path for src_path in src_paths if not (exclude and is_excluded(src_path,exclude))]
	source_roots = [os.path.abspath(src_path) for src_path in src_paths]
	extraDirs = set()
	print(pformat(dests))
	for dest in dests:
		for dirpath, dirnames, _ in os.walk(dest):
			walk_into = []
			for dirname in dirnames:
				# No trailing separator: excludes match a directory as the listing does ('*/cache').
				dest_dir_path = os.path.join(dirpath, dirname)
				# The walk already pruned excluded parents: only the directory's own source path is left to check.
				if exclude and (is_excluded(dest_dir_path,exclude) or _excluded_at_source(os.path.abspath(dest_dir_path), mapping, exclude, parents_checked=True)):
					print(f"Skipping excluded directory: {dest_dir_path}")
					continue
				if os.path.islink(dest_dir_path):
					continue
				if os.path.ismount(dest_dir_path):
					# Another filesystem: never walked into, so nothing in it is removed.
					print(f"Skipping mount point: {dest_dir_path}")
					continue
				walk_into.append(dirname)
				relative_path = os.path.relpath(dest_dir_path, dest)
				# Check if the directory exists in the source paths
				if dest_image:
					# The image mirrors this run's sources at their absolute paths: a directory is theirs when it lies on
					# the way to one (a parent process_copy created) or below one and still exists there.
					live_path = os.path.join(os.path.sep, relative_path)
					in_source = any(source == live_path or source.startswith(live_path.rstrip(os.sep) + os.sep)
						or (live_path.startswith(source.rstrip(os.sep) + os.sep) and os.path.exists(live_path)) for source in source_roots)
				else:
					in_source = any(os.path.exists(os.path.join(os.path.dirname(src_path), relative_path)) for src_path in src_paths)
				if not in_source and not _is_kept(os.path.abspath(dest_dir_path), kept, os.path.abspath(dest)):
					print(f"Deleting extra directory: {dest_dir_path}")
					extraDirs.add(dest_dir_path)
			# Excluded trees and other filesystems are never walked into.
			dirnames[:] = walk_into
	for dir in sorted(extraDirs, key=lambda path: len(pathlib.Path(path).parts), reverse=True):
		try:
			os.rmdir(dir)
		except OSError as e:
			if e.errno in (errno.ENOTEMPTY, errno.EEXIST):
				print(f"Keeping non-empty directory: {dir}")
			else:
				eprint(f'Remove directory failed: {dir}: {e}')

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

	The list is sorted by ``src_abs`` length (longest first) so that nested
	srcs are attributed to the most specific entry.
	"""
	mapping = []
	for src in src_paths or []:
		if not src:
			continue
		src_abs = os.path.abspath(src)
		# basename respects trailing separators: 'src/' -> '', 'src' -> 'src'
		sourceFolderName = os.path.basename(src)
		actual_dests = []
		for dest in dests:
			if dest_image:
				actual_dests.append(os.path.abspath(dest + src_abs))
			elif os.path.basename(dest) == '' and sourceFolderName:
				actual_dests.append(os.path.abspath(os.path.join(dest, sourceFolderName)))
			else:
				actual_dests.append(os.path.abspath(dest))
		mapping.append((src_abs, actual_dests))
	mapping.sort(key=lambda item: len(item[0]), reverse=True)
	return mapping

def _excluded_at_source(dest_path, mapping, exclude, parents_checked=False):
	"""True when dest_path is the copy of a source path that exclude leaves out: that path, or a directory above it
	below its source root (parents_checked: the caller already pruned those), or the source root itself as the copy
	listing names it ('path/' for a directory, so '*/cache/' matches it and '*/cache' does not). -e names source
	paths: an absolute pattern never matches the destination side itself. mapping: _src_to_dest_map's result."""
	if not exclude:
		return False
	for src_abs, actual_dests in mapping:
		for actual_dest in actual_dests:
			if dest_path != actual_dest and not dest_path.startswith(actual_dest.rstrip(os.sep) + os.sep):
				continue
			listed_root = src_abs.rstrip(os.sep) + os.sep if os.path.isdir(src_abs) else src_abs
			if is_excluded(listed_root, exclude):
				return True
			source = os.path.normpath(os.path.join(src_abs, os.path.relpath(dest_path, actual_dest)))
			parts = (source,) if parents_checked else _path_and_parents(source, stop=src_abs)
			if source != src_abs and any(is_excluded(part, exclude) for part in parts):
				return True
	return False

def _kept_paths(keep, mapping):
	"""The link targets -rme keeps, from create_sym_links' (destination path, source path) pairs: a target that holds a
	copied tree (a link to '..') is left out, or it would keep everything; so is one inside a copied tree that is that
	tree's copy of the same source path (judged like any other); one that came from another source is kept."""
	kept = set()
	for dest_target, source_target in keep:
		dest_target = os.path.abspath(dest_target)
		contains_a_copy = False
		covered = False
		for src_abs, actual_dests in mapping:
			for actual_dest in actual_dests:
				if actual_dest == dest_target or actual_dest.startswith(dest_target.rstrip(os.sep) + os.sep):
					contains_a_copy = True
				elif dest_target.startswith(actual_dest.rstrip(os.sep) + os.sep):
					mapped = os.path.normpath(os.path.join(src_abs, os.path.relpath(dest_target, actual_dest)))
					covered = covered or mapped == os.path.normpath(os.path.abspath(source_target))
		if not contains_a_copy and not covered:
			kept.add(dest_target)
	return kept

def _is_kept(path, kept, root):
	"""True when path, or a directory above it below root, is one of the kept paths (_kept_paths)."""
	return bool(kept) and any(part in kept for part in _path_and_parents(path, stop=root.rstrip(os.sep) or os.sep))

def _expected_dest_paths(src_entries, src_paths, dests, dest_image=False, mapping=None):
	"""Compute the set of absolute dest paths each src entry should occupy (dest_image: see _src_to_dest_map;
	mapping: its result, when the caller has it)."""
	expected = set()
	if not src_entries:
		return expected
	if mapping is None:
		mapping = _src_to_dest_map(src_paths, dests, dest_image)
	for entry in src_entries:
		if not entry or not os.path.isabs(entry):
			continue
		entry_abs = os.path.abspath(entry)
		for src_abs, actual_dests in mapping:
			if entry_abs == src_abs:
				expected.update(actual_dests)
				break
			if entry_abs.startswith(src_abs + os.path.sep):
				rel = os.path.relpath(entry_abs, src_abs)
				for actual_dest in actual_dests:
					expected.add(os.path.join(actual_dest, rel))
				break
	return expected

def remove_extra_files(total_file_list, dests,max_workers,verbose,files_per_job,single_thread=False,exclude=None,parallel_file_listing=False,src_paths=None,sym_links=None,dest_image=False,keep=()):
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
		create_sym_links tracked (see _kept_paths).
		"""
		print(f"Removing extra files from {dests} with {max_workers} workers")
		src_entries = set(total_file_list) if total_file_list else set()
		if sym_links:
			src_entries.update(sym_links.keys() if hasattr(sym_links, 'keys') else sym_links)
		inDestNotInSrc = set()
		if src_paths:
			mapping = _src_to_dest_map(src_paths, dests, dest_image)
			expected = _expected_dest_paths(src_entries, src_paths, dests, dest_image, mapping)
			kept = _kept_paths(keep, mapping)
			for dest in dests:
				dest_file_list, links, _, _ = get_file_list(dest, max_workers=max_workers, exclude=exclude, parallel_file_listing=parallel_file_listing)
				dest_abs_root = os.path.abspath(dest)
				actual = {os.path.abspath(p) for p in dest_file_list}
				actual.update({os.path.abspath(p) for p in links})
				# Only consider entries that actually live under this dest so we
				# never accidentally point os.remove outside of the destination.
				for path in actual - expected:
					if path == dest_abs_root or path.startswith(dest_abs_root + os.path.sep):
						# The copy of an excluded source path is not extra: it was left out, not deleted.
						if not _excluded_at_source(path, mapping, exclude) and not _is_kept(path, kept, dest_abs_root):
							inDestNotInSrc.add(path)
		else:
			for dest in dests:
				dest_file_list,links,_ ,_ = get_file_list(dest, max_workers=max_workers,exclude=exclude,parallel_file_listing=parallel_file_listing)
				dest_trimmed = trim_paths(dest_file_list,dest)
				dest_trimmed.update(trim_paths(links,dest))
				# trim_paths roots everything at os.path.dirname(dest); rebuild
				# absolute paths against the same base or we'd double up the
				# dest basename when ``dest`` lacks a trailing separator.
				dest_base = os.path.dirname(dest) or os.path.sep
				inDestNotInSrc.update([os.path.join(dest_base,file) for file in (dest_trimmed - src_entries)])
		print('-'*80)
		print("Files in dest but not in src:")
		for file in inDestNotInSrc:
			print(file)
		print('-'*80)
		if len(inDestNotInSrc) == 0:
			print(f"No extra files found in {dests}")
		else:
			if not sys.stdin.isatty():
				raise RuntimeError("User requested stop: Not deleting extra files as stdin is not a tty")
			print(f"Files in dest but not in src count: {len(inDestNotInSrc)}")
			print("Do you want to delete them? (y/n)")
			if not input().lower().startswith('y'):
				raise RuntimeError("User requested stop: Not deleting extra files")
			start_time = time.monotonic()
			if single_thread:
				for file in inDestNotInSrc:
					if os.path.isfile(file) or os.path.islink(file):
						os.remove(file)
					else:
						shutil.rmtree(file)
			else:
				delete_file_list_parallel(inDestNotInSrc, max_workers, verbose,files_per_job)
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

def verify_src_path(src_paths: list):
	for src in list(src_paths):
		if not os.path.exists(src):
			eprint(f"Source path not found: {src} does not exist")
			src_paths.remove(src)
		# if ':' in src:
		#     print(f"Remote syncing is not supported in this version, removing source path {src}.")
		#     src_paths.remove(src)
	if len(src_paths) == 0:
		raise RuntimeError("No source paths: No source paths specified, exiting")

def load_file_list(file_list):
	file_list_value = '' if file_list is None else str(file_list)
	if file_list_value in ('-', '/dev/stdin'):
		return frozenset([entry.strip() for entry in sys.stdin.read().splitlines() if entry.strip()])
	if not os.path.exists(file_list_value):
		eprint(f"File list not found: {file_list_value} does not exist")
		return frozenset()
	with open(file_list_value, 'r') as f:
		fileList = frozenset([entry.strip() for entry in f.read().splitlines() if entry.strip()])
	return fileList

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
		file_list_file = load_file_list(file_list)
		if len(src_paths) == 1:
			# file_list_file already holds trimmed (relative) paths, so we don't
			# pass src_paths -- entries are used as-is for the comparison.
			remove_extra_files(file_list_file, [src_paths[0]], max_workers, verbose, files_per_job, single_thread, exclude=exclude, parallel_file_listing=parallel_file_listing)
			print('-' * 80)
		else:
			print("Currently only supports removing extra files for a single src_path when using file_list")
		raise RuntimeError("Completed removing extra files, exiting")
	if compare_file_list or diff_file_list:
		# This means we have a file_list and a src_path so we compare them
		print(f"Comparing file list from {src_paths} with {file_list}")
		if diff_file_list == 'auto':
			if not src_str:
				src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
			diff_file_list = f'DIFF_{src_str}_TO_{os.path.basename(os.path.realpath(file_list))}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
		workers = 1 if single_thread else max_workers
		new_map = {}
		for src in src_paths:
			new_map.update(build_entry_map(src, _trim_prefix(src), exclude=exclude, max_workers=workers, parallel_file_listing=parallel_file_listing))
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
			fileList.update(trim_paths(files, src))
			fileList.update(trim_paths(links, src))
			fileList.update([folder_path + os.path.sep for folder_path in trim_paths(folders, src)])
		print(f"Number of files: {len(fileList)}")
		print(f"Writing file list to {fileList}")
		# sort the file list
		fileList = natural_sort(fileList)
		with open(file_list, 'w') as f:
			for file in fileList:
				f.write(file + '\n')

def process_remove(src_paths: list,single_thread = False, max_workers = multiprocessing.cpu_count(),verbose = False,
				  files_per_job = 1, remove_force = False,exclude=None,batch = False,parallel_file_listing=False):
	global REMOVE_FILES_WHILE_LISTING
	#src = os.path.abspath(src +os.path.sep)
	src_paths = [os.path.abspath(src + os.path.sep) for src in src_paths if os.path.exists(src)]
	processedPaths = []
	for path in src_paths:
		while os.path.basename(path) == '':
			path = os.path.dirname(path)
		if exclude and is_excluded(path,exclude):
			print(f'Skipping excluded path: {path}')
		else:
			processedPaths.append(path)
	print(f"Removing files from {pformat(processedPaths)} with {max_workers if not single_thread else '1'} workers")
	if not remove_force:
		if not sys.stdin.isatty():
			raise RuntimeError("User requested stop: Not removing files as stdin is not a tty")
		print("Do you want to continue? (y/n)")
		if not input().lower().startswith('y'):
			raise RuntimeError("User requested stop: Not removing files")
	for path in processedPaths:
		print('-'*80)
		start_time = time.monotonic()
		if single_thread:
			get_file_list_serial(path,exclude=exclude,remove_files_while_listing=REMOVE_FILES_WHILE_LISTING)
			delete_file_bulk([path], exclude=exclude)
		else:
			delete_files_parallel(processedPaths if batch else path, max_workers, verbose=verbose,files_per_job=files_per_job,exclude=exclude,batch=batch,parallel_file_listing=parallel_file_listing)
		endTime = time.monotonic()
		print(f"Time taken to remove files: {endTime-start_time:0.4f} seconds")
		print('-'*80)
		if batch and not single_thread:
			break

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
		run_command_in_multicmd_with_path_check(["mount",target_partition,target_mount_point],strict=False)
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
					run_command_in_multicmd_with_path_check(["mount",target_partition,target_mount_point],strict=False)
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
	if len(src_paths) == 1 and os.path.isdir(src_paths[0]) and not src_paths[0].endswith(os.path.sep) and not dest.endswith(os.path.sep):
		src_paths[0] += os.path.sep
		dest += os.path.sep
	if (len(src_paths) > 1 or src_paths[0].endswith(os.path.sep)) and not dest.endswith(os.path.sep):
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
	print(f"Comparing file list from {src_paths} with {dests}")
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
			new_map.update(build_entry_map(src, _trim_prefix(src), exclude=src_exclude, max_workers=max_workers, parallel_file_listing=parallel_file_listing))
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

def process_copy(src_paths: list, dests:list = [], single_thread = False, max_workers = multiprocessing.cpu_count(),verbose = False, 
				directory_only = False,no_directory_sync = False, full_hash = False, files_per_job = 1, parallel_file_listing = False,
				exclude=None,dest_image = None,batch = False, exit_not_enough_space = False):
	total_file_list = set()
	total_sym_links = {}
	taskCtr = 0
	argDest = dests.copy()
	if batch:
		print(f"Copying from {src_paths} to {dests}")
	max_workers = max_workers if not single_thread else 1
	cs = copy_scheduler(max_workers=max_workers,full_hash=full_hash,verbose=verbose,files_per_job=files_per_job,parallel_file_listing=parallel_file_listing,exclude=exclude,exit_not_enough_space=exit_not_enough_space)
	for src in src_paths:
		taskCtr += 1
		dests = argDest.copy()
		if dest_image:
			# if the destination is a mounted image, then we use the full path for src and add that to dest.
			src = os.path.abspath(src + os.path.sep)
			# the dest path have the src full src path, we will also recersively copy the dir meta data
			# Gather all parent dirs of src
			srcParentDirs = []
			srcParent = os.path.dirname(src)
			while srcParent != '/':
				srcParentDirs.append(srcParent)
				srcParent = os.path.dirname(srcParent)
			srcParentDirs.reverse()
			# create the parent dirs in dest
			#destParentDirs = [ os.path.abspath(dest + srcParentDir + os.path.sep) for srcParentDir in srcParentDirs]
			sync_directory_metadata_bulk(srcParentDirs,dests,src_path='/')
			#dest = os.path.abspath(dest + src + os.path.sep)
			dests = [os.path.abspath(dest + src + os.path.sep) for dest in dests]
		else:
			sourceFolderName = os.path.basename(src)
			if sourceFolderName:
				src += os.path.sep
			src = os.path.abspath(src)
			newDests = []
			for dest in dests:
				if os.path.basename(dest) == '' and sourceFolderName:
					dest = os.path.join(dest,sourceFolderName)
					dest += os.path.sep
				newDests.append(os.path.abspath(dest))
			dests = newDests
		if not batch:
			print('-'*80)
			print(f"Task {taskCtr} of {len(src_paths)}, copying from {src} to {dests}")
		# verify dest is writable
		for dest in dests:
			if not os.access(os.path.dirname(os.path.abspath(dest)), os.W_OK):
				print(f"Destination {dest} is not writable, continue with caution.")
		if os.path.islink(src):
			total_sym_links[src] = dests
			print(f"{src} is a symlink, creating symlink in dests")
			continue
		# if os.path.isfile(src):
		# 	print("Copying single file")
		# 	copy_file(src, dests,full_hash=full_hash,verbose=verbose)
		# 	continue
		if os.path.isdir(src):
			src += os.path.sep
			if no_directory_sync:
				print("Skipping directory sync")
				if not NO_CREATE_DIR:
					for dest in dests:
						os.makedirs(dest, exist_ok=True)
			else:
				cs.add_dir_sync(src, dests)
		if not directory_only:
			cs.add_copy(src, dests)
		if not batch:
			fl, sl = cs.process()
			total_file_list.update(fl)
			total_sym_links.update(sl)
			cs.clear()
			print('-'*80)
	if batch:
		fl, sl = cs.process()
		total_file_list.update(fl)
		total_sym_links.update(sl)
		cs.clear()
		print(f"Processed {len(total_file_list)} files and {len(total_sym_links)} symlinks in total")
		print('-'*80)
	return total_file_list, total_sym_links

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

def detach_loop_device(loop_device, retries=5, delay=0.5):
	"""Detach a loop device with sync and retries for intermittent EBUSY."""
	if not loop_device:
		return True
	losetup = _binPaths.get('losetup', 'losetup')
	for attempt in range(1, retries + 1):
		try:
			os.sync()
		except Exception:
			pass
		if not os.path.exists(loop_device):
			print(f"Loop device {loop_device} already gone.")
			return True
		if attempt > 1:
			# Give udev / partition holders a moment to release after a busy failure.
			udevadm = shutil.which('udevadm')
			if udevadm:
				multiCMD.run_commands([[udevadm, 'settle', '--timeout=5']], timeout=10, max_threads=1, quiet=True)
			time.sleep(delay * attempt)
		tasks = multiCMD.run_commands([[losetup, '-d', loop_device]], timeout=COMMAND_TIMEOUT, max_threads=1, quiet=True, return_object=True)
		if tasks and getattr(tasks[0], 'returncode', 1) == 0:
			if attempt > 1:
				print(f"Detached loop device {loop_device} on attempt {attempt}.")
			return True
		err = ''
		if tasks:
			stderr = getattr(tasks[0], 'stderr', None) or []
			err = (stderr[-1].strip() if stderr else '') or f"return code {tasks[0].returncode}"
		eprint(f"Failed to detach {loop_device} (attempt {attempt}/{retries}): {err}")
	eprint(f"Giving up detaching loop device {loop_device}")
	return False

def clean_up(mount_points: list,loop_devices: list, delayed_commands: list = None):
	# clean up loop devices and mount points if we are using a image
	if delayed_commands is None:
		delayed_commands = []
	for mount_point in list(mount_points):
		print(f"Unmounting {mount_point}")
		run_command_in_multicmd_with_path_check(["umount",mount_point],strict=False)
		if _is_mount_point(mount_point):
			# Still busy; sync and retry once before leaving it for loop detach retries.
			try:
				os.sync()
			except Exception:
				pass
			time.sleep(0.5)
			run_command_in_multicmd_with_path_check(["umount",mount_point],strict=False)
		print(f"Removing mount point {mount_point}")
		try:
			os.rmdir(mount_point)
		except OSError as e:
			eprint(f"Failed to remove mount point {mount_point}: {e}")
	mount_points.clear()
	# Fold any delayed losetup --detach into loop_devices so they get sync/retry handling.
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
	try:
		os.sync()
	except Exception:
		pass
	for loop_device_dest in list(loop_devices):
		print(f"Removing loop device {loop_device_dest}")
		detach_loop_device(loop_device_dest)
	loop_devices.clear()

HASH_SIZE = 1<<24

def get_args(args = None):
	parser = argparse.ArgumentParser(description='Copy files from source to destination',
								  epilog=f'Found bins: {list(_binPaths.values())}\n Missing bins: {_binCalled - set(_binPaths.keys())}')
	parser.add_argument('-s', '--single_thread', action='store_true', help='Use serial processing')
	parser.add_argument('-j','-m','-t','--max_workers', type=int, default=multiprocessing.cpu_count(), help='Max workers for parallel processing. Default is 1 * CPU count. Use negative numbers to indicate {n} * CPU count, 0 means 1/2 CPU count.')
	batch_group = parser.add_mutually_exclusive_group()
	batch_group.add_argument('-b','--batch',action='store_true', help='Batch mode, process all files in one go',default=True)
	batch_group.add_argument('-nb','--no_batch','--sequential',action='store_false', dest='batch', help='Do not use batch mode', default=False)
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
	parser.add_argument('-do', '--directory_only', action='store_true', help='Only copy directory structure')
	parser.add_argument('-nds', '--no_directory_sync', action='store_true', help='Do not sync directory metadata, useful for verfication')
	parser.add_argument('-fh', '--full_hash', action='store_true', help='Checks the full hash of files')
	parser.add_argument('-hs', '--hash_size', type=int, default=1<<16, help='Hash size in bytes, default is 65536. This means hpcp will only check the last 64 KiB (about 1 page in a SSD) of the file.')
	parser.add_argument('-fpj', '--files_per_job', type=int, default=1, help='Base number of files per job, will be adjusted dynamically. Default is 1')
	parser.add_argument('-sfl','-lfl', '--source_file_list', type=str, help='Load source file list from file, or use "-" to read from stdin. Will treat it raw meaning do not expand files / folders. files are seperated using newline.  If --compare_file_list is specified, it will be used as source for compare')
	parser.add_argument('-fl','-tfl', '--target_file_list', type=str,help='Specify the file_list file to store list of files in src_path to. If --compare_file_list is specified, it will be used as targets for compare')
	parser.add_argument('-cfl', '--compare_file_list',action='store_true', help='Only compare file list. Use --file_list to specify a existing file list or specify the dest_path to compare src_path with. When not using with file_list, will compare hash.')
	parser.add_argument('-dfl', '--diff_file_list', type=str, nargs='?', const="auto",default=None, help="Implies --compare_file_list, specify a file name to store the diff file list to or omit the value to auto-determine.")
	parser.add_argument('-tdfl', '--tar_diff_file_list', action='store_true', help='Generate a tar compatible diff file list. ( update / new files only )')
	parser.add_argument('-nhfl', '--no_hash_file_list', action='store_true', help='Do not append hash to file list')
	parser.add_argument('-rm', '--remove', action='store_true', help='Remove all files and folders specified in src_path')
	parser.add_argument('-rf', '--remove_force', action='store_true', help='Remove all files without prompt')
	parser.add_argument('-rme', '--remove_extra', action='store_true', help='Remove all files and folders in dest_path that are not in src_path')
	parser.add_argument('-rwloff','--do_not_remove_files_while_listing', action='store_true', help='Do not remove files while listing. This is useful if you want to remove files after listing.')
	parser.add_argument('-e', '--exclude', action='append', default=[], help='Exclude source files matching the pattern')
	parser.add_argument('-x', '--exclude_file', type=str, help='Exclude source files matching the pattern in the file')
	parser.add_argument('-nlt', '--no_link_tracking', action='store_true', help='Do not copy files that symlinks point to.')
	parser.add_argument('-V', '--version', action='version', version=f"%(prog)s {version} @ {COMMIT_DATE} with {('XXHash' if xxhash_available else 'Blake2b')}, multiCMD V{multiCMD.version}, and [ {', '.join(_binPaths.keys())} ]; High Performance CoPy (HPC coPy) by pan@zopyr.us")
	parser.add_argument('-pfl', '--parallel_file_listing', action='store_true', help='Use parallel processing for file listing')
	parser.add_argument('src_path', nargs='*', type=str, help='Source Path')
	parser.add_argument('-si','--src_image',action='append', type=str, help='Source Image, mount the image and copy the files from it.')
	parser.add_argument('-siff','--load_diff_image',action='append', type=str, help='Apply a patch image made by -diff to the target (-d / -di or a single path). Repeat for each -dis volume. Every path the patch touches is checked first; conflicts prompt before being overridden. A disk patch (made with -dd -diff) takes the target disk with -d.')
	parser.add_argument('-d','-C','--dest_path',action='append', type=str, help='Destination Path')
	parser.add_argument('-rds','--random_dest_selection', action='store_true', help='Randomly select destination path from the list of destination paths instead of filling round robin. Can speed up transfer if dests are on different devices. Warning: can cause unable to fit in big files as dests are filled up by smaller files.')
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
	parser.add_argument('-co','--content_only', action='store_true', help='Content-only copy: do not sync mode, owner, or timestamps on files or directories. Still create missing destination directories using filesystem-default permissions (so parent ACL/setgid/setuid inherit).')
	parser.add_argument('-ctl','--command_timeout_limit', type=int, default=0, help='Set the command timeout limit in seconds for external commands ( ex. cp / dd ). Default is 0: no timeout.')
	parser.add_argument('-enes','--exit_not_enough_space', action='store_true', help='Exit if there is not enough space on the destination instead of continuing (Note: Default is continue as in compressed fs copy can be down even if source is bigger than free space).')
	try:
		args = parser.parse_intermixed_args(args)
	except Exception:
		# try to parse the arguments using parse_known_args
		args, unknown = parser.parse_known_args()
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
def hpcp(src_path, dest_paths = [], single_thread = False, max_workers = multiprocessing.cpu_count(),
			verbose = False, directory_only = False,no_directory_sync = False, full_hash = False, files_per_job = 1, target_file_list = "",
			compare_file_list = False, diff_file_list = None, tar_diff_file_list = False, remove = False,remove_force = False, remove_extra = False, parallel_file_listing = False,
			exclude=None,exclude_file = None,dest_image = None,dest_image_size = '0', no_link_tracking = False,src_image = None,dd = False,dd_resize = 0,
			batch = False, append_hash_to_file_list = True, hash_size = ..., source_file_list = None, random_destination_selection = False, 
			bytes_rate_limit = None, files_rate_limit = None,target_file_system = None, no_create_dir = False, content_only = False, command_timeout_limit = 0,
			exit_not_enough_space = False,do_not_remove_files_while_listing = False,no_fs_param_mirror = False,
			get_diff_image = None, load_diff_image = None, force_apply = False):
	global HASH_SIZE
	global RANDOM_DESTINATION_SELECTION
	global BYTES_RATE_LIMIT
	global FILES_RATE_LIMIT
	global COMMAND_TIMEOUT
	global NO_CREATE_DIR
	global CONTENT_ONLY
	global MIRROR_FS_PARAMS
	global REMOVE_FILES_WHILE_LISTING
	NO_CREATE_DIR = no_create_dir
	CONTENT_ONLY = content_only
	MIRROR_FS_PARAMS = not no_fs_param_mirror
	remove = remove or remove_force
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
				run_command_in_multicmd_with_path_check(_rw_mount_command(src_partition_path, src_mount_point))
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
					eprint(f"Error mounting {src_partition_path}, usig dd for copying.")
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
					eprint(f"Error mounting {dest_partition_path}, usig dd for copying.")
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
						exit_not_enough_space = exit_not_enough_space)
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
			src_paths.extend(load_file_list(source_file_list))
		elif not src_path and not sys.stdin.isatty():
			print("Reading source file list from stdin...")
			src_paths.extend(load_file_list('-'))
		src_paths.extend(expand_paths(src_path))
		verify_src_path(src_paths)
		if not src_str:
			src_str = "-".join([os.path.basename(src) for src in src_paths])

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

		if compare_file_list or diff_file_list:
			if diff_file_list == 'auto':
				if not src_str:
					src_str = '-'.join([os.path.basename(os.path.realpath(src)) for src in src_paths])
				diff_file_list = f'DIFF_{src_str}_TO_{dest_folder_name}_{int(time.time())}_{"tar_" if tar_diff_file_list else ""}file_list.txt'
			process_compare_file_list(src_paths, dests, max_workers=max_workers,parallel_file_listing=parallel_file_listing,
							exclude=exclude,dest_image=dest_image,diff_file_list=diff_file_list,tar_diff_file_list=tar_diff_file_list,append_hash=append_hash_to_file_list,full_hash=full_hash,src_image_roots=src_image_roots)
			raise RuntimeError("Exiting after comparing file list.")
		
		dests_to_remove = []
		for dest in dests:
			try:
				if dest and dest != ... :
					if dest.endswith(os.path.sep) and (not (os.path.exists(dest) or os.path.ismount(dest))):
						if no_create_dir:
							eprint(f"Destination path does not exist: {dest} does not exist, skipping as --no_create_dir is specified, ignoring {dest}.")
							dests_to_remove.append(dest)
						else:
							os.makedirs(dest, exist_ok=True)
					elif target_file_system:
						destfs = get_fs_type_from_path(dest)
						if destfs and destfs.lower() != target_file_system.lower():
							eprint(f"Destination fs type mismatch: Expected {target_file_system}, got {destfs}, ignoring {dest}.")
							dests_to_remove.append(dest)
			except FileExistsError :
				eprint(f"Destination path {dest} may be a mounted dir, known issue with os.path.exists\nContinuing without creating dest folder...")
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
			# The remove gate below judges the errors recorded from here on. A listing cached earlier (create_image's size
			# scan, or a library caller's) would not record its errors again: list afresh.
			_get_file_list_cache.clear()
			copy_error_start = len(ERRORS)
			total_file_list, total_sym_links = process_copy(src_paths, dests, single_thread=single_thread, max_workers=max_workers,
																			verbose=verbose, directory_only=directory_only,no_directory_sync=no_directory_sync,
																			full_hash=full_hash, files_per_job=files_per_job, parallel_file_listing=parallel_file_listing,
																			exclude=exclude,dest_image=dest_image,batch = batch,exit_not_enough_space = exit_not_enough_space)
			if verbose:
				# sort file list and sym links
				for file in natural_sort(total_file_list):
					print(f"Copied File: {file}")
				for link in total_sym_links:
					print(f"Link: {link} -> {total_sym_links[link]}")
			link_targets = create_sym_links(total_sym_links,exclude=exclude,no_link_tracking=no_link_tracking)
			# A copy that reported errors is incomplete: removing the sources (a move) would lose what was not copied,
			# and an unlisted source subtree would make its destination counterpart look extra.
			copy_failures = [error for error in ERRORS[copy_error_start:] if _is_failure(error)]
			if remove_extra and copy_failures:
				eprint(f'Remove skipped: not removing extra files from {dests}: the copy reported {len(copy_failures)} error(s); '
					'fix them (or exclude those paths with -e) and re-run')
			elif remove_extra:
				print('-'*80)
				remove_extra_files(total_file_list, dests,max_workers,verbose,files_per_job,single_thread,exclude=exclude,parallel_file_listing=parallel_file_listing,src_paths=src_paths,sym_links=total_sym_links,
					dest_image=bool(dest_image),keep=link_targets)
				print('-'*80)
				print("Removing extra directories...")
				remove_extra_dirs(src_paths, dests,exclude=exclude,dest_image=bool(dest_image),keep=link_targets)
			print('-'*80)
			if len(src_paths) > 1:
				print("Overall Summary:")
				print(f"Number of files / links: {len(total_file_list)}")
				print(f"Number of links: {len(total_sym_links)}")
				print(f"Total time taken: {time.monotonic()-programStartTime:0.4f} seconds")
		if remove and copy_failures:
			eprint(f'Remove skipped: not removing {src_paths}: the copy reported {len(copy_failures)} error(s); '
				'fix them (or exclude those paths with -e) and re-run')
		elif remove:
			process_remove(src_paths,single_thread=single_thread, max_workers=max_workers,verbose=verbose,
						files_per_job=files_per_job, remove_force=remove_force,exclude=exclude,batch=batch,parallel_file_listing=parallel_file_listing)
	except KeyboardInterrupt:
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
		clean_up(mount_points,loop_devices,delayed_commands)
		get_partition_details.cache_clear()
		get_partition_infos.cache_clear()
		hash_file.cache_clear()
		_get_file_list_cache.clear()
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
def main():
	global RANDOM_DESTINATION_SELECTION
	# Paths with bytes that are not UTF-8 arrive surrogate-escaped; under a UTF-8 locale stdout is strict, and one
	# print of such a name would end the run partway. Show them escaped instead (stderr already does).
	try:
		sys.stdout.reconfigure(errors='backslashreplace')
	except (AttributeError, ValueError):
		pass
	args = get_args()
	# we run gui if the current platform is windows and src_path is not specified
	if os.name == 'nt' and len(args.src_path) == 0:
		hpcp_gui()
	else:
		rtnCode = hpcp(args.src_path, dest_paths = args.dest_path, single_thread = args.single_thread, max_workers = args.max_workers, verbose = args.verbose,
			 directory_only =  args.directory_only, no_directory_sync = args.no_directory_sync,full_hash = args.full_hash, files_per_job = args.files_per_job,
			 target_file_list = args.target_file_list, compare_file_list = args.compare_file_list , diff_file_list = args.diff_file_list, tar_diff_file_list = args.tar_diff_file_list,remove = args.remove, remove_force =args.remove_force,
			 remove_extra = args.remove_extra, parallel_file_listing = args.parallel_file_listing,exclude = args.exclude,exclude_file = args.exclude_file,
			 dest_image = args.dest_image,dest_image_size=args.dest_image_size,no_link_tracking = args.no_link_tracking,src_image = args.src_image,dd=args.disk_dump,
			 dd_resize=args.dd_resize,batch=args.batch,append_hash_to_file_list=not args.no_hash_file_list, hash_size=args.hash_size,source_file_list=args.source_file_list,
			 random_destination_selection = args.random_dest_selection,bytes_rate_limit = args.rate_limit,files_rate_limit = args.file_rate_limit,
			 target_file_system = args.target_file_system, no_create_dir = args.no_create_dir, content_only = args.content_only, command_timeout_limit = args.command_timeout_limit,
			 exit_not_enough_space = args.exit_not_enough_space,do_not_remove_files_while_listing = args.do_not_remove_files_while_listing,
			 no_fs_param_mirror = args.no_fs_param_mirror, get_diff_image = args.get_diff_image,
			 load_diff_image = args.load_diff_image, force_apply = args.force_apply)
		if rtnCode:
			exit(rtnCode)
		

if __name__ == '__main__':
	main()
