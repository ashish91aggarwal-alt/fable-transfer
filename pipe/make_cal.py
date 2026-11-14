"""Generate the tuning set on the pod with the stand-in model, then write jobs.jsonl for checks2."""
import subprocess, json, os
I='/tmp/dry/inputs/'; NN=['01','02','03','04','07','10','12','13','14','15','16']
jobs=[(n,f'{I}start_{n}.png',f'{I}end_{n}.png',s) for n in NN for s in (1001,1002,1003,1004)]+[('01w',f'{I}start_01.png',f'{I}start_01.png',s) for s in (1001,1002,1003)]+[('13w',f'{I}start_13.png',f'{I}start_13.png',s) for s in (1001,1002)]
os.makedirs('/w/cal',exist_ok=True); out=open('/w/cal/jobs.jsonl','w')
for n,st,en,s in jobs:
    o=f'/w/cal/{n}_{s}.mp4'
    p=subprocess.run(['python3','/tmp/dry/model/fakemodel.py','--start',st,'--end',en,'--prompt','cal '+n,'--seed',str(s),'--out',o],capture_output=True,text=True)
    if p.returncode==0 and os.path.exists(o): out.write(json.dumps({'clip':o,'start':st,'end':en,'n':n,'seed':s})+'\n')
    os.utime('/w/activity')
print('ok')
