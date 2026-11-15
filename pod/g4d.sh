#!/bin/bash
# G4d two-reference placement test. Usage: bash g4d.sh <SECRET> <HARD_MINUTES>
SECRET=$1; HARDM=${2:-17}
mkdir -p /w/share/out; touch /w/activity; date +%s > /w/start
cat > /w/watchdog.sh <<W
#!/bin/bash
LIM=$((HARDM*60))
while true; do
  sleep 30; NOW=\$(date +%s); START=\$(cat /w/start)
  if [ \$((NOW-START)) -gt \$LIM ]; then echo HARD-LIMIT >> /w/share/watchdog.log; runpodctl remove pod \$RUNPOD_POD_ID; fi
  if [ -f /w/share/DONE ] && [ \$((NOW-\$(stat -c %Y /w/share/DONE))) -gt 600 ]; then echo IDLE >> /w/share/watchdog.log; runpodctl remove pod \$RUNPOD_POD_ID; fi
done
W
chmod +x /w/watchdog.sh; nohup /w/watchdog.sh >/dev/null 2>&1 &
cd /w && git clone -q --depth 1 https://github.com/ashish91aggarwal-alt/fable-transfer repo
SECRET=$SECRET ROOT=/w/share nohup python3 repo/pod/fileserver.py 8000 > /w/fs.log 2>&1 &
echo "installing" > /w/share/STATUS
pip install -q hf_transfer; export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q -U "diffusers>=0.37" transformers accelerate "huggingface_hub>=1.5,<2" > /w/share/pip.log 2>&1
echo "running" > /w/share/STATUS
python3 repo/tortoise-hare/cast-job/klein.py both repo/tortoise-hare/cast-g4d/jobs.json > /w/share/job.log 2>&1
echo "EXIT $?" >> /w/share/job.log
cp -r repo/tortoise-hare/cast-g4d/out/. /w/share/out/ 2>/dev/null
(cd /w/share && ls out > MANIFEST.txt)
echo "done" > /w/share/STATUS; touch /w/share/DONE
sleep infinity
