#!/bin/bash
# Voice test on a Runpod pod: Chatterbox (MIT) renders the test passages.
# Usage (pod start command): bash voice_test.sh <SECRET> [HARD_HOURS]
set -x
SECRET=$1; HARD=${2:-1.5}
mkdir -p /w/share; touch /w/activity
cat > /w/watchdog.sh <<W
#!/bin/bash
START=\$(date +%s); LIM=\$(python3 -c "print(int($HARD*3600))")
while true; do
  sleep 60; NOW=\$(date +%s)
  if [ \$((NOW-START)) -gt \$LIM ]; then echo HARD-LIMIT >> /w/share/watchdog.log; runpodctl stop pod \$RUNPOD_POD_ID; fi
  if [ -f /w/share/DONE ] && [ \$((NOW-\$(stat -c %Y /w/activity))) -gt 1800 ]; then echo IDLE >> /w/share/watchdog.log; runpodctl stop pod \$RUNPOD_POD_ID; fi
done
W
chmod +x /w/watchdog.sh; nohup /w/watchdog.sh > /dev/null 2>&1 &
cd /w && git clone --depth 1 https://github.com/ashish91aggarwal-alt/fable-transfer repo
SECRET=$SECRET ROOT=/w/share nohup python3 repo/pod/fileserver.py 8000 > /w/fs.log 2>&1 &
pip install -q chatterbox-tts > /w/share/pip.log 2>&1
python3 repo/pod/voice_test_render.py /w/repo/channels/sleep-stories/voice-test /w/share > /w/share/render.log 2>&1
echo EXIT $? >> /w/share/render.log
touch /w/share/DONE /w/activity
sleep infinity
