#!/bin/bash
# Two open-weight TTS models (Qwen3-TTS VoiceDesign, VoxCPM2) in separate venvs + gate checks. Usage: bash models_test.sh <SECRET> [HARD_HOURS]
set -x
SECRET=$1; HARD=${2:-1}
mkdir -p /w/share/r6; touch /w/activity
cat > /w/watchdog.sh <<W
#!/bin/bash
START=\$(date +%s); LIM=\$(python3 -c "print(int($HARD*3600))")
while true; do
  sleep 60; NOW=\$(date +%s)
  if [ \$((NOW-START)) -gt \$LIM ]; then echo HARD-LIMIT >> /w/share/watchdog.log; runpodctl stop pod \$RUNPOD_POD_ID; fi
  if [ -f /w/share/DONE ] && [ \$((NOW-\$(stat -c %Y /w/share/DONE))) -gt 1200 ]; then echo IDLE >> /w/share/watchdog.log; runpodctl stop pod \$RUNPOD_POD_ID; fi
done
W
chmod +x /w/watchdog.sh; nohup /w/watchdog.sh > /dev/null 2>&1 &
cd /w && git clone --depth 1 https://github.com/ashish91aggarwal-alt/fable-transfer repo
SECRET=$SECRET ROOT=/w/share nohup python3 repo/pod/fileserver.py 8000 > /w/fs.log 2>&1 &
(apt-get update -qq && apt-get install -y -qq ffmpeg sox) > /w/share/apt.log 2>&1
SRC=/w/repo/channels/sleep-stories/voice-test/round6-models
python3 -m venv /w/q && /w/q/bin/pip install -q -U pip > /w/share/pip_q.log 2>&1 && /w/q/bin/pip install -q qwen-tts >> /w/share/pip_q.log 2>&1
/w/q/bin/python repo/pod/models_test.py qwen $SRC /w/share/r6 > /w/share/render_q.log 2>&1; echo EXIT $? >> /w/share/render_q.log
python3 -m venv /w/v && /w/v/bin/pip install -q -U pip > /w/share/pip_v.log 2>&1 && /w/v/bin/pip install -q voxcpm >> /w/share/pip_v.log 2>&1
/w/v/bin/python repo/pod/models_test.py voxcpm $SRC /w/share/r6 > /w/share/render_v.log 2>&1; echo EXIT $? >> /w/share/render_v.log
python3 -m venv /w/c && /w/c/bin/pip install -q faster-whisper librosa > /w/share/pip_c.log 2>&1
/w/c/bin/python repo/pod/models_check.py $SRC /w/share/r6 > /w/share/check.log 2>&1; echo EXIT $? >> /w/share/check.log
touch /w/share/DONE
sleep infinity
