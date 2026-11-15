#!/bin/bash
# Train the house-style LoRA on a Runpod pod. Usage (inside pod): bash train_lora.sh <SECRET> [HARD_HOURS]
# Data: lora/data (png + txt captions) and lora/train.yaml from the public repo. Output served by the file server.
set -x
SECRET=$1; HARD=${2:-4}
mkdir -p /w/share /w/lora_out; touch /w/activity
# --- kill switch: hard time limit + idle limit (no file-server activity and no trainer for 30 min)
cat > /w/watchdog.sh <<W
#!/bin/bash
START=\$(date +%s)
while true; do
  sleep 60; NOW=\$(date +%s)
  if [ \$((NOW-START)) -gt \$(( $HARD*3600 )) ]; then echo HARD-LIMIT >> /w/watchdog.log; runpodctl stop pod \$RUNPOD_POD_ID; fi
  if ! pgrep -f run.py >/dev/null && [ \$((NOW-\$(stat -c %Y /w/activity))) -gt 1800 ]; then echo IDLE >> /w/watchdog.log; runpodctl stop pod \$RUNPOD_POD_ID; fi
done
W
chmod +x /w/watchdog.sh; nohup /w/watchdog.sh > /dev/null 2>&1 &
cd /w && git clone --depth 1 https://github.com/ashish91aggarwal-alt/fable-transfer repo
SECRET=$SECRET ROOT=/w/share nohup python3 repo/pod/fileserver.py 8000 > /w/fs.log 2>&1 &
cp -r repo/lora/data /w/lora_data
git clone --depth 1 https://github.com/ostris/ai-toolkit /w/ait
cd /w/ait && pip install -q -r requirements.txt && pip install -q hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 run.py /w/repo/lora/train.yaml > /w/train.log 2>&1
echo TRAIN-EXIT $? >> /w/train.log
# publish: every saved LoRA split into 90 MB parts (git file limit), plus samples
cd /w/lora_out && for f in $(find . -name "*.safetensors"); do n=$(basename $f .safetensors); mkdir -p /w/share/lora; split -b 90M -d $f /w/share/lora/$n.part; sha256sum $f > /w/share/lora/$n.sha256; done
mkdir -p /w/share/samples && find /w/lora_out -path "*samples*" -name "*.jpg" -exec cp {} /w/share/samples/ \;
ls -la /w/share/lora /w/share/samples > /w/share/MANIFEST.txt; touch /w/activity; echo DONE >> /w/train.log
