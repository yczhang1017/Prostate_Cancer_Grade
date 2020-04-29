nohup python3.5 -u -O2 train.py > log.txt </dev/null 2>&1&
echo $! > pid.txt