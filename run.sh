nohup python3.5 -u -OO train.py > log.txt </dev/null 2>&1&
echo $! > pid.txt