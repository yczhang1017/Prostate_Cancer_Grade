nohup python3 -u train.py > log.txt </dev/null 2>&1&
echo $! > pid.txt
