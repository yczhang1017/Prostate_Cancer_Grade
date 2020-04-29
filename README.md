# Prostate_Cancer_Grade


##Download data
sudo pip install kaggle
mkdir .kaggle
echo '{"username":"yczhang1017","key":"d0c91e34db48eaa1b7e05792763e3996"}' > .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json
PATH=$PATH:./local/bin
kaggle competitions download -c prostate-cancer-grade-assessment && unzip prostate-cancer-grade-assessment.zip

##install packages
git clone https://github.com/yczhang1017/Prostate_Cancer_Grade.git 
cd Prostate_Cancer_Grade
sudo apt-get install openslide-tools
pip install openslide-python
pip install efficientnet_pytorch
pip install torch-multi-head-attention

##run 
nohup python3 -u train.py --checkpoint 'save/out_1.pth' --resume_epoch 1 > aa.log </dev/null 2>&1&
echo $! > pid.txt