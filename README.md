# Prostate_Cancer_Grade


##Download data
pip install kaggle
mkdir .kaggle
echo '{"username":"yczhang1017","key":"d0c91e34db48eaa1b7e05792763e3996"}' > .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json
PATH=$PATH:./local/bin
kaggle competitions download -c prostate-cancer-grade-assessment

##install packages
pip install openslide-python

pip install efficientnet_pytorch
pip install torch-multi-head-attention