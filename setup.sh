sed -i 's|cd /data/coding|# cd /data/coding|' ~/.bashrc

git config --global user.name "complooea"
git config --global user.email "complooea@gmail.com"

apt-get update
apt-get install zlib1g-dev

source ~/.bashrc

pip install -e .