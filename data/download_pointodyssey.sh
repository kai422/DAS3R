# Download point_odyssey
mkdir -p point_odyssey
cd point_odyssey
# train
# gdown --id 1ivaHRZV6iwxxH4qk8IAIyrOF9jrppDIP
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partaa
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partab
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partac
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partad

cat train.tar.gz.part* > train.tar.gz

# test
# gdown --id 1jn8l28BBNw9f9wYFmd5WOCERH48-GsgB
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/test.tar.gz

# sample
# gdown --id 1dnl9XMImdwKX2KcZCTuVDhcy5h8qzQIO
wget https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/sample.tar.gz

# unzip all *.tar.gz
find . -name "*.tar.gz" -exec tar -zxvf {} \;
# remove all zip files
find . -name "*.tar.gz" -exec rm {} \;