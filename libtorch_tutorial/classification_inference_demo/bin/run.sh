cd ../ && mkdir release && cd ./release && cmake .. && make &&cd ../ && rm -rf release && cd ./bin

python resnet18_classification.py &&./out resnet18_model.pt dog.png synset_words.txt

rm out resnet18_model.pt
