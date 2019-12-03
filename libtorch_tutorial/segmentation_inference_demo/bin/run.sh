cd ../ && mkdir release && cd ./release && cmake .. && make &&cd ../ && rm -rf release && cd ./bin

python segmentation_inference.py 

./out deeplabv3_resnet101_model.pt test_image.jpg 

rm out deeplabv3_resnet101_model.pt
