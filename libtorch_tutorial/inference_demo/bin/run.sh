
cd ../release&& rm -rf ./* &&cmake .. && make  && rm -rf ./* && cd ../bin

python my_model.py &&./out my_model.pt


