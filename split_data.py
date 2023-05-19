from regex import D
from sklearn.datasets import load_iris # 샘플 데이터 로딩
from sklearn.model_selection import train_test_split
from glob import glob
import cv2


data_label = []
data_val = []

data_path = './archive/brain_tumor_dataset'
categ  = glob(data_path + '/*')
print(categ)
for e_cate in categ :
    
    data_pics = glob(e_cate + '/*')
    
    
    for e_pic in data_pics :


        pic_val = cv2.imread(e_pic, cv2.IMREAD_COLOR)
        data_val.append(pic_val)

        data_label.append(str(e_cate.split('\\')[-1]))


X_train, X_val, y_train, y_val = train_test_split(data_val, data_label, test_size = 0.2, shuffle = True, stratify = data_label)

print(len(X_train))
print(len(X_val))
print(len(y_train))
print(len(y_val))
print(set(y_val))

train_path = './brain_tumor_data_splited/train'
val_path = './brain_tumor_data_splited/val'

for pic_num in range(len(X_train)) :
    cv2.imwrite(train_path + '/' + str(y_train[pic_num]) + '/train_'+str(pic_num)+'.jpg', X_train[pic_num])

for pic_num2 in range(len(X_val)) :
    cv2.imwrite(val_path + '/' + str(y_val[pic_num2]) + '/val_'+str(pic_num2)+'.jpg', X_val[pic_num2])