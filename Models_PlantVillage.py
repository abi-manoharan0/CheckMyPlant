from os import listdir
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from image_convertor import image_convertor
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#dataset:plantvillage

#load image(diretory can be changed(3 layers of folder to images)
diretory_root='C:\CenStudy\Software development 2\Plant Disease\PlantVillage'
image_list,label_list=[],[]
try:
    print("Loading Images")
    root_dir=listdir(diretory_root)
    print("a")
    for i in root_dir:
        if i ==".DS_Store":
            root_dir.remove(i)


    for plant_disease_folder in root_dir:
            plant_disease_image_list=listdir(f"{diretory_root}/{plant_disease_folder}")
            print("c")
            for image in plant_disease_image_list[:10]:#change the number for more data in
                image_directory=f"{diretory_root}/{plant_disease_folder}/{image}"
                print("d")
                if image_directory.endswith(".jpg")!=True or image_directory.endswith(".JPG")!=True:#dave
                    the_image=image_convertor(image_directory)

                    image_list.append(the_image.flatten())
                    label_list.append(plant_disease_folder)
                    print("e")
    print("Image Loaded")
except Exception as e:
    print(f"Error:{e}")
#Binary label (whether a class or not)
label_binarizer=LabelBinarizer()
image_labels=label_binarizer.fit_transform(label_list)
n_classes=len(label_binarizer.classes_)
print(label_binarizer.classes_)
print(image_labels)
#scale and test and train split(svm)
scal_image_list=np.array(image_list,dtype=np.float16)/255.0
x_train,x_test,y_train,y_test=train_test_split(scal_image_list,label_list,test_size=0.2,random_state=11,stratify=image_labels)

print("ss",len(label_list))
#SVM
svm_model=SVC(C=.2,kernel='rbf')
svm_model.fit(x_train,y_train)
print(classification_report(y_test,svm_model.predict(x_test)))

#RandomForest
RF_model=RandomForestClassifier()
RF_model.fit(x_train,y_train)
print(classification_report(y_test,RF_model.predict(x_test)))
#ANN
import tensorflow as tf
ann_model=keras.Sequential([

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])

ann_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)

x_train,x_test,y_train,y_test=train_test_split(scal_image_list,image_labels,test_size=0.2,random_state=11,stratify=image_labels)

print(y_train.shape)
print(x_train.shape)
print(y_test.shape)
print(x_test.shape)
his=ann_model.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=10,

)
#epocha and batch size is small, adjust it for better accuracy