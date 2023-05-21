import pickle

# pip install scikit-learn
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

app = Flask(__name__)

menu = [{"name": "Линейная регрессия", "url": "p_knn"},
        {"name": "SGD", "url": "p_lab2"},
        {"name": "KNN", "url": "p_lab3"},
        {"name": "Метрики качества", "url": "p_lab4"}]

loaded_model_knn = pickle.load(open('model/knn_pickle_file', 'rb'))

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Кузиной Е.С.", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        dataset_2 = np.array(pd.read_csv('D:/ЛИЗА/БГИТУ/Python Казаков (МО)/Lizok/model/boston.csv')[['crim', 'rm', 'age']])
        dataset_4 = np.array(pd.read_csv('D:/ЛИЗА/БГИТУ/Python Казаков (МО)/Lizok/model/boston.csv')[['medv']])
        model = LinearRegression()
        model.fit(dataset_2, dataset_4)
        pred = model.predict([[X_new[0][0], X_new[0][1], X_new[0][2]]])
        #pred = int(loaded_model_knn.predict(X_new))
        #sort = ['setosa', 'versicolor', 'virginica']
        #result = sort[pred]
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + str(*pred[0]))

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Метод SGD", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7'])]])
        dataset_2 = np.array(
            pd.read_excel('D:/ЛИЗА/БГИТУ/Python Казаков (МО)/Lizok/model/seeds_dataset.xlsx', nrows=649)[['площадь', 'периметр', 'компактность', 'длина ядра', 'ширина ядра', 'коэффициент асимметрии', 'длина желобка ядра']])
        dataset_4 = np.array(pd.read_excel('D:/ЛИЗА/БГИТУ/Python Казаков (МО)/Lizok/model/seeds_dataset.xlsx', nrows=649)[['класс']])
        model = SGDClassifier(alpha=0.001, max_iter=1000, random_state = 0)
        model.fit(dataset_2, dataset_4)
        pred = model.predict([[X_new[0][0], X_new[0][1], X_new[0][2], X_new[0][3], X_new[0][4], X_new[0][5], X_new[0][6]]])
        print(pred)
        return render_template('lab2.html', title="Метод SGD", menu=menu,
                               class_model="Это " + str(*pred) + " сорт")

@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Метод KNN", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           float(request.form['list6']),
                           float(request.form['list7'])]])
        dataset_2 = np.array(
            pd.read_excel('D:/ЛИЗА/БГИТУ/Python Казаков (МО)/Lizok/model/seeds_dataset.xlsx', nrows=649)[
                ['площадь', 'периметр', 'компактность', 'длина ядра', 'ширина ядра', 'коэффициент асимметрии',
                 'длина желобка ядра']])
        dataset_4 = np.array(
            pd.read_excel('D:/ЛИЗА/БГИТУ/Python Казаков (МО)/Lizok/model/seeds_dataset.xlsx', nrows=649)[['класс']])
        model = KNeighborsClassifier(n_neighbors=6)
        model.fit(dataset_2, dataset_4)
        pred = model.predict(
            [[X_new[0][0], X_new[0][1], X_new[0][2], X_new[0][3], X_new[0][4], X_new[0][5], X_new[0][6]]])
        print(pred)
        return render_template('lab3.html', title="Метод KNN", menu=menu,
                               class_model="Это " + str(*pred) + " сорт")

@app.route("/p_lab4")
def f_lab4():
    return render_template('lab4.html', title="Метрики качества", menu=menu)

if __name__ == "__main__":
    app.run(debug=True)
