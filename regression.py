import csv

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def clean_words(inp):
    r = inp.replace('.', ' ')
    r = r.replace('-', ' ')
    r = r.replace('؟', ' ')
    r = r.replace('!', ' ')
    r = r.replace(')', ' ')
    r = r.replace('(', ' ')
    r = r.replace('/', ' ')
    r = r.replace('"', ' ')
    r = r.replace('[', ' ')
    r = r.replace(']', ' ')
    return r.split(' ')


def read_data():
    train_table_func = []
    counts_func = []
    counts_title_func = []
    counts_advantages_func = []
    counts_disadvantages_func = []
    counts_products_func = []
    different_words_func = []
    different_words_title_func = []
    different_words_advantages_func = []
    different_words_disadvantages_func = []
    different_products_func = []
    train_y_func = []
    with open('data.csv', 'r', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=',')
        count_func = 0
        for row_func in reader:
            count_func += 1
            if count_func > 1:
                words = clean_words(row_func[5])
                words_title = clean_words(row_func[4])
                words_advantages = clean_words(row_func[6])
                words_disadvantages = clean_words(row_func[7])
                train_table_func.append([words, words_title, words_advantages, words_disadvantages, row_func[1]])
                result = row_func[9]
                if result == "rejected":
                    train_y_func.append(0)
                else:
                    train_y_func.append(1)
                for word in words:
                    if len(word) > 4:
                        if word not in different_words_func:
                            different_words_func.append(word)
                            counts_func.append(1)
                        else:
                            counts_func[different_words_func.index(word)] += 1
                for word in words_title:
                    if len(word) > 4:
                        if word not in different_words_title_func:
                            different_words_title_func.append(word)
                            counts_title_func.append(1)
                        else:
                            counts_title_func[different_words_title_func.index(word)] += 1
                for word in words_advantages:
                    if word not in different_words_advantages_func:
                        different_words_advantages_func.append(word)
                        counts_advantages_func.append(1)
                    else:
                        counts_advantages_func[different_words_advantages_func.index(word)] += 1
                for word in words_disadvantages:
                    if word not in different_words_disadvantages_func:
                        different_words_disadvantages_func.append(word)
                        counts_disadvantages_func.append(1)
                    else:
                        counts_disadvantages_func[different_words_disadvantages_func.index(word)] += 1
                if row_func[1] not in different_products_func:
                    different_products_func.append(row_func[1])
                    counts_products_func.append(1)
                else:
                    counts_products_func[different_products_func.index(row_func[1])] += 1
                if count_func % 10000 == 0:
                    print(len(different_words_func), len(different_words_title_func),
                          len(different_words_advantages_func),
                          len(different_words_disadvantages_func), len(different_products_func))
    return different_words_func, different_words_title_func, different_words_advantages_func, \
           different_words_disadvantages_func, different_products_func, counts_func, counts_title_func, \
           counts_advantages_func, counts_disadvantages_func, counts_products_func, \
           train_table_func, train_y_func


def write_final_rows(table, different_words, different_words_title, different_words_advantages,
                     different_words_disadvantages, different_products):
    count_func = 0
    with open("rows.csv", "w") as f:
        writer = csv.writer(f)
        for row_func in table:
            count_func += 1
            temprow = [0 for _ in range(
                len(different_words) + len(different_words_title) + len(different_words_advantages) + len(
                    different_words_disadvantages) + len(different_products))]
            for i_func in row_func[0]:
                if i_func in different_words:
                    temprow[different_words.index(i_func)] += 1
            for i_func in row_func[1]:
                if i_func in different_words_title:
                    temprow[len(different_words) + different_words_title.index(i_func)] += 1
            for i_func in row_func[2]:
                if i_func in different_words_advantages:
                    temprow[
                        len(different_words) + len(different_words_title) + different_words_advantages.index(
                            i_func)] += 1
            for i_func in row_func[3]:
                if i_func in different_words_disadvantages:
                    temprow[
                        len(different_words) + len(different_words_title) + len(
                            different_words_advantages) + different_words_disadvantages.index(i_func)] += 1
            if row_func[4] in different_products:
                temprow[
                    len(different_words) + len(different_words_title) + len(different_words_advantages) + len(
                        different_words_disadvantages) + different_products.index(row_func[4])] += 1
            if count_func % 10000 == 0:
                print(count_func)
            writer.writerow(temprow)


def create_final_rows(table, different_words, different_words_title, different_words_advantages,
                      different_words_disadvantages, different_products):
    count = 0
    train_x_func = []
    for row in table:
        count += 1
        train_x_func.append([0 for _ in range(
            len(different_words) + len(different_words_title) + len(different_words_advantages) + len(
                different_words_disadvantages) + len(different_products))])
        for i in row[0]:
            if i in different_words:
                train_x_func[len(train_x_func) - 1][different_words.index(i)] += 1
        for i in row[1]:
            if i in different_words_title:
                train_x_func[len(train_x_func) - 1][len(different_words) + different_words_title.index(i)] += 1
        for i in row[2]:
            if i in different_words_advantages:
                train_x_func[len(train_x_func) - 1][
                    len(different_words) + len(different_words_title) + different_words_advantages.index(i)] += 1
        for i in row[3]:
            if i in different_words_disadvantages:
                train_x_func[len(train_x_func) - 1][
                    len(different_words) + len(different_words_title) + len(
                        different_words_advantages) + different_words_disadvantages.index(i)] += 1
        if row[4] in different_products:
            train_x_func[len(train_x_func) - 1][
                len(different_words) + len(different_words_title) + len(different_words_advantages) + len(
                    different_words_disadvantages) + different_products.index(row[4])] += 1
        if count % 10000 == 0:
            print(count)
    return train_x_func


def clean_different_objects(different_objects, counts_objects, upper, lower):
    count = 0
    while count < len(different_objects):
        if counts_objects[count] > upper or counts_objects[count] < lower:
            different_objects.pop(count)
            counts_objects.pop(count)
            count -= 1
        count += 1
    return different_objects


def read_final_rows():
    final_table_func = []
    count = 0
    with open("rows.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if count % 2 == 0:
                final_table_func.append([int(row[i]) for i in range(len(row))])
            count += 1
    return final_table_func


def read_data_and_write_final_rows():
    different_words, different_words_title, different_words_advantages, different_words_disadvantages, \
    different_products, counts, counts_title, counts_advantages, counts_disadvantages, counts_products, \
    train_table, train_y = read_data()

    different_words = clean_different_objects(different_words, counts, 5000, 30)
    different_words_title = clean_different_objects(different_words_title, counts_title, 5000, 30)
    different_words_advantages = clean_different_objects(different_words_advantages, counts_advantages, 5000, 30)
    different_words_disadvantages = clean_different_objects(different_words_disadvantages, counts_disadvantages, 5000,
                                                            30)
    different_products = clean_different_objects(different_products, counts_products, 2000, 10)

    print("final:", len(different_words), len(different_words_title), len(different_words_advantages),
          len(different_words_disadvantages), len(different_products))

    write_final_rows(train_table, different_words, different_words_title, different_words_advantages,
                     different_words_disadvantages, different_products)


def find_responses():
    train_y_func = []
    with open('data.csv', 'r', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=',')
        count_func = 0
        for row_func in reader:
            count_func += 1
            if count_func > 1:
                result = row_func[9]
                if result == "rejected":
                    train_y_func.append(0)
                else:
                    train_y_func.append(1)
    return train_y_func


# train_x = create_final_rows(train_table)

def regress():
    final_table = read_final_rows()
    responses = find_responses()
    train_x = np.array(final_table[0:50000])
    test_x = np.array(final_table[50000:100000])
    train_y = np.array(responses[0:50000])
    test_y = np.array(responses[50000:100000])
    verified_num = 0
    rejected_num = 0
    for i in test_y:
        if i == 0:
            rejected_num += 1
        else:
            verified_num += 1
    print(verified_num, rejected_num)
    # logistic_regression(train_x, train_y, test_x, test_y, verified_num, rejected_num, 0.3)
    # logistic_regression(train_x, train_y, test_x, test_y, verified_num, rejected_num, 0.4)
    # logistic_regression(train_x, train_y, test_x, test_y, verified_num, rejected_num, 0.5)
    # logistic_regression(train_x, train_y, test_x, test_y, verified_num, rejected_num, 0.6)
    # lda(train_x, train_y, test_x, test_y, verified_num, rejected_num, [0.3, 0.4, 0.5, 0.6])
    qda(train_x, train_y, test_x, test_y, verified_num, rejected_num, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


# 10-fold CV
# print("----------------------Logistic regression--------------------")
# total_false_positive = 0
# total_false_negative = 0
# total_verified_num = 0
# total_rejected_num = 0
# for k in range(10):
#     print("fold: ", (k + 1))
#     if k == 0:
#         np_train_x = np.array(train_x[10000:100000])
#         np_train_y = np.array(train_y[10000:100000])
#         np_test_x = np.array(train_x[0:10000])
#         np_test_y = np.array(train_y[0:10000])
#     else:
#         np_train_x = np.array(train_x[0:k * 10000])
#         np_train_x.append(np.array(train_x[(k + 1) * 10000: 100000]))
#         np_train_y = np.array(train_y[0:k * 10000])
#         np_train_y.append(np.array(train_y[(k + 1) * 10000: 100000]))
#         np_test_x = np.array(train_x[k * 10000:(k + 1) * 1000])
#         np_test_y = np.array(train_y[k * 10000:(k + 1) * 1000])
#     verified_num = 0
#     rejected_num = 0
#     for i in range(len(np_test_x)):
#         if np_test_y[i] == 1:
#             verified_num += 1
#         if np_test_y[i] == 0:
#             rejected_num += 1
#     print("total number of verified samples in test set:", verified_num)
#     print("total number of rejected samples in test set:", rejected_num)
#
#     # ----------------Logistic regression---------------
#     model_logistic = LogisticRegression()
#     model_logistic.fit(np_train_x, np_train_y)
#     predicted_values_logistic = np.where(model_logistic.predict_proba(np_test_x)[:, 1] > 0.4, 1, 0)
#
#     total_miss_classified_logistic = 0
#     reject_wrong_logistic = 0
#     verify_wrong_logistic = 0
#     for i in range(len(np_test_x)):
#         total_miss_classified_logistic += abs(np_test_y[i] - predicted_values_logistic[i])
#         if np_test_y[i] == 1 and predicted_values_logistic[i] == 0:
#             reject_wrong_logistic += 1
#         if np_test_y[i] == 0 and predicted_values_logistic[i] == 1:
#             verify_wrong_logistic += 1
#     model_logistic = None
#     print("miss-classification rate :", total_miss_classified_logistic / (verified_num + rejected_num),
#           "\nFalse negative rate (type1 error) :", reject_wrong_logistic / verified_num,
#           "\nFalse positive rate (type2 error) :", verify_wrong_logistic / rejected_num)
#     total_false_negative += reject_wrong_logistic
#     total_false_positive += verify_wrong_logistic
#     total_rejected_num += rejected_num
#     total_verified_num += verified_num
# print("total False negative rate:", total_false_negative / total_verified_num)
# print("total False positive rate:", total_false_positive / total_rejected_num)
# print("total miss-classification rate:",
#       (total_false_positive + total_false_negative) / (total_verified_num + total_rejected_num))


# ----------------Logistic regression---------------
def logistic_regression(np_train_x, np_train_y, np_test_x, np_test_y, verified_num, rejected_num, p):
    model_logistic = LogisticRegression()
    model_logistic.fit(np_train_x, np_train_y)
    predicted_values_logistic = np.where(model_logistic.predict_proba(np_test_x)[:, 1] > p, 1, 0)

    total_miss_classified_logistic = 0
    reject_wrong_logistic = 0
    verify_wrong_logistic = 0
    for i in range(len(np_test_x)):
        total_miss_classified_logistic += abs(np_test_y[i] - predicted_values_logistic[i])
        if np_test_y[i] == 1 and predicted_values_logistic[i] == 0:
            reject_wrong_logistic += 1
        if np_test_y[i] == 0 and predicted_values_logistic[i] == 1:
            verify_wrong_logistic += 1
    print("miss-classification rate :", total_miss_classified_logistic / (verified_num + rejected_num),
          "\nFalse negative rate (type1 error) :", reject_wrong_logistic / verified_num,
          "\nFalse positive rate (type2 error) :", verify_wrong_logistic / rejected_num)


# ----------------LDA---------------
def lda(np_train_x, np_train_y, np_test_x, np_test_y, verified_num, rejected_num, p):
    model_LDA = LinearDiscriminantAnalysis()
    model_LDA.fit(np_train_x, np_train_y)
    for prob in p:
        predicted_values_LDA = np.where(model_LDA.predict_proba(np_test_x)[:, 1] > prob, 1, 0)

        total_miss_classified_LDA = 0
        reject_wrong_LDA = 0
        verify_wrong_LDA = 0
        for i in range(len(np_test_x)):
            total_miss_classified_LDA += abs(np_test_y[i] - predicted_values_LDA[i])
            if np_test_y[i] == 1 and predicted_values_LDA[i] == 0:
                reject_wrong_LDA += 1
            if np_test_y[i] == 0 and predicted_values_LDA[i] == 1:
                verify_wrong_LDA += 1
        print("\n----------------------Linear Discriminant Analysis prob:", prob, "--------------------")
        print("miss-classification rate :", total_miss_classified_LDA / 25000,
              "\nFalse negative rate (type1 error) :", reject_wrong_LDA / verified_num,
              "\nFalse positive rate (type2 error) :", verify_wrong_LDA / rejected_num)


# ----------------QDA---------------
def qda(np_train_x, np_train_y, np_test_x, np_test_y, verified_num, rejected_num, p):
    model_QDA = QuadraticDiscriminantAnalysis()
    model_QDA.fit(np_train_x, np_train_y)
    for prob in p:
        predicted_values_QDA = np.where(model_QDA.predict_proba(np_test_x)[:, 1] > prob, 1, 0)

        total_miss_classified_QDA = 0
        reject_wrong_QDA = 0
        verify_wrong_QDA = 0
        for i in range(len(np_test_x)):
            total_miss_classified_QDA += abs(np_test_y[i] - predicted_values_QDA[i])
            if np_test_y[i] == 1 and predicted_values_QDA[i] == 0:
                reject_wrong_QDA += 1
            if np_test_y[i] == 0 and predicted_values_QDA[i] == 1:
                verify_wrong_QDA += 1
        print("\n----------------------Quadratic Discriminant Analysis prob:", prob, "--------------------")
        print("miss-classification rate :", total_miss_classified_QDA / (rejected_num + verified_num),
              "\nFalse negative rate (type1 error) :", reject_wrong_QDA / verified_num,
              "\nFalse positive rate (type2 error) :", verify_wrong_QDA / rejected_num)


# -----------------KNN 100----------------
def knn(np_train_x, np_train_y, np_test_x, np_test_y, verified_num, rejected_num, k):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(np_train_x, np_train_y)

    total_miss_classified_KNN100 = 0
    reject_wrong_KNN100 = 0
    verify_wrong_KNN100 = 0
    for i in range(len(np_test_x)):
        predicted = model_knn.predict(np_test_x[i].reshape(1, -1))
        total_miss_classified_KNN100 += abs(np_test_y[i] - predicted)
        if np_test_y[i] == 1 and predicted == 0:
            reject_wrong_KNN100 += 1
        if np_test_y[i] == 0 and predicted == 1:
            verify_wrong_KNN100 += 1
    print("\n----------------------KNN", k, "--------------------")
    print("miss-classification rate :", total_miss_classified_KNN100 / (rejected_num + verified_num),
          "\nFalse negative rate (type1 error) :", reject_wrong_KNN100 / verified_num,
          "\nFalse positive rate (type2 error) :", verify_wrong_KNN100 / rejected_num)


regress()
