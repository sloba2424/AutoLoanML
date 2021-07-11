import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
import re
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from train_model import train_model1
from sklearn.svm import SVC

if __name__ == '__main__':

    ds_train = pd.read_csv('train.csv')
    ds_test = pd.read_csv('test.csv')

    # Prikaz broja redova i kolona
    print("Izgled train data seta "+str(ds_train.shape))
    print("Izgled test data seta "+str(ds_test.shape))



    # Konvertovanje kolone 'CREDIT.HISTORY.LENGTH' u mesece u trening setu
    ds_train['CREDIT.HISTORY.LENGTH'] = ds_train['CREDIT.HISTORY.LENGTH'].apply(
        lambda x: (re.sub('[a-z]', '', x)).split())
    ds_train['CREDIT.HISTORY.LENGTH'] = ds_train['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(x[0]) * 12 + int(x[1]))

    # Konvertovanje kolone 'AVERAGE.ACCT.AGE' u mesece u trening setu
    ds_train['AVERAGE.ACCT.AGE'] = ds_train['AVERAGE.ACCT.AGE'].apply(lambda x: (re.sub('[a-z]', '', x)).split())
    ds_train['AVERAGE.ACCT.AGE'] = ds_train['AVERAGE.ACCT.AGE'].apply(lambda x: int(x[0]) * 12 + int(x[1]))

    # Konvertovanje kolone 'CREDIT.HISTORY.LENGTH' u mesece u test setu
    ds_test['CREDIT.HISTORY.LENGTH'] = ds_test['CREDIT.HISTORY.LENGTH'].apply(
        lambda x: (re.sub('[a-z]', '', x)).split())
    ds_test['CREDIT.HISTORY.LENGTH'] = ds_test['CREDIT.HISTORY.LENGTH'].apply(lambda x: int(x[0]) * 12 + int(x[1]))

    # Konvertovanje kolone 'AVERAGE.ACCT.AGE' u mesece u test setu
    ds_test['AVERAGE.ACCT.AGE'] = ds_test['AVERAGE.ACCT.AGE'].apply(lambda x: (re.sub('[a-z]', '', x)).split())
    ds_test['AVERAGE.ACCT.AGE'] = ds_test['AVERAGE.ACCT.AGE'].apply(lambda x: int(x[0]) * 12 + int(x[1]))

    # Pretvaranje object u date, posto imamo dve kolone koje su datumske [date of birth,DisbursalDate]
    ds_train['Date.of.Birth'] = pd.to_datetime(ds_train['Date.of.Birth'])
    ds_train['DisbursalDate'] = pd.to_datetime(ds_train['DisbursalDate'])

    ds_test['Date.of.Birth'] = pd.to_datetime(ds_test['Date.of.Birth'])
    ds_test['DisbursalDate'] = pd.to_datetime(ds_test['DisbursalDate'])

    # Vremenske kolone prebacujemo i kreiramo nove koje ce uzimati vrednosti
    ds_train['Year'] = ds_train['Date.of.Birth'].dt.year
    ds_train['Month'] = ds_train['Date.of.Birth'].dt.month
    ds_train['Day'] = ds_train['Date.of.Birth'].dt.day

    ds_test['Year'] = ds_test['Date.of.Birth'].dt.year
    ds_test['Month'] = ds_test['Date.of.Birth'].dt.month
    ds_test['Day'] = ds_test['Date.of.Birth'].dt.day

    ds_train['Year_DD'] = ds_train['DisbursalDate'].dt.year
    ds_train['Month_DD'] = ds_train['DisbursalDate'].dt.month
    ds_train['Day_DD'] = ds_train['DisbursalDate'].dt.day

    ds_test['Year_DD'] = ds_test['DisbursalDate'].dt.year
    ds_test['Month_DD'] = ds_test['DisbursalDate'].dt.month
    ds_test['Day_DD'] = ds_test['DisbursalDate'].dt.day

    # Pretvaram kolonu Year u string kako bi sredili lose ocitane podatke
    ds_train['Year'] = ds_train['Year'].astype(str)
    ds_test['Year'] = ds_test['Year'].astype(str)

    for i in range(0, ds_train.shape[0]):
        if ds_train['Year'].iloc[i] == '1900':
            ds_train['Year'].iloc[i] = '2000'

    for i in range(0, ds_test.shape[0]):
        if ds_test['Year'].iloc[i] == '1900':
            ds_test['Year'].iloc[i] = '2000'

    for i in range(0, ds_train.shape[0]):
        if ds_train['Year'].iloc[i] == '1901':
            ds_train['Year'].iloc[i] = '2001'

    for i in range(0, ds_test.shape[0]):
        if ds_test['Year'].iloc[i] == '1901':
            ds_test['Year'].iloc[i] = '2001'

    ds_train['Year'] = pd.to_numeric(ds_train['Year'])
    ds_test['Year'] = pd.to_numeric(ds_test['Year'])

    # Posto nema varijansu izbrisacemo tabelu i iz trening i iz testa, brise se kolona ID i datumske kolone
    ds_train.drop(['UniqueID'], axis=1, inplace=True)
    ds_train.drop(['MobileNo_Avl_Flag'], axis=1, inplace=True)

    ds_test.drop(['UniqueID'], axis=1, inplace=True)
    ds_test.drop(['MobileNo_Avl_Flag'], axis=1, inplace=True)


    def rad_sa_podacima(CSV):
        CSV['DisbursalDate'] = pd.to_datetime(CSV['DisbursalDate'], format="%d-%m-%y", infer_datetime_format=True)
        CSV['Date.of.Birth'] = pd.to_datetime(CSV['Date.of.Birth'], format="%d-%m-%y", infer_datetime_format=True)

        now = pd.Timestamp('now')
        CSV['Age'] = (now - CSV['Date.of.Birth']).astype('<m8[Y]').astype(int)
        age_mean = int(CSV[CSV['Age'] > 0]['Age'].mean())
        CSV.loc[:, 'age'] = CSV['Age'].apply(lambda x: x if x > 0 else age_mean)

        CSV['disbursal_months_passed'] = ((now - CSV['DisbursalDate']) / np.timedelta64(1, 'M')).astype(int)
        CSV['number_of_0'] = (CSV == 0).astype(int).sum(axis=1)

        CSV.loc[:, 'loan_to_asset_ratio'] = CSV['disbursed_amount'] / CSV['asset_cost']
        CSV.loc[:, 'no_of_accts'] = CSV['PRI.NO.OF.ACCTS'] + CSV['SEC.NO.OF.ACCTS']

        CSV.loc[:, 'pri_inactive_accts'] = CSV['PRI.NO.OF.ACCTS'] - CSV['PRI.ACTIVE.ACCTS']
        CSV.loc[:, 'sec_inactive_accts'] = CSV['SEC.NO.OF.ACCTS'] - CSV['SEC.ACTIVE.ACCTS']
        CSV.loc[:, 'tot_inactive_accts'] = CSV['pri_inactive_accts'] + CSV['sec_inactive_accts']
        CSV.loc[:, 'tot_overdue_accts'] = CSV['PRI.OVERDUE.ACCTS'] + CSV['SEC.OVERDUE.ACCTS']
        CSV.loc[:, 'tot_current_balance'] = CSV['PRI.CURRENT.BALANCE'] + CSV['SEC.CURRENT.BALANCE']
        CSV.loc[:, 'tot_sanctioned_amount'] = CSV['PRI.SANCTIONED.AMOUNT'] + CSV['SEC.SANCTIONED.AMOUNT']
        CSV.loc[:, 'tot_disbursed_amount'] = CSV['PRI.DISBURSED.AMOUNT'] + CSV['SEC.DISBURSED.AMOUNT']
        CSV.loc[:, 'tot_installment'] = CSV['PRIMARY.INSTAL.AMT'] + CSV['SEC.INSTAL.AMT']
        CSV.loc[:, 'bal_disburse_ratio'] = np.round(
            (1 + CSV['tot_disbursed_amount']) / (1 + CSV['tot_current_balance']), 2)
        CSV.loc[:, 'pri_tenure'] = (CSV['PRI.DISBURSED.AMOUNT'] / (CSV['PRIMARY.INSTAL.AMT'] + 1)).astype(int)
        CSV.loc[:, 'sec_tenure'] = (CSV['SEC.DISBURSED.AMOUNT'] / (CSV['SEC.INSTAL.AMT'] + 1)).astype(int)
        CSV.loc[:, 'disburse_to_sactioned_ratio'] = np.round(
            (CSV['tot_disbursed_amount'] + 1) / (1 + CSV['tot_sanctioned_amount']), 2)
        CSV.loc[:, 'active_to_inactive_act_ratio'] = np.round(
            (CSV['no_of_accts'] + 1) / (1 + CSV['tot_inactive_accts']), 2)

        return CSV



    ds_train = rad_sa_podacima(ds_train)
    ds_test = rad_sa_podacima(ds_test)

    ds_train.drop(['Year_DD'], axis=1, inplace=True)
    ds_test.drop(['Year_DD'], axis=1, inplace=True)

    # Broj null vrednosti train skupa
    ds_train.isnull().sum()

    # broj null vrednosti employement type kolone
    ds_train['Employment.Type'].value_counts()

    # Posto je utvrdjeno da ima samo 7661 redova koji nemaju vrednosti, dropovacemo ove redove
    print("Odnos null redova u odnosu na ceo data set:",
          ds_train['Employment.Type'].isnull().sum() / ds_train.shape[0] * 100, "%")

    ds_train = ds_train.dropna()
    ds_test = ds_test.dropna()

    # Dropujemo stare kolone, jer su obradjene i vise ne trebaju
    ds_train.drop(labels=['Date.of.Birth'], axis=1, inplace=True)
    ds_train.drop(labels=['DisbursalDate'], axis=1, inplace=True)

    ds_test.drop(labels=['Date.of.Birth'], axis=1, inplace=True)
    ds_test.drop(labels=['DisbursalDate'], axis=1, inplace=True)

    # podela kategorickih i numerickih vrednosti zbog prikaza na grafiku
    ds_cat = ds_train.select_dtypes(include='object')
    ds_num = ds_train.select_dtypes(exclude='object')

    # Parametri n1,m1,n2,m2 uzimaju brojeve redova i kolona kako bi napravili grafik
    n1, m1 = ds_cat.shape
    n2, m2 = ds_num.shape

    # Prikaz koliko je u data setu kategorickih a koliko numerickih kolona
    objects = ('Numericki', 'Kategoricki')
    bar = np.arange(len(objects))
    vrednost = [m2, m1]

    plt.bar(bar, vrednost, align='center', alpha=0.5)
    plt.xticks(bar, objects)
    plt.title('Odnos numerickih i kategorickih podataka')
    plt.show()

    print("Procentualni odnos kategorickih i numerickih podataka:", m1 / (m1 + m2) * 100, "%")


    # Label Encoder za kolonu employment type
    le = LabelEncoder()
    ds_train.iloc[:, 7] = le.fit_transform(ds_train.iloc[:, 7])

    ds_test.iloc[:, 7] = le.fit_transform(ds_test.iloc[:, 7])

    # Posto je veliki broj unknown, dodeljujemo vrednost -1, dok ostale po visini rasporedjujemo
    # ovaj nain je mnogo bolji nego preko getdummy funkcije
    MapiranjeCNSKolone = {'No Bureau History Available': -1,
                          'I-Medium Risk': 2,
                          'L-Very High Risk': 0,
                          'A-Very Low Risk': 4,
                          'Not Scored: Not Enough Info available on the customer': -1,
                          'D-Very Low Risk': 4,
                          'M-Very High Risk': 0,
                          'B-Very Low Risk': 4,
                          'C-Very Low Risk': 4,
                          'E-Low Risk': 3,
                          'H-Medium Risk': 2,
                          'F-Low Risk': 3,
                          'K-High Risk': 1,
                          'Not Scored: No Activity seen on the customer (Inactive)': -1,
                          'Not Scored: Sufficient History Not Available': -1,
                          'Not Scored: No Updates available in last 36 months': -1,
                          'G-Low Risk': 3,
                          'J-High Risk': 1,
                          'Not Scored: Only a Guarantor': -1,
                          'Not Scored: More than 50 active Accounts found': -1
                          }

    # Formira se nova kolona koja ce imati mapirane vrednosti
    ds_train.loc[:, 'kolona_kreditnog_rizika'] = ds_train["PERFORM_CNS.SCORE.DESCRIPTION"].apply(
        lambda x: MapiranjeCNSKolone[x])
    ds_test.loc[:, 'kolona_kreditnog_rizika'] = ds_test["PERFORM_CNS.SCORE.DESCRIPTION"].apply(
        lambda x: MapiranjeCNSKolone[x])

    # Kolone nad kojim ce se raditi normalizacija
    KolonezaNormalizaciju = [
        'disbursed_amount', 'asset_cost', 'ltv', 'branch_id',
        'supplier_id', 'manufacturer_id', 'Current_pincode_ID',
        'Employment.Type', 'State_ID', 'Employee_code_ID',
        'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag',
        'Passport_flag', 'PERFORM_CNS.SCORE', 'PRI.NO.OF.ACCTS',
        'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE',
        'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS',
        'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',
        'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT',
        'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
        'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES',
        'Age', 'age', 'disbursal_months_passed'
        , 'number_of_0',
        'loan_to_asset_ratio', 'no_of_accts', 'pri_inactive_accts',
        'sec_inactive_accts', 'tot_inactive_accts', 'tot_overdue_accts',
        'tot_current_balance', 'tot_sanctioned_amount', 'tot_disbursed_amount',
        'tot_installment', 'bal_disburse_ratio', 'pri_tenure', 'sec_tenure',
        'disburse_to_sactioned_ratio', 'active_to_inactive_act_ratio',
        'Month', 'Day', 'Month_DD', 'Day_DD', 'Year', 'kolona_kreditnog_rizika'

    ]

    # Normalizacija
    for i in KolonezaNormalizaciju:
        ds_train[i] = (ds_train[i] - min(ds_train[i])) / (max(ds_train[i]) - min(ds_train[i]))
        ds_test[i] = (ds_test[i] - min(ds_test[i])) / (max(ds_test[i]) - min(ds_test[i]))

    # Dropuje se kolona perform cns
    ds_train.drop(labels=['PERFORM_CNS.SCORE.DESCRIPTION'], axis=1, inplace=True)
    ds_test.drop(labels=['PERFORM_CNS.SCORE.DESCRIPTION'], axis=1, inplace=True)

    # Dodeljivanje kolone label u promenljivu outcome, i dropovanje iz ds_traina
    outcome = ds_train['loan_default']

    ds_train.drop(['loan_default'], axis=1, inplace=True)

    # posto je ustanovljeno da su null vrednosti u koloni bal_disburse_ratio, popunicemo polja srednom vrednoscu
    ds_train['bal_disburse_ratio'] = ds_train['bal_disburse_ratio'].fillna(ds_train['bal_disburse_ratio'].mean())
    ds_test['bal_disburse_ratio'] = ds_test['bal_disburse_ratio'].fillna(ds_test['bal_disburse_ratio'].mean())

    # Priprema za podelu trening podataka na train/test
    X = ds_train
    y = outcome

    # Podela na train i test u odnosu 70 prema 30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # PCA

    pca = PCA(n_components=None)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Bitnost atributa
    recnik = {}
    for x in range(0, len(pca.components_)):

        prva_comp = abs(pca.components_[x])
        najveci = 0
        indeks = 0
        for i, a in enumerate(prva_comp):
            if a > najveci:
                najveci = a
                indeks = i

        recnik.update({indeks: najveci})

    recnik_bitan = {k: v for k, v in recnik.items() if v > 0.5}

    # Ovde se vidi da je formirana kolona kreditnog rizika najuticajnija na model
    #for x in recnik_bitan.keys():
        #print(ds_train.columns[x])

    # Light Gradient Boosting
    d_train = lgb.Dataset(X_train, label=y_train)
    params = {}

    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['num_leaves'] = 10
    params['max_depth'] = 10
    params['min_data'] = 50
    params['sub_feature'] = 0.5

    clf = lgb.train(params, d_train, 100)

    pred = clf.predict(X_test)
    for i in range(len(pred)):
        if pred[i] >= .4:  # setting threshold to .4
            pred[i] = 1
        else:
            pred[i] = 0
    print("------------------------------------------------------\n")
    print("Light Gradient Boosting")
    print('r_a_s: ', roc_auc_score(y_test, pred))
    print('f1_score: ', f1_score(y_test, pred))
    print('Preciznost: ', accuracy_score(y_test, pred))
    print('Odziv: ', recall_score(y_test, pred))
    print('Greska pri klasifikaciji: ', 1 - accuracy_score(y_test, pred))
    print("------------------------------------------------------\n")



    xgb = XGBClassifier(use_label_encoder=False)
    xgb = train_model1("XGB",xgb, X_train, y_train, X_test, y_test)

    # Random Forest
    rfc = RandomForestClassifier(n_estimators=300,
                                 max_depth=12,
                                 n_jobs=-1)
    rfc = train_model1("RandomForrestClassifier", rfc, X_train, y_train, X_test, y_test)

    # Logistical Regression
    lr = LogisticRegression(max_iter=1000)
    lr = train_model1("Logistical regression", lr, X_train, y_train, X_test, y_test)

    # ADA BOOSTING
    abc = AdaBoostClassifier(n_estimators=50,
                             learning_rate=1)

    abc = train_model1("ADABoostingClassifier", abc, X_train, y_train, X_test, y_test)






    # VOTING ensemble
    kfold = model_selection.KFold(n_splits=10)
    estimators = []

    model1 = RandomForestClassifier()
    estimators.append(('rfc', model1))

    model2 = DecisionTreeClassifier()
    estimators.append(('dtc', model2))

    model3 = XGBClassifier(use_label_encoder=False)
    estimators.append(('xgb', model3))

    ensemble = VotingClassifier(estimators)
    results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)
    print("------------------------------------------------------\n")
    print("Voting ensmeble")
    print('Preciznost: ', results.mean())
    ce = results.mean()
    print('Greska pri klasifikaciji: ', (1.00 - ce))
    print("------------------------------------------------------\n")


    # KNN
    knn = KNeighborsClassifier(n_jobs=5, n_neighbors=13, metric='euclidean')
    knn.fit(X_train, y_train)
    knn.predict(X_test)
    print("------------------------------------------------------\n")
    print("KNN")
    print('r_a_s: ', roc_auc_score(y_test, pred))
    print('f1_score: ', f1_score(y_test, pred))
    print('Preciznost: ', accuracy_score(y_test, pred))
    print('Odziv: ', recall_score(y_test, pred))
    print('Greska pri klasifikaciji: ', 1 - accuracy_score(y_test, pred))
    print("------------------------------------------------------\n")

    # SVM
    svc = SVC(kernel='linear', random_state=0, probability=True)
    svc = train_model1("SVM", svc, X_train, y_train, X_test, y_test)










