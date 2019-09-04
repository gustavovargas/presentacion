# -*- coding: utf-8 -*-
"""utils
Archivo con funciones auxiliares para la ejecución de scripts de
interpretabilidad en modelos de riesgo de crédito
"""

__author__ = "Gustavo Vargas"
__copyright__ = "Copyright 2019, TFM Afi Escuela de Finanzas"
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Gustavo Vargas"
__email__ = "ge.vargasn@gmail.com"

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

import category_encoders as ce

class LoansTransformer(BaseEstimator):
    
    def __init__(self):
        # guardamos ciertos parámetros como parámetros públicos

        self.del_columns = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv',
                            'installment', 'grade', 'emp_title', 'pymnt_plan', 'url',
                            'desc', 'zip_code', 'mths_since_last_delinq', 'mths_since_last_record',
                            'out_prncp_inv', 'total_pymnt_inv', 'next_pymnt_d',
                            'mths_since_last_major_derog', 'policy_code', 'annual_inc_joint',
                            'dti_joint', 'verification_status_joint', 'tot_coll_amt',
                            'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
                            'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
                            'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
                            'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
                            'title', 'application_type']

        self.categorical_features = ['home_ownership', 'verification_status', 'purpose', 'addr_state']
        
    def fit(self, X):
        # Asumimos que X es un DataFrame
        self._columns = X.columns.values

        # sub_grades
        grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        sub_grades = ['1', '2', '3', '4', '5']
        cat_sub_grade = [j+i for j in grades for i in sub_grades]
        levels = np.arange(0, len(cat_sub_grade))
        self.dict_grades = dict(zip(cat_sub_grade, levels))

        # issue_d. Hay 103 meses distintos en issue_d. Repensar si se tiene que subir a producción.
        values_issue = np.arange(0, 103)
        months_issue =  pd.date_range(pd.to_datetime('2007-06-01 00:00:00'), pd.to_datetime('2015-12-01 00:00:00'), freq='MS')
        self.dict_issue = dict(zip(months_issue, values_issue))

        # earliest_cr_line: falta por hacer
        months_earl =  pd.date_range(pd.to_datetime('1944-01-01 00:00:00'), pd.to_datetime('2012-11-01 00:00:00'), freq='MS')
        values_earl = np.arange(0,len(months_earl))
        self.dict_earliest = dict(zip(months_earl, values_earl))

        # last_payment
        months_last =  pd.date_range(pd.to_datetime('2007-12-01 00:00:00'), pd.to_datetime('2016-01-01 00:00:00'), freq='MS')
        values_last = np.arange(0,len(months_last))
        self.dict_last = dict(zip(months_last, values_last))

        # last_credit
        months_credit =  pd.date_range(pd.to_datetime('2007-05-01 00:00:00'), pd.to_datetime('2016-01-01 00:00:00'), freq='MS')
        values_credit = np.arange(0,len(months_credit))
        self.dict_credit = dict(zip(months_credit, values_credit))

        # One Hot Encoder
        # self.le =  ce.OneHotEncoder(cols = self.categorical_features, return_df=True, handle_unknown="ignore")
        self.le =  ce.OrdinalEncoder(cols = self.categorical_features, return_df=True, handle_unknown="ignore")
        self.le.fit(X)

        return self
        
    def transform(self, X):
        # comprobamos que tiene las mismas columnas que el DataFrame con el que hicimos el fit
        if set(self._columns) != set(X.columns):
            raise ValueError('Las columnas de este DataFrame son distintas de las que se hicieron en el fit')
        elif len(self._columns) != len(X.columns):
            raise ValueError('El número de columnas de este DataFrame es distinto del número con el que se hizo el fit')
        
        # One Hot Encoder. Ojo: tenemos que hacerlo antes de eliminar columnas
#         df = self.le.transform(X)

        # Eliminamos columnas no útiles y filas con na
        df = X.drop(self.del_columns, axis=1)
        df = df.dropna()

        # term
        df.term = X.term.apply(lambda x: 0 if x == ' 36 months' else 1)

        # sub_grade
        # Ya que hay orden, vamos a hacer numéricas las categorías que tenemos
        df.sub_grade = X.sub_grade.apply(lambda x: self.dict_grades[x])

        # emp_length
        df.emp_length = df.emp_length.apply(lambda x: '0 years' if x =='< 1 year' else x)
        df.emp_length = df.emp_length.str.extract(r'(.*\d+)')
        df.emp_length = df.emp_length.apply(int)

        # issue_d
        df.issue_d = pd.to_datetime(df.issue_d, format='%b-%Y')
        df.issue_d = df.issue_d.apply(lambda x: self.dict_issue[x])

        # loan_status
        df.loan_status = df.loan_status.apply(lambda x: 'Fully Paid' if x == 'Current' or x == 'Fully Paid' else 'Default')
        df.loan_status = df.loan_status.apply(lambda x: 0 if x == 'Fully Paid' else 1)

        # dti
        df = df[df.dti < 300]

        # earliest_cr_line: Pasamos las fechas a una variable continua
        df.earliest_cr_line = pd.to_datetime(df.earliest_cr_line, format='%b-%Y')
        df.earliest_cr_line = df.earliest_cr_line.apply(lambda x: self.dict_earliest[x])

        # initial_list_status
        df.initial_list_status = df.initial_list_status.apply(lambda x: 0 if x in ['w'] else 1)

        # last_payment
        # Pasamos las fechas a una variable continua
        df.last_pymnt_d = pd.to_datetime(df.last_pymnt_d, format='%b-%Y')
        df.last_pymnt_d = df.last_pymnt_d.apply(lambda x: self.dict_last[x])

        # last_credit_pull_d
        df.last_credit_pull_d = pd.to_datetime(df.last_credit_pull_d, format='%b-%Y')
        df.last_credit_pull_d = df.last_credit_pull_d.apply(lambda x: self.dict_credit[x])

        # Obtenemos la X y la y
        target = df.loan_status
        features = df.drop('loan_status', axis=1)
        return features, target
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def LimeFormat(X, categorical_names, col_names=None, invert=False):
    # If the data isn't a dataframe, we need to be able to build it
    if not isinstance(X, pd.DataFrame):
        X_lime = pd.DataFrame(X, columns=col_names)
    else:
        X_lime = X.copy()
    for k, v in categorical_names.items():
        if not invert:
            label_map = {str_label: int_label for int_label, str_label in enumerate(v)}
        else:
            label_map = { int_label: str_label for int_label, str_label in enumerate(v)}
        X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)

    return X_lime
