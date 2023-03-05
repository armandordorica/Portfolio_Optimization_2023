import pandas as pd
import numpy as np
import seaborn as sns


def get_categorical_rate_df(df, var_name):
    """
    Given a dataframe and a variable name to specify a column in that dataframe, it will return a new dataframe
    with number of scans for each of the values of the variable name as well as percentage over total.
    """
    col_name = df.groupby([var_name]).count().columns[0]

    temp_df = df.groupby([var_name])[[col_name]].count()
    temp_df['total_scans'] = temp_df[col_name].sum()
    temp_df['pct_over_total'] = temp_df[col_name] / temp_df['total_scans']

    temp_df.rename(columns={temp_df.columns[0]: "num_entries"}, inplace=True)

    # col_names = list(temp_df.columns)
    # col_names[0] = 'num_entries'
    # temp_df.columns = col_names
    temp_df.sort_values(by='pct_over_total', ascending=False, inplace=True)

    return temp_df



def get_important_fields_df(df, var_name, threshold=100):
    """
    Given a dataframe and a variable name corresponding to a column name, it returns a subset of the dataframe
    where each of the entries belong to a popular variable that occurs more than `threshold` times.
    """
    
    temp_df = df.groupby([var_name]).size().reset_index(name='count')
    temp_df2 = temp_df[temp_df['count'] > threshold]

    main_values = list(temp_df2[var_name])

    return df[df[var_name].isin(main_values)]



def get_correlation_df(df, var_name, target_var='fraud', min_correlation=0.05):
    """
    Calculates the correlation between a specific the values of a specific column name in a dataframe and the
    target_var such as 'fraud'. It outputs only the values that have a correlation higher than `min_correlation`.
    """

    df2 = pd.get_dummies(df[[var_name, target_var]], columns=[var_name])

    # x_labels = []
    # labels = list(df2.columns[1:])

    # for i in range(0, len(labels)):
    #     x_labels.append(labels[i].split('_')[-1])

    df3 = df2.corrwith(df2[target_var])[1:].to_frame()
    df3.columns = ['Correlation with {}'.format(target_var)]

    df3 = df3[np.abs(df3['Correlation with {}'.format(target_var)]) > min_correlation]
    df3['var_name'] = var_name

    values = list(df3.index)

    df3['values'] = [value.replace(var_name + "_", "") for value in values]

    return df3

def get_most_important_variables(df, potential_important_vars, threshold_count=100, min_correlation=0.08,
                                 target_var='fraud'):
    """
    Inputs:
    * Base Dataframe
    * List with potential_important_vars

    Output:
    * A dataframe showing the correlation of each of the values from potential_important_vars
    with fraud (target_var) above a threshold (min_correlation) and
    making sure that that value shows up at least X times (threshold_count).
        * Index of output is the value, i.e. country_VNM
        * Columns are `Correlation with fraud`, `var_name`, and `values`.

    Dependent functions:
    * get_important_fields_df()
    * get_correlation_df()
    """
    var_name = potential_important_vars[0]
    df_new_index = df.reset_index()
    
    temp_df = get_important_fields_df(df_new_index, var_name, threshold=threshold_count)

    high_corr_df = get_correlation_df(temp_df, var_name, target_var=target_var,
                                           min_correlation=min_correlation)

    for i in range(1, len(potential_important_vars)):
        var_name = potential_important_vars[i]
        reduced_df = get_important_fields_df(df_new_index, var_name, threshold=threshold_count)
        high_corr_temp_df = get_correlation_df(reduced_df, var_name, target_var=target_var,
                                                    min_correlation=min_correlation)
        high_corr_df = high_corr_df.append(high_corr_temp_df)
        print(var_name)
        print(high_corr_temp_df)

    return high_corr_df


def get_corr_matrix_multiple_actions(input_df, categorical_var, actions, min_correlation=0.005):

    i=0
    a = get_correlation_df(input_df, categorical_var, target_var=f'sum_{actions[i]}',
                                               min_correlation=min_correlation)
    a.sort_values(by=f'Correlation with sum_{actions[i]}', ascending = False, inplace=True) 
    a = a[['values', f'Correlation with sum_{actions[i]}']].reset_index().drop(columns='index')
    
    for i in range(1,len(actions)): 
        b = get_correlation_df(input_df, categorical_var, target_var=f'sum_{actions[i]}',
                                                   min_correlation=min_correlation)
        b.sort_values(by=f'Correlation with sum_{actions[i]}', ascending = False, inplace=True) 
        b = b[['values', f'Correlation with sum_{actions[i]}']].reset_index().drop(columns='index')
        a = a.merge(b, how='outer', on='values')
    
    column_names = [f'Correlation with sum_{x}' for x in actions]
    a.sort_values(by=column_names, ascending = [False, False, True, True], inplace=True)
    
    return a.style.bar(align='mid', color=['red', 'lightgreen'])