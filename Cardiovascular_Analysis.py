#----- import libraries-------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
#------loading function-------#
def load_data(file_path):
    return pd.read_csv(file_path, delimiter=';')

#------finding_primary_information-------#
def finding_primary_information(df_cardio):
    print(f"df_cardio_head(10) =\n  {df_cardio.head(10)}")
    print(f"df_cardio_tail(10) =\n {df_cardio.tail(10)}")
    print(f"df_cardio_10 sample =\n {df_cardio.sample(10)}")
    print(f"df_cardio_shape =\n {df_cardio.shape}")
    print(f"df_cardio_dimention =\n {df_cardio.ndim}") 
    print("Dataset Information  befor data manipulation = ")
    df_cardio.info()
    print(f"df cardio statistical befor data manipulation: \n {df_cardio.describe()}")

#--------dropping useless columns--------#
def drop_columns(df_cardio):
    df_cardio = df_cardio.drop(columns = ["id"])
    print(df_cardio.columns)
    print("Dataset Information after droping= ")
    df_cardio.info()
    return df_cardio

#------- finding nulls in the dataset-------#
def null(df_cardio):
    nan_df_cardio = df_cardio.isna()
    sum_nan_df_cardio = df_cardio.isna().sum()
    print(nan_df_cardio)
    print(f"sum of null =\n {sum_nan_df_cardio}")
    return df_cardio

#------- finding Duplicated and removing duplicated in the dataset-------#
def duplicated_data(df_cardio):
    duplicated_df_cardio = df_cardio.duplicated()
    sum_duplicated_df_cardio = df_cardio.duplicated().sum()
    print(duplicated_df_cardio)
    print(f"sum of duplication = {sum_duplicated_df_cardio}")
    duplicates = df_cardio[df_cardio.duplicated()]
    print(duplicates.head(24))

#------removing duplicated in the dataset-------#
def removing_duplicates(df_cardio):
    print(f"dataset shape befor removing duplicates = {df_cardio.shape} ")
    duplicates_index = df_cardio[df_cardio.duplicated()].index
    df_cardio = df_cardio.drop(duplicates_index)
    print(f"dataset shape after removing duplicates = {df_cardio.shape} ")
    return df_cardio

#---------changing age(day) to age(year)--------#
def change_day_to_year (df_cardio):
    df_cardio["age"] = (df_cardio["age"]/365)
    return df_cardio

#---------Changing Data Types---------#
def changing_datatypes(df_cardio):
    df_cardio_col=df_cardio.columns
    print(df_cardio_col)
    for col in df_cardio_col:
        if col == "age" or col=="weight":
            df_cardio[col] = df_cardio[col].astype("float32")
        elif col == "ap_hi" or col=="ap_lo":
            df_cardio[col] = df_cardio[col].astype("int16")
        else:
            df_cardio[col] = df_cardio[col].astype("uint8")
    return df_cardio

#------- statistical and information------#
def statistical_and_information(df_cardio): 
    print("Dataset Information = ")
    df_cardio.info()
    print(f"df_cardio_statistical: \n {df_cardio.describe()}")

#-------histogram plot--------#
def hist_data(df_cardio):
    numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
    plt.figure(figsize=(15, 7))
    for col in range (len(numerical_columns)):
        plt.subplot(3 , 2 , col+1)
        sns.histplot(df_cardio , x = numerical_columns[col])
        plt.title(f"Histogram Plot of {numerical_columns[col]}")
    plt.tight_layout()
    plt.show()

#-------box plot--------# 
def box_plot(df_cardio ):
      numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
      plt.figure(figsize=(15, 7))
      for col in range(len(numerical_columns)):
          plt.subplot(3,2, col+1)
          sns.boxplot(df_cardio , y=numerical_columns[col])
          plt.title(f"Box Plot of {numerical_columns[col]}")
      plt.tight_layout()
      plt.show()    

#-----calculating outliers with IQR approach-----#
def IQR_calculat(df_cardio):
    upper_bond={}
    lower_bond={}
    IQR_val={}
    numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
    for cols in numerical_columns:
        Q1= np.quantile(df_cardio[cols], 0.25 )
        Q3= np.quantile(df_cardio[cols] , 0.75)
        IQR_val[cols]=Q3-Q1
        upper_bond[cols]=Q3+(1.5*IQR_val[cols])
        lower_bond[cols]=Q1-(1.5*IQR_val[cols])
    plt.figure(figsize=(15,7))
    for cols in range(len(numerical_columns)):
        plt.subplot(3,2, cols+1)
        sns.scatterplot(x = df_cardio[numerical_columns[cols]].index , y = df_cardio[numerical_columns[cols]])
        plt.hlines(y = upper_bond[numerical_columns[cols]],
                   xmin=df_cardio.index.min() ,
                   xmax=df_cardio.index.max() , 
                   colors= "red" , 
                   linestyles= "dashed" , 
                   label="Upper bond"
                   )
        plt.hlines(y= lower_bond[numerical_columns[cols]] ,
                    xmin=df_cardio.index.min() ,
                    xmax=df_cardio.index.max() , 
                    colors= "green" , 
                    linestyles= "dashed" , 
                    label="Lower bond"
                    )
        plt.xlabel("Sample Index")
        plt.title(numerical_columns[cols])
        plt.legend()
    plt.tight_layout()
    plt.show()

#----- deleting illogical values in ap_hi and ap_lo in therm of medical science----#
""" Outliers were not removed from the age, height, and weight columns 
because these values may represent real people. 
However, the ap_hi and ap_lo columns contain many illogical or medically impossible values. 
Therefore, these variables will be cleaned according to medical knowledge instead of relying only on the IQR method. """
def deleted_illogical_values(df_cardio):
    numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
    invalid_rows = df_cardio[
    (df_cardio["ap_hi"] < 70) |
    (df_cardio["ap_hi"] > 250) |
    (df_cardio["ap_lo"] < 40) |
    (df_cardio["ap_lo"] > 150) |
    (df_cardio["ap_hi"] <= df_cardio["ap_lo"])]
    print(f"invalid rows =\n {invalid_rows}")
    print(f"shape of invalid rows = {invalid_rows.shape}")
    df_cardio = df_cardio.drop(invalid_rows.index)
          #------ calculating IQR after removing illogical data in medical criteria with calling IQR_calculat function---#
    IQR_calculat(df_cardio)
    print("Dataset Information after removing  illogical ap_hi and ap_lo : ")
    df_cardio.info()
    print(f"df cardio statistical after removing  illogical ap_hi and ap_lo : \n {df_cardio.describe()}")
    return df_cardio

#--------showimg categorical data---------# 
def categorical_df_cardio(df_cardio):
    categorical_columns=["gender" , "cholesterol" , "gluc",  "smoke" , "alco" , "active" , "cardio" ]
    labels = {
        "gender": ["Female", "Male"],
        "cholesterol":["Normal", "Above Normal" , "Well Above Normal"],
        "gluc":["Normal", "Above Normal" , "Well Above Normal"],
        "smoke": ["No", "Yes"],
        "alco": ["No", "Yes"],
        "active": ["No", "Yes"],
        "cardio": ["No Disease", "Disease"]
    }
    plt.figure(figsize=(15,8))
    for cols in range(len(categorical_columns)):
        plt.subplot(4,2, cols+1)
        print(f" value count = {categorical_columns[cols]} \n {df_cardio[categorical_columns[cols]].value_counts()}")
        ax=sns.countplot(data=df_cardio, x=categorical_columns[cols] , color = "red")
        ax.set_xticklabels(labels[categorical_columns[cols]])
        for container in ax.containers:
            ax.bar_label(container, fontsize=8)
        plt.title(f"Count Plot of {categorical_columns[cols]}")
    plt.tight_layout()
    plt.show()
    return df_cardio

#------ pairplot------#
#def pairplot_df_cardio(df_cardio):
    #numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
    #categorical_columns=["gender" , "cholesterol" , "gluc",  "smoke" , "alco" , "active" , "cardio" ]
    #for col_cat in categorical_columns:
        #sns.pairplot(df_cardio , vars = numerical_columns , hue = col_cat)
        #plt.suptitle(f"Pair Plot by {col_cat}", y=1.02)
        #plt.show()

#---------- calculating corrolation with heatmap-------#
def corr_df_cardio(df_cardio):
    corr_df_cardio= df_cardio.corr()
    plt.figure(figsize=(10,15))
    sns.heatmap(corr_df_cardio , annot = True , fmt = "0.3f" , cmap = "Reds")
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 0)
    plt.show()
    return corr_df_cardio

#--------relationship beetween categorical data values-------#
def relationship_categorical_values(df_cardio):
    columns_categorical_data=["gender" , "cholesterol" , "gluc",  "smoke" , "alco" , "active" , "cardio" ]
    labels = {
        "gender": ["Female", "Male"],
        "cholesterol":["Normal", "Above Normal" , "Well Above Normal"],
        "gluc":["Normal", "Above Normal" , "Well Above Normal"],
        "smoke": ["No", "Yes"],
        "alco": ["No", "Yes"],
        "active": ["No", "Yes"],
        "cardio": ["No Disease", "Disease"]
    }
    plt.figure(figsize=(15,8))
    for col in columns_categorical_data:
        if col != "cardio":
            group_values=df_cardio.groupby(col)["cardio"].value_counts()
            print(f"droup of {col} and cardio =\n {group_values}")
            table = pd.crosstab(df_cardio[col] , df_cardio["cardio"] , normalize="index") * 100
            print(table.round(3))
            ax = sns.countplot(data=df_cardio, x=col, hue="cardio", stat="percent", palette="Set2")
            ax.set_xticklabels(labels[col])
            handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(handles , ["No Disease", "Disease"] , title="Cardiovascular")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.3f%%")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.title(f"{col} vs Cardiovascular Disease")
            plt.tight_layout()
            plt.show()
        else:
            continue

#------ numerical data analysis distribution-------#
def numerical_df_cardio(df_cardio):
    numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
    for col in numerical_columns:
        plt.figure(figsize=(15,8))
        sns.kdeplot(data= df_cardio , x = col , hue = "cardio")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"{col} vs Cardiovascular Disease")
        plt.legend(["No Disease", "Disease"])
        plt.tight_layout()
        plt.show()

#--------- comparing between no disease and disease---------#
def compare_numerical_df_cardio(df_cardio):
    numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
    for col in numerical_columns:
        plt.figure(figsize=(15,8))
        sns.boxenplot(data= df_cardio , x = "cardio" , y = col ,palette="Set2")
        plt.xticks([0,1], ["No Disease","Disease"])
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"{col} vs Cardiovascular Disease")
        plt.tight_layout()
        plt.show()

#------- shapiro-wilk for normality test in numerical data--------#
def shapiro_wilk(df_cardio):
    #H0 = data fallow normal distribution 
    #H1 = data do not fallow normal distribution
    numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
    colors = {
        "age": "blue", 
        "height": "green",
        "weight": "orange",
        "ap_hi": "red",
        "ap_lo": "purple"
        }
        
    for  col in numerical_columns:
        statistic , p_value = stats.shapiro(df_cardio[col])
        print(f"column = {col}")
        print(f"statistic = {statistic:.5f}")
        print(f"p_value = {p_value:.5f}")
        if p_value > 0.05:
            print("Normal Distribution")
        else:
            print("Not Normal Distribution")
        plt.figure(figsize=(10,8))
        sns.histplot(df_cardio[col], kde=True , color = colors[col])
        plt.show()

#------- chi_square_test categorical data--------#
def chi_square_test(df_cardio):
    categorical_data=["gender" , "cholesterol" , "gluc",  "smoke" , "alco" , "active"]
    labels = {
        "gender": ["Female", "Male"],
        "cholesterol":["Normal", "Above Normal" , "Well Above Normal"],
        "gluc":["Normal", "Above Normal" , "Well Above Normal"],
        "smoke": ["No", "Yes"],
        "alco": ["No", "Yes"],
        "active": ["No", "Yes"],
        "cardio": ["No Disease", "Disease"]
        }
    for col in categorical_data:
        table_data = pd.crosstab(df_cardio[col] , df_cardio["cardio"])
        print(f"Table {[col]} and Cardio = \n {table_data}")
        chi2 , p_value , dof , expected= chi2_contingency(table_data) 
        print(f" Chi-square = {chi2:.3f}")
        print(f"Degrees of freedom = {dof}")
        print(f"P-value = {p_value:.5f}")
        table_data_percentage = pd.crosstab(df_cardio[col] , df_cardio["cardio"] , normalize = "index")*100
        ax=table_data_percentage.plot(kind = "bar" ,stacked = True , figsize=(10,7) , colormap="viridis")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%")
        ax.set_xticklabels(labels[col], rotation = 0)
        plt.ylabel("Percentage")
        plt.title(f"{col} vs Cardiovascular Disease")
        plt.legend(["No Disease", "Disease"])
        plt.tight_layout()
        plt.show()

#--------Mann-whitney U Test--------#
def mann_whitney_test(df_cardio):
    numerical_columns = ["age", "height", "weight", "ap_hi", "ap_lo"]
    for col in numerical_columns:
        healthy_persons=df_cardio.loc[df_cardio["cardio"] == 0 , col]
        patient_persons=df_cardio.loc[df_cardio["cardio"] == 1 , col]
        u_test , p_value = mannwhitneyu(healthy_persons, patient_persons)
        print(f"U-test = {u_test}")
        print(f"p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("Important difference")
        else:
            print("No Important difference")
#------making BMI ---------#
def BMI_data(df_cardio):
    df_cardio["BMI"] = df_cardio["weight"]/((df_cardio["height"]/100)**2)
    BMI_patient = df_cardio.loc[df_cardio["cardio"] == 1 ,"BMI"]
    BMI_health = df_cardio.loc[df_cardio["cardio"] == 0 , "BMI"] 
    print(f"statistacal parameter for BMI=\n {df_cardio["BMI"].describe()}")
    print(f"statistacal parameter for BMI_patient =\n {BMI_patient.describe()}")
    print(f"statistacal parameter for BMI_health =\n {BMI_health.describe()}")
    range_BMI = [0, 18.5, 25, 30, np.inf]
    label_BMI = ["Underweight", "Normal", "Overweight", "Obese"]
    df_cardio["BMI_category"]=pd.cut(df_cardio["BMI"], bins = range_BMI , labels = label_BMI )
    return df_cardio

#-------ploting MBI data------#
def plot_BMI(df_cardio):
    table = pd.crosstab(df_cardio["BMI_category"], df_cardio["cardio"] , normalize = "index" )*100
    print(table.round(3))
    ax = sns.countplot(data=df_cardio, x="BMI_category", hue="cardio", stat="percent", palette="Set2")
    plt.xlabel("BMI Category")
    ax.legend(["No Disease", "Disease"] , title="Cardiovascular")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f%%")
    plt.xlabel("BMI_category")
    plt.ylabel("percentage(%)")
    plt.title(f"BMI_category vs Cardiovascular Disease")
    plt.tight_layout()
    plt.show()

#------- plot BMI percentage-----#
def plot_percentage(df_cardio):
    table_percentage = pd.crosstab(df_cardio["BMI_category"], df_cardio["cardio"] , normalize = "index" )*100
    ax=table_percentage.plot(kind = "bar" ,stacked = True , figsize=(10,7) , colormap="viridis")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%")
    plt.xticks(rotation = 0)
    plt.xlabel("BMI_category")
    plt.ylabel("percentage(%)")
    plt.legend(["No Disease", "Disease"], title="Cardiovascular")
    plt.title(f"BMI_category vs Cardiovascular Disease")
    plt.tight_layout()
    plt.show()

#------- calling data set-------#
df_cardio=load_data("cardio_train.csv")
#--------calling primary information-------#
finding_primary_information(df_cardio)
#-----dataset manipulating(drop)------#
df_cardio = drop_columns(df_cardio)
#----------finding null in the dataset-------#
null(df_cardio)
#----------finding duplicated data in the dataset-------#
duplicated_data(df_cardio)
#--------deleting duplicats--------#
df_cardio = removing_duplicates(df_cardio)
#--------calling change age(day) to age(year)-----#
df_cardio = change_day_to_year(df_cardio)
#------- cahnging datatypes------#
df_cardio=changing_datatypes(df_cardio)
#------- statistical and information------#
statistical_and_information(df_cardio)
#------- calling histogram plot--------#
hist_data(df_cardio)
#------- calling Box plot--------#
box_plot(df_cardio)
#------calculating IQR for data-----#
IQR_calculat(df_cardio)
#---removing illogical values in ap_hi and ap_lo----#
df_cardio = deleted_illogical_values(df_cardio)
#--------showimg categorical data---------# 
categorical_df_cardio(df_cardio)
#--------pair plots-------#
#pairplot_df_cardio(df_cardio)
#--------- correlation-----#
corr_df_cardio(df_cardio)
#------ groupping  categorical data-------#
relationship_categorical_values(df_cardio)
#------numerical data analysis-------#
numerical_df_cardio(df_cardio)
#--------- comparing between no disease and disease---------#
compare_numerical_df_cardio(df_cardio)
#------- shapiro-wilk for normality test --------#
shapiro_wilk(df_cardio)
#------- chi_square_test categorical data--------#
chi_square_test(df_cardio)
#--------Mann-whitney U Test--------#
mann_whitney_test(df_cardio)
#------making BMI ---------#
BMI_data(df_cardio)
#-------ploting MBI data------#
plot_BMI(df_cardio)
#------- plot BMI percentage-----#
plot_percentage(df_cardio)
#--------saving data-------#
df_cardio.to_csv("clean_cardio.csv", index=False)

