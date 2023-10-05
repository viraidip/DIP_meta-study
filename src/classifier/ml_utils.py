

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_y(df):
    y = list()
    median = df["NGS_log_norm"].median()

    for row in df.iterrows():
        r = row[1]
        y.append("low" if r["NGS_log_norm"] < median else "high")

    print(y.count("low"))
    print(y.count("high"))

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)   

    return y

def preprocessing(df):
    '''
    
    '''
    # Separate features (X) and target (y)
    X = df.drop(["Unnamed: 0", "Segment", "NGS_read_count", "class", "dataset_name","Strain","NGS_log","NGS_norm","NGS_log_norm","int_dup","Duplicate","comb_dup"], axis=1).copy()
    y = prepare_y(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (optional but recommended)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test