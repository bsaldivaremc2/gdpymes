from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import numpy as np

def split_data(X, y=None, 
               split_type="train_test", 
               test_size=0.2, val_size=0.2, 
               n_splits=5, 
               stratify=None, 
               random_state=42):
    """
    Flexible data splitting function for different scenarios:
    
    Parameters
    ----------
    X : array-like
        Features dataset (numpy array, pandas DataFrame, or similar).
    y : array-like, optional
        Target labels for stratified splitting (default=None).
    split_type : str, default="train_test"
        Type of split to perform. Options:
        - "train_test" : simple train/test split
        - "train_val_test" : split into train, validation, and test sets
        - "train_val_kfold" : split into train/val for k-fold cross-validation
        - "train_val_test_kfold" : first split test, then split train into k-folds
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (used in train_test and train_val_test).
    val_size : float, default=0.2
        Proportion of the remaining dataset to use for validation (used in train_val_test and k-fold splits).
    n_splits : int, default=5
        Number of folds for K-Fold cross-validation (used in k-fold splits).
    stratify : array-like, optional
        Class labels for stratified splitting (default=None). Can be multiclass.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    Depending on `split_type`:
    - "train_test" : X_train, X_test, y_train, y_test
    - "train_val_test" : X_train, X_val, X_test, y_train, y_val, y_test
    - "train_val_kfold" : list of (X_train_fold, X_val_fold, y_train_fold, y_val_fold)
    - "train_val_test_kfold" : X_test, y_test, list of (X_train_fold, X_val_fold, y_train_fold, y_val_fold)
    
    Notes
    -----
    - For stratified splits, provide the `stratify` argument with class labels.
    - KFold splits do not shuffle by default unless `shuffle=True` in StratifiedKFold/KFold.
    """
    
    if split_type == "train_test":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        return X_train, X_test, y_train, y_test
    
    elif split_type == "train_val_test":
        # First split out the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        # Determine stratify vector for train/val split
        stratify_val = y_temp if stratify is not None else None
        
        # Split remaining data into train and val
        val_ratio = val_size / (1 - test_size)  # adjust val_size relative to remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=stratify_val
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    elif split_type == "train_val_kfold":
        if stratify is not None:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        folds = []
        for train_idx, val_idx in kf.split(X, stratify):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx] if y is not None else (None, None)
            folds.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))
        return folds
    
    elif split_type == "train_val_test_kfold":
        # Split out test set first
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        stratify_val = y_temp if stratify is not None else None
        
        # K-fold split on the remaining train data
        if stratify is not None:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        folds = []
        for train_idx, val_idx in kf.split(X_temp, stratify_val):
            X_train_fold, X_val_fold = X_temp[train_idx], X_temp[val_idx]
            y_train_fold, y_val_fold = y_temp[train_idx], y_temp[val_idx] if y is not None else (None, None)
            folds.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))
        
        return X_test, y_test, folds
    
    else:
        raise ValueError(f"Invalid split_type: {split_type}. Choose from ['train_test', 'train_val_test', 'train_val_kfold', 'train_val_test_kfold'].")


# ------------------------------
# Example usage:
# ------------------------------
# X_train, X_test, y_train, y_test = split_data(X, y, split_type="train_test")
# X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, split_type="train_val_test", stratify=y)
# folds = split_data(X, y, split_type="train_val_kfold", n_splits=5, stratify=y)
# X_test, y_test, folds = split_data(X, y, split_type="train_val_test_kfold", n_splits=5, stratify=y)