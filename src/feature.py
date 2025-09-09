from sklearn.feature_extraction.text import CountVectorizer

def make_features(df, vectorizer=None, fit_vectorizer=True):
    if 'is_comic' in df.columns:
        y = df["is_comic"]
    else:
        y = None
    
    if vectorizer is None:
        vectorizer = CountVectorizer(
            lowercase=True,  
            stop_words='english',  
            min_df=2,  
            max_df=0.95,  
        )
    
    # Transformer les titres de vidéos
    if fit_vectorizer:
        X = vectorizer.fit_transform(df["video_name"])
    else:
        X = vectorizer.transform(df["video_name"])
    
    return X, y, vectorizer
