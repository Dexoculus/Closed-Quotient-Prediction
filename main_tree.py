from Module.tree_trainer import *

def main_XGB():
    feature_list = ['Raw', 'Pitch', 'MFCC', 'Chroma', 'ZCR', 'Energy', 'Spectrogram']
    train_XGB(max_depth=4,
              mcw=2,
              eta=0.05,
              gamma=0.2,
              sub=0.8,
              bytree=0.8,
              lam=1,
              alpha=0,
              seed=42,
              rounds=1000,
              feature=feature_list[2])

def main_forest():
    feature_list = ['Raw', 'Pitch', 'MFCC', 'Chroma', 'ZCR', 'Energy', 'Spectrogram']
    train_RandomForest(
        n_estimators=400,
        max_depth=80,
        feature=feature_list[2]
    )

def main():
    main_XGB()
    main_forest()

if __name__=='__main__':
    main()