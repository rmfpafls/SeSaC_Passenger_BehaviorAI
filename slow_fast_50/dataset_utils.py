import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_merge_csv(directory_path):
    # 하위 폴더까지 포함하여 모든 .csv 파일 탐색
    all_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(directory_path)
        for file in files if file.endswith('.csv')
    ]
    print("csv file loaded counts  ",len(all_files))
    # 모든 파일을 DataFrame으로 읽기
    dataframes = [pd.read_csv(file) for file in all_files]

    # DataFrame 병합
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    return merged_df


def filter_minority_classes(df: pd.DataFrame, stratify_column: str, min_samples: int=10):
    """
    부족한 레이블을 확인하고 해당 데이터를 삭제하는 함수.

    Parameters:
    - df: pandas DataFrame
    - stratify_column: Stratify 기준이 되는 레이블 컬럼
    - min_samples: 각 클래스에 최소 필요한 샘플 개수

    Returns:
    - df: 부족한 레이블이 제거된 데이터프레임
    """
    label_counts = df[stratify_column].value_counts()
    print(f"Before filtering: \n {label_counts}")

    label_count = str(label_counts).split("\n")
    count2 = 0
    for i in label_count:
      b = i.split("  ")
      c = re.findall(r'\d+', str(b))
      
      if len(c) == 2:
        count2 += int(c[1])
    
    print("sum", count2)

    # 부족한 레이블 필터링
    insufficient_labels = label_counts[label_counts < min_samples].index
    print("\nLabels with insufficient samples (removed):")
    for label in insufficient_labels:
        print(f"Label: {label}, Count: {label_counts[label]}")

    # 부족한 레이블 삭제
    filtered_df = df[~df[stratify_column].isin(insufficient_labels)]

    print("\nAfter filtering:")
    print(filtered_df[stratify_column].value_counts())

    return filtered_df


def split_dataset(df, test_size=0.15, val_size=0.15, stratify_column="label_idx"):
    """
    데이터프레임을 train, validation, test로 분할.
    부족한 레이블은 필터링하여 Stratify 조건을 만족시킴.
    """
    if stratify_column in df:
        # 부족한 레이블 필터링
        df = filter_minority_classes(df, stratify_column=stratify_column, min_samples=28)
        stratify = df[stratify_column]
    else:
        stratify = None

    # Train/Test 비율 계산
    train_size = 1 - test_size - val_size

    # Train + Test로 분할
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_size), stratify=stratify, random_state=42
    )

    # Validation/Test 비율 계산
    temp_stratify = temp_df[stratify_column] if stratify_column in temp_df else None
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (test_size + val_size), stratify=temp_stratify, random_state=42
    )

    return train_df, val_df, test_df

# 매핑 관련 함수
def mapping_label(label_map_path):
    label_map = pd.read_csv(label_map_path)
    index_to_action = dict(zip(label_map['index'], label_map['midlevel_activity']))
    return index_to_action


if __name__ == '__main__':
    directory_path = "/home/elicer/kinect_color/"
    merged_df = load_and_merge_csv(directory_path)
    print('merged_df length ', merged_df.shape)

    # 데이터 나누기
    train_df, val_df, test_df = split_dataset(merged_df, test_size=0.1, val_size=0.2, stratify_column="label_idx")
    print('train_df type', type(train_df))
