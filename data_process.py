"""
데이터 전처리 모듈
원본 JSON 데이터를 로드하여 BIO 태깅 후 train/val/test로 분할하여 저장
"""

import json
import random
from pathlib import Path
from tqdm import tqdm


def load_and_process_data(input_file, max_samples=None):
    """
    JSON 파일을 로드하고 BIO 태깅 처리
    
    Args:
        input_file: 입력 JSON 파일 경로
        max_samples: 최대 샘플 수 (None이면 전체)
    
    Returns:
        processed_data: 처리된 데이터 리스트
        labels: 발견된 레이블 set
    """
    print(f"\n데이터 처리 중: {input_file}")
    processed_data = []
    labels = set(['O'])
    
    print("JSON 파일 로딩 중...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    total_samples = len(data)
    print(f"총 {total_samples:,}개 샘플 발견")
    
    # 데이터 처리
    for example in tqdm(data, desc="Processing examples"):
        sentence = example.get('sentence', '')
        tokens = example.get('tokens', [])
        
        if sentence and tokens:
            char_tags = ['O'] * len(sentence)
            
            for token in tokens:
                start = token['start']
                length = token['length']
                facet = token.get('facet', 'TERM')
                
                if start < len(sentence):
                    char_tags[start] = f'B-{facet}'
                
                for i in range(start + 1, start + length):
                    if i < len(sentence):
                        char_tags[i] = f'I-{facet}'
            
            chars = list(sentence)
            tags = char_tags
            labels.update(tags)
            
            processed_data.append({
                'id': example.get('id'),
                'sentence': sentence,
                'chars': chars,
                'tags': tags,
                'tokens': tokens
            })
    
    print(f"처리 완료: {len(processed_data):,}개 예제")
    return processed_data, labels


def split_data(train_processed_data, val_processed_data=None, train_ratio=0.9):
    """
    데이터를 train/val/test로 분할
    
    Args:
        train_processed_data: 학습 데이터
        val_processed_data: 검증 데이터 (None이면 train에서 분할)
        train_ratio: 학습 데이터 비율 (val_processed_data가 있을 때)
    
    Returns:
        train_data, val_data, test_data
    """
    if val_processed_data:
        # 별도 검증 데이터가 있으면 학습 데이터에서 테스트만 분할
        random.shuffle(train_processed_data)
        train_size = int(len(train_processed_data) * train_ratio)
        
        train_data = train_processed_data[:train_size]
        test_data = train_processed_data[train_size:]
        val_data = val_processed_data
    else:
        # 검증 데이터가 없으면 train에서 모두 분할
        random.shuffle(train_processed_data)
        total = len(train_processed_data)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        
        train_data = train_processed_data[:train_size]
        val_data = train_processed_data[train_size:train_size + val_size]
        test_data = train_processed_data[train_size + val_size:]
    
    print(f"\n데이터 분할 완료:")
    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val: {len(val_data)} samples")
    print(f"  - Test: {len(test_data)} samples")
    
    return train_data, val_data, test_data


def save_processed_data(train_data, val_data, test_data, labels, output_dir="processed"):
    """
    전처리된 데이터와 레이블 맵 저장
    
    Args:
        train_data, val_data, test_data: 분할된 데이터
        labels: 레이블 set
        output_dir: 출력 디렉토리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 데이터 저장
    with open(output_path / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(output_path / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(output_path / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 레이블 맵 저장
    label2id = {label: idx for idx, label in enumerate(sorted(labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    
    label_map = {
        'label2id': label2id,
        'id2label': id2label,
        'num_labels': len(labels)
    }
    
    with open(output_path / "label_map.json", 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    print(f"\n전처리 완료! 데이터 저장됨: {output_dir}/")


def main():
    """메인 실행 함수"""
    # 설정
    train_input_file = Path("dataset/TL/용례_게임tl.json")
    val_input_file = Path("dataset/VL/용례_게임vl.json")
    
    # 학습 데이터 처리
    train_processed_data, labels = load_and_process_data(train_input_file)
    
    # 검증 데이터 처리 (파일이 있으면)
    val_processed_data = None
    if val_input_file.exists():
        val_processed_data, val_labels = load_and_process_data(val_input_file)
        labels.update(val_labels)
    
    # 데이터 분할
    train_data, val_data, test_data = split_data(train_processed_data, val_processed_data)
    
    # 저장
    save_processed_data(train_data, val_data, test_data, labels)
    
    print("\n" + "=" * 60)
    print("전처리 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
