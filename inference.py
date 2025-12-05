"""
추론 모듈
학습된 모델을 사용하여 게임 용어 NER 수행
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification


class GameNERInference:
    """게임 용어 NER 추론 클래스"""
    
    def __init__(self, model_path="models/final_model", label_map_path="processed/label_map.json"):
        """
        초기화
        
        Args:
            model_path: 학습된 모델 경로
            label_map_path: 레이블 맵 파일 경로
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # 모델 로드
        print(f"모델 로딩 중: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            use_safetensors=True
        )
        self.model.eval()
        self.model.to(self.device)
        
        # 레이블 맵 로드
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
        
        print("모델 로드 완료!")
    
    def predict(self, sentence):
        """
        문장에서 게임 용어 추출
        
        Args:
            sentence: 입력 문장
        
        Returns:
            list: 추출된 엔티티 리스트
                [{'term': str, 'label': str, 'confidence': float}, ...]
        """
        # 토큰화
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            probabilities = torch.softmax(outputs.logits, dim=2)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predictions = predictions[0].cpu().numpy()
        probabilities = probabilities[0].cpu().numpy()
        
        # 엔티티 추출
        entities = []
        current_entity = None
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.id2label[pred_id]
            confidence = probabilities[i][pred_id]
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'term': token.replace('##', ''),
                    'label': entity_type,
                    'confidence': confidence
                }
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity['label']:
                    current_entity['term'] += token.replace('##', '')
                    current_entity['confidence'] = (current_entity['confidence'] + confidence) / 2
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, sentences):
        """
        여러 문장에서 게임 용어 추출
        
        Args:
            sentences: 입력 문장 리스트
        
        Returns:
            list: 각 문장의 엔티티 리스트
        """
        results = []
        for sentence in sentences:
            entities = self.predict(sentence)
            results.append({
                'sentence': sentence,
                'entities': entities
            })
        return results
    
    def print_result(self, sentence, entities):
        """
        결과 출력
        
        Args:
            sentence: 입력 문장
            entities: 추출된 엔티티 리스트
        """
        print(f"\n입력: {sentence}")
        print(f"결과: {len(entities)}개 용어 인식")
        if entities:
            for entity in entities:
                print(f"  - {entity['term']} ({entity['label']}, {entity['confidence']:.2%})")
        else:
            print("  (인식된 게임 용어 없음)")


def interactive_mode():
    """대화형 모드 - 사용자 입력 받아 추론"""
    print("\n" + "=" * 60)
    print("게임 용어 NER 추론 (대화형 모드)")
    print("=" * 60)
    
    # 모델 로드
    inferencer = GameNERInference()
    
    print("\n문장을 입력하세요 (종료: 'quit' 또는 'exit')")
    print("-" * 60)
    
    while True:
        sentence = input("\n문장: ").strip()
        
        if sentence.lower() in ['quit', 'exit', 'q']:
            print("종료합니다.")
            break
        
        if not sentence:
            print("문장을 입력해주세요.")
            continue
        
        # 추론
        entities = inferencer.predict(sentence)
        inferencer.print_result(sentence, entities)


def batch_mode(sentences):
    """배치 모드 - 여러 문장 한번에 추론"""
    print("\n" + "=" * 60)
    print("게임 용어 NER 추론 (배치 모드)")
    print("=" * 60)
    
    # 모델 로드
    inferencer = GameNERInference()
    
    # 추론
    results = inferencer.predict_batch(sentences)
    
    # 결과 출력
    for result in results:
        inferencer.print_result(result['sentence'], result['entities'])
    
    print("\n" + "=" * 60)


def main():
    """메인 실행 함수"""
    import sys
    
    if len(sys.argv) > 1:
        # 명령행 인자로 문장 전달
        sentence = ' '.join(sys.argv[1:])
        inferencer = GameNERInference()
        entities = inferencer.predict(sentence)
        inferencer.print_result(sentence, entities)
    else:
        # 대화형 모드
        interactive_mode()


if __name__ == "__main__":
    # 예제 실행
    test_sentences = [
        "퀘스트 사냥꾼은 대체로 컨트롤 성향 덱에 더 강한 모습을 보인다.",
        "라이아 보스는 강력한 공격 패턴을 가지고 있다.",
        "탈진 스킬로 적을 기절시켰다."
    ]
    
    print("=" * 60)
    print("예제 실행")
    print("=" * 60)
    batch_mode(test_sentences)
    
    # 대화형 모드 시작
    print("\n\n")
    interactive_mode()
