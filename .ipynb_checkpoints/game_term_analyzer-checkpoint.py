"""
게임 용어 분석 시스템 (통합 버전)
문장에서 게임 용어를 인식하고 간결하게 해석 결과를 출력합니다.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification


class GameTermAnalyzer:
    """게임 용어 분석기 (NER + 해석 통합)"""
    
    def __init__(self, model_path: str, term_dict_path: str):
        """
        Args:
            model_path: 학습된 NER 모델 경로
            term_dict_path: 용어.json 파일 경로
        """
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 게임 용어 분석 시스템 초기화 중...")
        print(f"   디바이스: {self.device}")
        
        # NER 모델 로드
        self._load_ner_model()
        
        # 용어 사전 로드
        self._load_term_dictionary(term_dict_path)
        
        print(f"✅ 초기화 완료!\n")
    
    def _load_ner_model(self):
        """NER 모델 로드"""
        print(f"   NER 모델 로딩: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 레이블 맵 로드
        label_map_path = self.model_path / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
                self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
        else:
            self.id2label = self.model.config.id2label
        
        print(f"   ✓ NER 모델 로드 완료 (레이블 수: {len(self.id2label)})")
    
    def _load_term_dictionary(self, term_dict_path: str):
        """용어 사전 로드"""
        term_dict_path = Path(term_dict_path)
        print(f"   용어 사전 로딩: {term_dict_path.name}")
        
        if not term_dict_path.exists():
            raise FileNotFoundError(f"용어 사전 파일을 찾을 수 없습니다: {term_dict_path}")
        
        with open(term_dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 용어를 키로 하는 딕셔너리로 변환
        self.terms = {}
        for item in data:
            term = item['term']
            if term not in self.terms:
                self.terms[term] = []
            
            self.terms[term].append({
                'definition': item['definition'],
                'facet': item.get('facet', ''),
                'game': item.get('level3', ''),
            })
        
        print(f"   ✓ 용어 사전 로드 완료 (고유 용어: {len(self.terms):,}개)")
    
    def extract_entities(self, sentence: str, confidence_threshold: float = 0.0):
        """
        문장에서 게임 용어 추출
        
        Args:
            sentence: 입력 문장
            confidence_threshold: 최소 신뢰도 (0.0 ~ 1.0)
            
        Returns:
            추출된 엔티티 리스트
        """
        # 토크나이징
        inputs = self.tokenizer(
            sentence, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            probabilities = torch.softmax(outputs.logits, dim=2)
        
        # 토큰과 예측 결과
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
                if current_entity and current_entity['confidence'] >= confidence_threshold:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'term': token.replace('##', ''),
                    'facet': entity_type,
                    'confidence': confidence
                }
            
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity['facet']:
                    current_entity['term'] += token.replace('##', '')
                    current_entity['confidence'] = (current_entity['confidence'] + confidence) / 2
            
            else:  # 'O' 태그
                if current_entity and current_entity['confidence'] >= confidence_threshold:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity and current_entity['confidence'] >= confidence_threshold:
            entities.append(current_entity)
        
        return entities
    
    def get_definition(self, term: str, facet: Optional[str] = None) -> Optional[str]:
        """
        용어 정의 가져오기
        
        Args:
            term: 용어
            facet: 용어 유형 (필터링용)
            
        Returns:
            용어 정의 또는 None
        """
        interpretations = self.terms.get(term)
        
        if not interpretations:
            return None
        
        # facet으로 필터링
        if facet and len(interpretations) > 1:
            filtered = [i for i in interpretations if i['facet'] == facet]
            if filtered:
                return filtered[0]['definition']
        
        return interpretations[0]['definition']
    
    def analyze(self, sentence: str, confidence_threshold: float = 0.3):
        """
        문장 분석 및 결과 출력
        
        Args:
            sentence: 입력 문장
            confidence_threshold: 최소 신뢰도
        """
        print(f"\n{'='*70}")
        print(f"📝 입력: {sentence}")
        print(f"{'='*70}")
        
        # 용어 추출
        entities = self.extract_entities(sentence, confidence_threshold)
        
        if not entities:
            print("❌ 인식된 게임 용어가 없습니다.")
            return
        
        # 결과 출력
        print(f"✅ 인식된 용어: {len(entities)}개\n")
        
        for idx, entity in enumerate(entities, 1):
            term = entity['term']
            facet = entity['facet']
            confidence = entity['confidence']
            
            print(f"[{idx}] {term}")
            print(f"    유형: {facet}")
            print(f"    신뢰도: {confidence:.1%}")
            
            # 정의 가져오기
            definition = self.get_definition(term, facet)
            if definition:
                print(f"    정의: {definition}")
            else:
                print(f"    정의: (사전에 미등록된 용어)")
            print()
    
    def batch_analyze(self, sentences: List[str], confidence_threshold: float = 0.3):
        """
        여러 문장 일괄 분석
        
        Args:
            sentences: 문장 리스트
            confidence_threshold: 최소 신뢰도
        """
        for i, sentence in enumerate(sentences, 1):
            print(f"\n\n{'#'*70}")
            print(f"# 문장 {i}/{len(sentences)}")
            print(f"{'#'*70}")
            self.analyze(sentence, confidence_threshold)


def main():
    """메인 실행 함수"""
    
    # ========== 경로 설정 ==========
    MODEL_PATH = r"C:\Users\dhkst\OneDrive\문서\GitHub\2025_NLP_project\models\final_model"
    TERM_DICT_PATH = r"C:\Users\dhkst\OneDrive\바탕 화면\내꺼\대\4-2\자연언어처리\160.문화, 게임 콘텐츠 분야 용어 말뭉치\01-1.정식개방데이터\Training\02.라벨링데이터\TL\용어.json"
    
    # ========== 시스템 초기화 ==========
    analyzer = GameTermAnalyzer(MODEL_PATH, TERM_DICT_PATH)
    
    # ========== 예시 1: 단일 문장 분석 ==========
    print("\n" + "="*70)
    print("예시 1: 단일 문장 분석")
    print("="*70)
    
    analyzer.analyze("라이아가 너무 강해서 공략이 어려워요")
    
    # ========== 예시 2: 여러 문장 분석 ==========
    print("\n\n" + "="*70)
    print("예시 2: 여러 문장 일괄 분석")
    print("="*70)
    
    test_sentences = [
        "방반 가격이 너무 비싸요",
        "스모커의 성장 물약을 사용하면 경험치가 7배 올라요",
        "가디언을 잡으려면 불 속성 공격이 필요해요"
    ]
    
    analyzer.batch_analyze(test_sentences)
    
    # ========== 예시 3: 대화형 모드 ==========
    print("\n\n" + "="*70)
    print("예시 3: 대화형 모드")
    print("="*70)
    print("\n게임 용어가 포함된 문장을 입력하세요")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요")
    print("="*70)
    
    while True:
        try:
            user_input = input("\n📝 문장 입력: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 프로그램을 종료합니다.")
                break
            
            if not user_input:
                print("⚠️  문장을 입력해주세요.")
                continue
            
            analyzer.analyze(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
