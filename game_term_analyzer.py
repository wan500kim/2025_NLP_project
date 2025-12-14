import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification


class GameTermAnalyzer:
    
    def __init__(self, model_path: str, term_dict_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f" 게임 용어 분석 시스템 초기화 중...")
        print(f"   디바이스: {self.device}")
        
        self._load_ner_model()     
        self._load_term_dictionary(term_dict_path)
        
        print(f" 초기화 완료!\n")
    
    def _load_ner_model(self):
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
        
        print(f"    NER 모델 로드 완료 (레이블 수: {len(self.id2label)})")
    
    def _load_term_dictionary(self, term_dict_path: str):
        term_dict_path = Path(term_dict_path)
        print(f"   용어 사전 로딩: {term_dict_path.name}")
        
        if not term_dict_path.exists():
            raise FileNotFoundError(f"용어 사전 파일을 찾을 수 없습니다: {term_dict_path}")
        
        with open(term_dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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
        
        print(f"   용어 사전 로드 완료 (고유 용어: {len(self.terms):,}개)")
    
    def extract_entities(self, sentence: str, confidence_threshold: float = 0.0):
        inputs = self.tokenizer(
            sentence, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            probabilities = torch.softmax(outputs.logits, dim=2)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predictions = predictions[0].cpu().numpy()
        probabilities = probabilities[0].cpu().numpy()
        
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
            
            else:
                if current_entity and current_entity['confidence'] >= confidence_threshold:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity and current_entity['confidence'] >= confidence_threshold:
            entities.append(current_entity)
        
        return entities
    
    def get_definition(self, term: str, facet: Optional[str] = None) -> Optional[str]:
        interpretations = self.terms.get(term)
        
        if not interpretations:
            return None
        
        if facet and len(interpretations) > 1:
            filtered = [i for i in interpretations if i['facet'] == facet]
            if filtered:
                return filtered[0]['definition']
        
        return interpretations[0]['definition']
    
    def analyze(self, sentence: str, confidence_threshold: float = 0.3):
        print(f"\n{'='*70}")
        print(f"📝 입력: {sentence}")
        print(f"{'='*70}")
        
        entities = self.extract_entities(sentence, confidence_threshold)
        
        if not entities:
            print(" 인식된 게임 용어가 없습니다.")
            return
        
        print(f" 인식된 용어: {len(entities)}개\n")
        
        for idx, entity in enumerate(entities, 1):
            term = entity['term']
            facet = entity['facet']
            confidence = entity['confidence']
            
            print(f"[{idx}] {term}")
            print(f"    유형: {facet}")
            print(f"    신뢰도: {confidence:.1%}")
            
            definition = self.get_definition(term, facet)
            if definition:
                print(f"    정의: {definition}")
            else:
                print(f"    정의: (사전에 미등록된 용어)")
            print()
    
    def batch_analyze(self, sentences: List[str], confidence_threshold: float = 0.3):
        for i, sentence in enumerate(sentences, 1):
            print(f"\n\n{'#'*70}")
            print(f"# 문장 {i}/{len(sentences)}")
            print(f"{'#'*70}")
            self.analyze(sentence, confidence_threshold)


def main():
    MODEL_PATH = Path("models/final_model")
    TERM_DICT_PATH = Path("dataset/용어.json")
    analyzer = GameTermAnalyzer(MODEL_PATH, TERM_DICT_PATH)
    
    print("\n게임 용어가 포함된 문장을 입력하세요")
    print("종료하려면 q를 입력하세요")
    
    while True:
        try:
            user_input = input("\n 문장 입력: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n 프로그램을 종료합니다.")
                break
            
            if not user_input:
                print("  문장을 입력해주세요.")
                continue
            
            analyzer.analyze(user_input)
            
        except KeyboardInterrupt:
            print("\n\n 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n 오류 발생: {e}")


if __name__ == "__main__":
    main()