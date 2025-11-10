
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from typing import Dict, List
from pathlib import Path


class VirtualAssistant:
    def __init__(
        self,
        sequence_model_path: str = None,
        token_model_path: str = None,
        use_local: bool = False
    ):
        # Chemins par défaut
        if use_local:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            self.sequence_model_path = sequence_model_path or os.path.join(project_root, "models", "sequence_classifier")
            self.token_model_path = token_model_path or os.path.join(project_root, "models", "token_classifier")
        else:
            self.sequence_model_path = sequence_model_path or "AmirDHOUIB/distilbert-finetuned-query-classifier-5IABD2"
            self.token_model_path = token_model_path or "AmirDHOUIB/distilbert-finetuned-token-classifier-5IABD2"
        
        print(f"Chargement de l'assistant virtuel...")
        print(f"Modèle de séquence: {self.sequence_model_path}")
        print(f"Modèle de tokens: {self.token_model_path}")
        
        try:
            self.sequence_tokenizer = AutoTokenizer.from_pretrained(self.sequence_model_path)
            self.sequence_model = AutoModelForSequenceClassification.from_pretrained(
                self.sequence_model_path
            )
            self.sequence_model.eval()
            print("✓ Modèle de classification de séquence chargé")
        except Exception as e:
            print(f"⚠ Erreur lors du chargement du modèle de séquence: {e}")
            print(f"  Assurez-vous d'avoir entraîné le modèle avec train_sequence_classifier.py")
            self.sequence_model = None
        
        try:
            self.token_tokenizer = AutoTokenizer.from_pretrained(self.token_model_path)
            self.token_model = AutoModelForTokenClassification.from_pretrained(
                self.token_model_path
            )
            self.token_model.eval()
            print("✓ Modèle de classification de tokens chargé")
        except Exception as e:
            print(f"⚠ Erreur lors du chargement du modèle de tokens: {e}")
            print(f"  Assurez-vous d'avoir entraîné le modèle avec train_token_classifier.py")
            self.token_model = None
        
        print("Assistant virtuel initialisé!\n")
    
    def classify_query(self, query: str) -> str:
        """
        Classifier la requête en "question_rag" ou "send_message".
        
        Args:
            query: La requête utilisateur
            
        Returns:
            "question_rag" ou "send_message"
        """
        if self.sequence_model is None:
            send_keywords = ["ask", "send", "message", "text", "write", "tell"]
            if any(keyword in query.lower() for keyword in send_keywords):
                return "send_message"
            return "question_rag"
        
        inputs = self.sequence_tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            logits = self.sequence_model(**inputs).logits
        
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_label = self.sequence_model.config.id2label[predicted_class]
        
        return predicted_label
    
    def extract_person_and_content(self, query: str) -> Dict[str, str]:
        """
        Extraire la personne (receiver) et le contenu (content) d'une requête send_message.
        
        Args:
            query: La requête utilisateur
            
        Returns:
            {"receiver": "...", "content": "..."}
        """
        if self.token_model is None:
            # Fallback simple
            return {
                "receiver": "unknown",
                "content": query
            }
        
        # Tokenizer la requête
        words = query.split()
        inputs = self.token_tokenizer(
            words,
            return_tensors="pt",
            is_split_into_words=True,
            truncation=True,
            max_length=128
        )
        
        # Prédire les labels
        with torch.no_grad():
            logits = self.token_model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)
        
        # Aligner les prédictions avec les mots
        word_labels = []
        word_ids = inputs.word_ids()
        previous_word_idx = None
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                label_id = predictions[0][idx].item()
                label = self.token_model.config.id2label[label_id]
                word_labels.append(label)
                previous_word_idx = word_idx
        
        # Extraire receiver et content
        receiver_words = []
        content_words = []
        
        for word, label in zip(words, word_labels):
            if "PERSON" in label:
                receiver_words.append(word)
            elif "CONTENT" in label:
                content_words.append(word)
        
        receiver = " ".join(receiver_words) if receiver_words else "unknown"
        content = " ".join(content_words) if content_words else query
        
        return {
            "receiver": receiver,
            "content": content
        }
    
    def process_query(self, query: str) -> Dict:
        """
        Traiter une requête utilisateur complète.
        
        Args:
            query: La requête utilisateur
            
        Returns:
            Un dictionnaire contenant la tâche et les informations extraites
        """
        # Classifier la requête
        task_type = self.classify_query(query)
        
        if task_type == "question_rag":
            return {
                "task": "ask_RAG",
                "reply": f"asked_to_rag: {query}"
            }
        else:  # send_message
            parsed = self.extract_person_and_content(query)
            return {
                "task": "send_message",
                "receiver": parsed["receiver"],
                "content": parsed["content"]
            }


# Instance globale de l'assistant (chargée une seule fois)
_assistant_instance = None


def call_virtual_assistant(user_query: str, use_local: bool = True) -> Dict:

    global _assistant_instance
    
    # Charger l'assistant une seule fois
    if _assistant_instance is None:
        _assistant_instance = VirtualAssistant(use_local=use_local)
    
    return _assistant_instance.process_query(user_query)


def main():
    """
    Fonction de test pour l'assistant virtuel.
    """
    print("="*70)
    print("VIRTUAL ASSISTANT - TD5 Part 3")
    print("="*70)
    
    # Exemples de test
    test_queries = [
        "Does the React course cover the use of hooks?",
        "What are the prerequisites for machine learning?",
        "Ask the python teacher when is the next class",
        "Send a message to John about the meeting",
        "Text the professor asking about the exam format",
        "How does the cybersecurity course address security risks?",
        "Write to Mom telling her I'll be home late",
    ]
    
    print("\nTest de l'assistant virtuel avec plusieurs requêtes:\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        result = call_virtual_assistant(query)
        print(f"Result: {result}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("Tests terminés!")
    print("="*70)
    
    # Mode interactif
    print("\nMode interactif (tapez 'quit' pour quitter):")
    while True:
        try:
            user_input = input("\nVotre requête: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
            
            result = call_virtual_assistant(user_input)
            print(f"Résultat: {result}")
        
        except KeyboardInterrupt:
            print("\n\nAu revoir!")
            break
        except Exception as e:
            print(f"Erreur: {e}")


if __name__ == "__main__":
    main()

