import spacy
import networkx as nx
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self):
        try:
            self.graph = nx.DiGraph()
            self.nlp = spacy.load("zh_core_web_sm")
            logger.info("KnowledgeGraph initialized with spaCy zh_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraph: {e}")
            raise

    def extract_entities_and_relations(self, text: str, chunk_id: str, format_type: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        try:
            doc = self.nlp(text)
            entities = []
            relations = []
            entity_texts = set()

            # Extract entities using NER
            for ent in doc.ents:
                entities.append((ent.text, ent.label_))
                entity_texts.add(ent.text)
                self.graph.add_node(ent.text, type=ent.label_, chunk_id=chunk_id)

            # Format-specific relation extraction
            if format_type == 'pdf':
                # PDF: Verb-based and fallback co-occurrence
                for sent in doc.sents:
                    for token in sent:
                        if token.pos_ in ("VERB", "AUX"):
                            subj = None
                            obj = None
                            for child in token.children:
                                if child.dep_ in ("nsubj", "nsubjpass"):
                                    subj = child.text
                                if child.dep_ in ("dobj", "attr", "obl"):
                                    obj = child.text
                            if subj and obj and subj in entity_texts and obj in entity_texts:
                                relations.append((subj, token.text, obj))
                                self.graph.add_edge(subj, obj, relation=token.text, weight=1.0)
                    # Fallback: Link co-occurring entities
                    sent_entities = [ent.text for ent in sent.ents]
                    if len(sent_entities) >= 2:
                        for i in range(len(sent_entities) - 1):
                            relations.append((sent_entities[i], "相关", sent_entities[i+1]))
                            self.graph.add_edge(sent_entities[i], sent_entities[i+1], relation="相关", weight=0.5)

            elif format_type == 'transcript':
                # Transcript: Speaker-based and conversational links
                for sent in doc.sents:
                    for token in sent:
                        if token.pos_ in ("VERB", "AUX"):
                            subj = None
                            obj = None
                            for child in token.children:
                                if child.dep_ in ("nsubj", "nsubjpass"):
                                    subj = child.text
                                if child.dep_ in ("dobj", "attr", "obl"):
                                    obj = child.text
                            if subj and obj and subj in entity_texts and obj in entity_texts:
                                relations.append((subj, token.text, obj))
                                self.graph.add_edge(subj, obj, relation=token.text, weight=1.0)
                    # Link entities across speaker turns
                    if text.startswith('Speaker'):
                        speaker = text.split(':')[0]
                        for ent in sent.ents:
                            self.graph.add_node(speaker, type="SPEAKER", chunk_id=chunk_id)
                            relations.append((speaker, "提到", ent.text))
                            self.graph.add_edge(speaker, ent.text, relation="提到", weight=0.7)

            elif format_type == 'user_guide':
                # User Guide: Hierarchical and term-based
                for sent in doc.sents:
                    for token in sent:
                        if token.pos_ in ("VERB", "AUX"):
                            subj = None
                            obj = None
                            for child in token.children:
                                if child.dep_ in ("nsubj", "nsubjpass"):
                                    subj = child.text
                                if child.dep_ in ("dobj", "attr", "obl"):
                                    obj = child.text
                            if subj and obj and subj in entity_texts and obj in entity_texts:
                                relations.append((subj, token.text, obj))
                                self.graph.add_edge(subj, obj, relation=token.text, weight=1.0)
                    # Link section terms
                    sent_entities = [ent.text for ent in sent.ents]
                    if len(sent_entities) >= 2:
                        for i in range(len(sent_entities) - 1):
                            relations.append((sent_entities[i], "包含", sent_entities[i+1]))
                            self.graph.add_edge(sent_entities[i], sent_entities[i+1], relation="包含", weight=0.5)

            else:
                # Default: Verb-based
                for sent in doc.sents:
                    for token in sent:
                        if token.pos_ in ("VERB", "AUX"):
                            subj = None
                            obj = None
                            for child in token.children:
                                if child.dep_ in ("nsubj", "nsubjpass"):
                                    subj = child.text
                                if child.dep_ in ("dobj", "attr", "obl"):
                                    obj = child.text
                            if subj and obj and subj in entity_texts and obj in entity_texts:
                                relations.append((subj, token.text, obj))
                                self.graph.add_edge(subj, obj, relation=token.text, weight=1.0)

            logger.info(f"Extracted {len(entities)} entities, {len(relations)} relations from chunk {chunk_id} (format: {format_type})")
            return entities, relations
        except Exception as e:
            logger.error(f"Failed to extract entities/relations for chunk {chunk_id}: {e}")
            raise

    def get_related_chunks(self, query: str, top_k: int = 3) -> List[str]:
        try:
            doc = self.nlp(query)
            query_entities = {ent.text for ent in doc.ents}
            related_chunks = set()
            for entity in query_entities:
                if entity in self.graph:
                    for neighbor in self.graph.neighbors(entity):
                        chunk_id = self.graph.nodes[neighbor].get("chunk_id")
                        if chunk_id:
                            related_chunks.add(chunk_id)
            result = list(related_chunks)[:top_k]
            logger.info(f"Found {len(result)} related chunks for query entities: {query_entities}")
            return result
        except Exception as e:
            logger.error(f"Failed to get related chunks for query: {e}")
            return []