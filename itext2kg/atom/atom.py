from itext2kg.atom.models import KnowledgeGraph, Entity, Relationship, RelationshipProperties
from itext2kg.atom.graph_matching import GraphMatcher
from itext2kg.llm_output_parsing import LangchainOutputParser
from itext2kg.atom.models.schemas import Relationship as RelationshipSchema
from itext2kg.atom.models.schemas import RelationshipsExtractor
import concurrent.futures
from typing import Dict, List, Optional
from itext2kg.atom.models.prompts import Prompt
from dateutil import parser
import asyncio
from itext2kg.logging_config import get_logger
import re
from collections import Counter

logger = get_logger(__name__)

class Atom:
    def __init__(self, 
                 llm_model,
                 embeddings_model,
                 llm_output_parser: Optional[LangchainOutputParser] = None,
                 ) -> None:        
        """
        Initializes the ATOM with specified language model, embeddings model, and operational parameters.
        
        Args:
        matcher: The matcher instance to be used for matching entities and relationships.
        llm_output_parser: The language model instance to be used for extracting entities and relationships from text.
        """
        self.matcher = GraphMatcher()
        self.llm_output_parser = llm_output_parser or LangchainOutputParser(
            llm_model=llm_model,
            embeddings_model=embeddings_model,
        )
    
    async def extract_quintuples(self, atomic_facts: List[str], observation_timestamp: str) -> List[RelationshipsExtractor]:
        """
        Extracts relationships from atomic facts using the language model.
        """
        return await self.extract_quintuples_with_language(
            atomic_facts=atomic_facts,
            observation_timestamp=observation_timestamp,
            output_language="en",
        )

    async def extract_quintuples_with_language(
        self,
        atomic_facts: List[str],
        observation_timestamp: str,
        output_language: str = "en",
        entity_name_mode: str = "normalized",
        relation_name_mode: str = "en_snake",
        entity_label_allowlist: Optional[List[str]] = None,
    ) -> List[RelationshipsExtractor]:
        return await self.llm_output_parser.extract_information_as_json_for_context(
            output_data_structure=RelationshipsExtractor,
            contexts=atomic_facts,
            system_query=Prompt.temporal_system_query(
                observation_timestamp,
                output_language=output_language,
                entity_name_mode=entity_name_mode,
                relation_name_mode=relation_name_mode,
                entity_label_allowlist=entity_label_allowlist,
            )
            + Prompt.examples(output_language),
        )

    @staticmethod
    def _normalize_entity_label(
        label: str,
        *,
        allowlist: Optional[List[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        unknown_label: str = "unknown",
    ) -> str:
        raw = str(label or "").strip()
        if not raw:
            return unknown_label

        aliases_map: Dict[str, str] = {}
        if aliases:
            aliases_map = {str(k).strip().casefold(): str(v).strip() for k, v in aliases.items() if str(k).strip() != ""}

        mapped = aliases_map.get(raw.casefold(), raw)
        mapped = str(mapped).strip()
        if not mapped:
            return unknown_label

        canonical = mapped.lower()
        if allowlist:
            allow = {str(x).strip().lower() for x in allowlist if str(x).strip() != ""}
            if canonical not in allow:
                return unknown_label
        return canonical

    @classmethod
    def _normalize_relationship_schemas(
        cls,
        relationships: List[RelationshipSchema],
        *,
        entity_label_allowlist: Optional[List[str]] = None,
        entity_label_aliases: Optional[Dict[str, str]] = None,
        unknown_entity_label: str = "unknown",
        drop_unknown_entity_label: bool = False,
        relation_fallback_name: str = "related_to",
    ) -> List[RelationshipSchema]:
        cleaned: List[RelationshipSchema] = []
        for rel in relationships:
            rel_name = str(getattr(rel, "name", "") or "").strip()
            rel.name = rel_name if rel_name else str(relation_fallback_name or "related_to")

            s_name = str(getattr(rel.startNode, "name", "") or "").strip()
            o_name = str(getattr(rel.endNode, "name", "") or "").strip()
            if not s_name or not o_name:
                continue
            rel.startNode.name = s_name
            rel.endNode.name = o_name

            s_label = cls._normalize_entity_label(
                getattr(rel.startNode, "label", ""),
                allowlist=entity_label_allowlist,
                aliases=entity_label_aliases,
                unknown_label=unknown_entity_label,
            )
            o_label = cls._normalize_entity_label(
                getattr(rel.endNode, "label", ""),
                allowlist=entity_label_allowlist,
                aliases=entity_label_aliases,
                unknown_label=unknown_entity_label,
            )
            if drop_unknown_entity_label and (s_label == unknown_entity_label or o_label == unknown_entity_label):
                continue
            rel.startNode.label = s_label
            rel.endNode.label = o_label
            cleaned.append(rel)
        return cleaned

    def merge_two_kgs(
        self,
        kg1,
        kg2,
        rel_threshold: float = 0.8,
        ent_threshold: float = 0.8,
        require_same_entity_label: bool = False,
        rename_relationship_by_embedding: bool = True,
    ):
        """
        Merges two KGs using the same logic as the sequential approach above.
        Returns a single KnowledgeGraph.
        """
        updated_entities, updated_relationships = self.matcher.match_entities_and_update_relationships(
            entities_2=kg1.entities,
            relationships_2=kg1.relationships,
            entities_1=kg2.entities,
            relationships_1=kg2.relationships,
            rel_threshold=rel_threshold,
            ent_threshold=ent_threshold,
            require_same_entity_label=require_same_entity_label,
            rename_relationship_by_embedding=rename_relationship_by_embedding,
        )
        return KnowledgeGraph(entities=updated_entities, relationships=updated_relationships)

    def parallel_atomic_merge(
        self,
        kgs: List[KnowledgeGraph],
        existing_kg: Optional[KnowledgeGraph] = None,
        rel_threshold: float = 0.8,
        ent_threshold: float = 0.8,
        max_workers: int = 4,
        require_same_entity_label: bool = False,
        rename_relationship_by_embedding: bool = True,
    ) -> KnowledgeGraph:
        """
        Merges a list of KnowledgeGraphs in parallel, reducing them pairwise.
        """
        # Handle empty input list
        if not kgs:
            if existing_kg and not existing_kg.is_empty():
                return existing_kg
            return KnowledgeGraph()
        
        # Keep merging until we have just one KG
        current = kgs
        while len(current) > 1:
            merged_results = []
            
            # Prepare pairs
            pairs = [(current[i], current[i+1]) 
                    for i in range(0, len(current) - 1, 2)]
            
            # If there's an odd KG out, keep it aside to append later
            leftover = current[-1] if len(current) % 2 == 1 else None
            
            # Merge pairs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.merge_two_kgs,
                        p[0],
                        p[1],
                        rel_threshold,
                        ent_threshold,
                        require_same_entity_label,
                        rename_relationship_by_embedding,
                    )
                    for p in pairs
                ]
                for f in concurrent.futures.as_completed(futures):
                    merged_results.append(f.result())
            
            # Rebuild current list from newly merged KGs + leftover
            if leftover:
                merged_results.append(leftover)
            
            current = merged_results
        
        # Handle case where current becomes empty after merging
        if not current:
            if existing_kg and not existing_kg.is_empty():
                return existing_kg
            return KnowledgeGraph()
        
        if existing_kg and not existing_kg.is_empty():
            return self.merge_two_kgs(
                current[0],
                existing_kg,
                rel_threshold,
                ent_threshold,
                require_same_entity_label,
                rename_relationship_by_embedding,
            )
        return current[0]

    async def build_atomic_kg_from_quintuples(self, 
        relationships:list[RelationshipSchema], 
        entity_name_weight:float=0.8, 
        entity_label_weight:float=0.2,
        rel_threshold:float=0.8,
        ent_threshold:float=0.8,
        max_workers:int=8,
        require_same_entity_label: bool = False,
        rename_relationship_by_embedding: bool = True,
        entity_label_allowlist: Optional[List[str]] = None,
        entity_label_aliases: Optional[Dict[str, str]] = None,
        unknown_entity_label: str = "unknown",
        drop_unknown_entity_label: bool = False,
        relation_fallback_name: str = "related_to",
        ):
        embedded_relationships = []
        relationships = self._normalize_relationship_schemas(
            relationships,
            entity_label_allowlist=entity_label_allowlist,
            entity_label_aliases=entity_label_aliases,
            unknown_entity_label=unknown_entity_label,
            drop_unknown_entity_label=drop_unknown_entity_label,
            relation_fallback_name=relation_fallback_name,
        )
        if not relationships:
            return KnowledgeGraph()
        temp_kg = KnowledgeGraph(entities=[Entity(**rel.startNode.model_dump()) for rel in relationships] + [Entity(**rel.endNode.model_dump()) for rel in relationships])
        await temp_kg.embed_entities(embeddings_function=self.llm_output_parser.calculate_embeddings, entity_name_weight=entity_name_weight, entity_label_weight=entity_label_weight)

        for relationship in relationships:
            if relationship.t_start is None:
                relationship.t_start = []
            elif relationship.t_end is None:
                relationship.t_end = []
            
            start_entity = temp_kg.get_entity(Entity(**relationship.startNode.model_dump()))
            end_entity = temp_kg.get_entity(Entity(**relationship.endNode.model_dump()))
            
            # Handle the case where entities might not be found (though they should be)
            if start_entity is None or end_entity is None:
                raise ValueError(f"Could not find entities for relationship {relationship.name}")
            
            # Handle timestamp parsing with None checks and error handling
            t_start_timestamps = []
            if relationship.t_start:
                for ts in relationship.t_start:
                    try:
                        parsed_dt = parser.parse(ts)
                        if parsed_dt is not None:
                            t_start_timestamps.append(parsed_dt.timestamp())
                    except Exception as e:
                        logger.warning(f"Could not parse t_start timestamp '{ts}': {e}. Skipping this timestamp.")
                        # Keep the place empty by simply not adding anything to the list
                        continue
            
            t_end_timestamps = []
            if relationship.t_end:
                for ts in relationship.t_end:
                    try:
                        parsed_dt = parser.parse(ts)
                        if parsed_dt is not None:
                            t_end_timestamps.append(parsed_dt.timestamp())
                    except Exception as e:
                        logger.warning(f"Could not parse t_end timestamp '{ts}': {e}. Skipping this timestamp.")
                        # Keep the place empty by simply not adding anything to the list
                        continue
            
            embedded_relationships.append(Relationship(name=relationship.name, 
                                        startEntity=start_entity, 
                                        endEntity=end_entity,
                                        properties = RelationshipProperties(t_start=t_start_timestamps, 
                                                                            t_end=t_end_timestamps)))
            
        

        kg = KnowledgeGraph(entities=temp_kg.entities, relationships=embedded_relationships)
        await kg.embed_relationships(embeddings_function=self.llm_output_parser.calculate_embeddings)
        # this line is just to ensure there are no duplicates entities and relationships inside the same factoid.
        atomic_kgs = kg.split_into_atomic_kgs()
        
        return self.parallel_atomic_merge(
            kgs=atomic_kgs, 
            rel_threshold=rel_threshold, 
            ent_threshold=ent_threshold, 
            max_workers=max_workers,
            require_same_entity_label=require_same_entity_label,
            rename_relationship_by_embedding=rename_relationship_by_embedding,
        )

    async def build_graph(self, 
                          atomic_facts:List[str],
                          obs_timestamp: str,
                          existing_knowledge_graph:KnowledgeGraph=None,
                          ent_threshold:float = 0.8,
                          rel_threshold:float = 0.7,
                          entity_name_weight:float=0.8,
                          entity_label_weight:float=0.2,
                          max_workers:int=8,
                          output_language: str = "en",
                          entity_name_mode: str = "normalized",
                          relation_name_mode: str = "en_snake",
                          require_same_entity_label: bool = False,
                          rename_relationship_by_embedding: bool = True,
                          entity_label_allowlist: Optional[List[str]] = None,
                          entity_label_aliases: Optional[Dict[str, str]] = None,
                          unknown_entity_label: str = "unknown",
                          drop_unknown_entity_label: bool = False,
                          debug_log_empty_relation_name: bool = False,
                          debug_relation_name_sample_size: int = 5,
                          relation_fallback_name: str = "related_to",
                        ) -> KnowledgeGraph:
        system_query = Prompt.temporal_system_query(
            obs_timestamp=obs_timestamp,
            output_language=output_language,
            entity_name_mode=entity_name_mode,
            relation_name_mode=relation_name_mode,
            entity_label_allowlist=entity_label_allowlist,
        )
        examples = Prompt.examples(output_language)
        logger.info("------- Extracting Quintuples---------")
        relationships = await self.llm_output_parser.extract_information_as_json_for_context(output_data_structure=RelationshipsExtractor, contexts=atomic_facts, system_query=system_query+examples)

        if debug_log_empty_relation_name:
            def _clean_relation_name(name: str) -> str:
                # Mirror Relationship.process() cleaning logic without importing to avoid cycles.
                label_pattern = re.compile(r"[^\w]+", flags=re.UNICODE)
                cleaned = label_pattern.sub("_", str(name or "")).replace("&", "and")
                cleaned = re.sub(r"_+", "_", cleaned).strip("_")
                return cleaned.lower()

            total = 0
            empty_raw = 0
            empty_after_clean = 0
            related_to_like_raw = 0
            related_to_like_clean = 0
            raw_counter: Counter[str] = Counter()
            clean_counter: Counter[str] = Counter()
            samples: List[str] = []

            for fact, rel_block in zip(atomic_facts, relationships):
                rels = list(getattr(rel_block, "relationships", []) or [])
                for r in rels:
                    total += 1
                    raw_name = str(getattr(r, "name", "") or "")
                    raw_stripped = raw_name.strip()
                    cleaned = _clean_relation_name(raw_name)
                    raw_counter.update([raw_stripped])
                    clean_counter.update([cleaned])

                    if raw_stripped.lower() in {"related_to", "related to", "related"}:
                        related_to_like_raw += 1
                    if cleaned in {"related_to", "related"}:
                        related_to_like_clean += 1

                    if raw_stripped == "":
                        empty_raw += 1
                        if len(samples) < int(debug_relation_name_sample_size):
                            samples.append(
                                f"EMPTY_RAW name={raw_name!r} s=({r.startNode.label},{r.startNode.name}) "
                                f"o=({r.endNode.label},{r.endNode.name}) fact={str(fact)[:80]!r}"
                            )
                    elif cleaned == "":
                        empty_after_clean += 1
                        if len(samples) < int(debug_relation_name_sample_size):
                            samples.append(
                                f"EMPTY_AFTER_CLEAN name={raw_name!r} s=({r.startNode.label},{r.startNode.name}) "
                                f"o=({r.endNode.label},{r.endNode.name}) fact={str(fact)[:80]!r}"
                            )
                    elif cleaned in {"related_to", "related"}:
                        if len(samples) < int(debug_relation_name_sample_size):
                            samples.append(
                                f"GENERIC_REL name={raw_name!r} cleaned={cleaned!r} s=({r.startNode.label},{r.startNode.name}) "
                                f"o=({r.endNode.label},{r.endNode.name}) fact={str(fact)[:80]!r}"
                            )

            logger.info(
                "关系名诊断：total=%d empty_raw=%d empty_after_clean=%d generic_raw=%d generic_clean=%d fallback=%r",
                total,
                empty_raw,
                empty_after_clean,
                related_to_like_raw,
                related_to_like_clean,
                relation_fallback_name,
            )
            if total > 0:
                logger.info("关系名TOP(清洗后)：%s", clean_counter.most_common(10))
                logger.info("关系名TOP(原始)：%s", raw_counter.most_common(10))
            for line in samples:
                logger.info("关系名样例：%s", line)
        
        logger.info("------- Building Atomic KGs---------")
        
        atomic_kgs = await asyncio.gather(*list(map(
            self.build_atomic_kg_from_quintuples, 
            [relation.relationships for relation in relationships], 
            [entity_name_weight for _ in relationships], 
            [entity_label_weight for _ in relationships],
            [rel_threshold for _ in relationships],
            [ent_threshold for _ in relationships],
            [max_workers for _ in relationships],
            [require_same_entity_label for _ in relationships],
            [rename_relationship_by_embedding for _ in relationships],
            [entity_label_allowlist for _ in relationships],
            [entity_label_aliases for _ in relationships],
            [unknown_entity_label for _ in relationships],
            [drop_unknown_entity_label for _ in relationships],
            [relation_fallback_name for _ in relationships],
        )))

        logger.info("------- Adding Atomic Facts to Atomic KGs---------")
        for atomic_kg, fact in zip(atomic_kgs, atomic_facts):
            atomic_kg.add_atomic_facts_to_relationships(atomic_facts=[fact])

        logger.info("------- Merging Atomic KGs---------")
        cleaned_atomic_kgs = [kg for kg in atomic_kgs if kg.relationships != []]
        merged_kg = self.parallel_atomic_merge(kgs=cleaned_atomic_kgs, 
        rel_threshold=rel_threshold, 
        ent_threshold=ent_threshold, 
        max_workers=max_workers,
        require_same_entity_label=require_same_entity_label,
        rename_relationship_by_embedding=rename_relationship_by_embedding,
        )

        logger.info("------- Adding Observation Timestamp to Relationships---------")
        merged_kg.add_t_obs_to_relationships(t_obs=[obs_timestamp])
    
        if existing_knowledge_graph:
            global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(entities_1=merged_kg.entities,
                                                                 entities_2=existing_knowledge_graph.entities,
                                                                 relationships_1=merged_kg.relationships,
                                                                 relationships_2=existing_knowledge_graph.relationships,
                                                                 ent_threshold=ent_threshold,
                                                                 rel_threshold=rel_threshold,
                                                                 require_same_entity_label=require_same_entity_label,
                                                                 rename_relationship_by_embedding=rename_relationship_by_embedding,
                                                                #  entity_name_weight=entity_name_weight,
                                                                #  entity_label_weight=entity_label_weight
                                                                 )    
        
            constructed_kg = KnowledgeGraph(entities=global_entities, relationships=global_relationships)
            return constructed_kg
        return merged_kg
    
    async def build_graph_from_different_obs_times(self,
                                                   atomic_facts_with_obs_timestamps:dict,
                                                    existing_knowledge_graph:KnowledgeGraph=None,
                                                    ent_threshold:float = 0.8,
                                                    rel_threshold:float = 0.7,
                                                    entity_name_weight:float=0.8,
                                                    entity_label_weight:float=0.2,
                                                    max_workers:int=8,
                                                    output_language: str = "en",
                                                    entity_name_mode: str = "normalized",
                                                    relation_name_mode: str = "en_snake",
                                                    require_same_entity_label: bool = False,
                                                    rename_relationship_by_embedding: bool = True,
                                               ):
        kgs = await asyncio.gather(*[
                        self.build_graph(
                            atomic_facts=atomic_facts_with_obs_timestamps[timestamp], 
                            obs_timestamp=timestamp,
                            output_language=output_language,
                            entity_name_mode=entity_name_mode,
                            relation_name_mode=relation_name_mode,
                            require_same_entity_label=require_same_entity_label,
                            rename_relationship_by_embedding=rename_relationship_by_embedding,
                            ent_threshold=ent_threshold,
                            rel_threshold=rel_threshold,
                            entity_name_weight=entity_name_weight,
                            entity_label_weight=entity_label_weight,
                            existing_knowledge_graph=None,
                        ) for timestamp in atomic_facts_with_obs_timestamps
                    ])
        if existing_knowledge_graph:
            return self.parallel_atomic_merge(
                kgs=[existing_knowledge_graph] + kgs,
                rel_threshold=rel_threshold,
                ent_threshold=ent_threshold,
                max_workers=max_workers,
                require_same_entity_label=require_same_entity_label,
                rename_relationship_by_embedding=rename_relationship_by_embedding,
            )
        
        return self.parallel_atomic_merge(
            kgs=kgs,
            rel_threshold=rel_threshold,
            ent_threshold=ent_threshold,
            max_workers=max_workers,
            require_same_entity_label=require_same_entity_label,
            rename_relationship_by_embedding=rename_relationship_by_embedding,
        )
