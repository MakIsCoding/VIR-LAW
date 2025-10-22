"""
VirLaw Evaluation Metrics Calculator
Calculates Precision@5, Recall@10, MRR, Accuracy, and Hallucination Rate
"""

import json
import re
import time
from typing import List, Dict, Any, Tuple
import numpy as np

class VirLawEvaluator:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.test_results = []
        
        # Ground truth: Which articles exist in your PDF
        self.valid_articles = list(range(12, 36))  # Articles 12-35
        
        # Ground truth: Articles per document (you'll populate this)
        self.article_to_docs = self._build_article_index()
    
    def _build_article_index(self) -> Dict[int, List[str]]:
        """
        Build index of which doc_ids contain each article.
        This is ground truth for recall calculation.
        """
        article_index = {}
        
        # Access your docstore to build the index
        try:
            for doc_id, doc in self.rag_system.doc_stores['text'].store.items():
                content = doc.get('page_content', '')
                metadata = doc.get('metadata', {})
                
                # Extract article numbers from content
                article_matches = re.findall(r'\bArticle\s+(\d+)', content, re.IGNORECASE)
                
                for article_str in article_matches:
                    article_num = int(article_str)
                    if article_num not in article_index:
                        article_index[article_num] = []
                    article_index[article_num].append(doc_id)
        
        except Exception as e:
            print(f"Warning: Could not build article index: {e}")
        
        return article_index
    
    def calculate_precision_at_k(self, retrieved_docs: List[Any], query_articles: List[int], k: int = 5) -> float:
        """
        Calculate Precision@K: Relevant docs in top K / K
        
        Args:
            retrieved_docs: List of retrieved Document objects
            query_articles: List of article numbers that are relevant to the query
            k: Number of top documents to consider (default 5)
        
        Returns:
            Precision@K as float (0-1)
        """
        if not retrieved_docs or not query_articles:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_count = 0
        
        for doc in top_k:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            # Check if any of the query articles are mentioned in this doc
            for article in query_articles:
                if f"Article {article}" in content or f"article {article}" in content.lower():
                    relevant_count += 1
                    break  # Count doc only once even if multiple articles match
        
        precision = relevant_count / k
        return precision
    
    def calculate_recall_at_k(self, retrieved_docs: List[Any], query_articles: List[int], k: int = 10) -> float:
        """
        Calculate Recall@K: Relevant docs retrieved in top K / Total relevant docs in corpus
        
        Args:
            retrieved_docs: List of retrieved Document objects
            query_articles: List of article numbers relevant to query
            k: Number of top documents to consider (default 10)
        
        Returns:
            Recall@K as float (0-1)
        """
        if not query_articles:
            return 0.0
        
        # Get all relevant doc_ids from ground truth
        relevant_doc_ids = set()
        for article in query_articles:
            if article in self.article_to_docs:
                relevant_doc_ids.update(self.article_to_docs[article])
        
        if not relevant_doc_ids:
            return 0.0
        
        # Check how many were retrieved in top K
        top_k = retrieved_docs[:k]
        retrieved_relevant = 0
        
        for doc in top_k:
            doc_id = doc.metadata.get('doc_id', '') if hasattr(doc, 'metadata') else ''
            if doc_id in relevant_doc_ids:
                retrieved_relevant += 1
        
        recall = retrieved_relevant / len(relevant_doc_ids)
        return recall
    
    def calculate_reciprocal_rank(self, retrieved_docs: List[Any], query_articles: List[int]) -> float:
        """
        Calculate Reciprocal Rank: 1 / rank_of_first_relevant_document
        
        Args:
            retrieved_docs: List of retrieved Document objects
            query_articles: List of article numbers relevant to query
        
        Returns:
            Reciprocal rank as float (0-1)
        """
        if not retrieved_docs or not query_articles:
            return 0.0
        
        for rank, doc in enumerate(retrieved_docs, start=1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            
            # Check if any query article is in this doc
            for article in query_articles:
                if f"Article {article}" in content or f"article {article}" in content.lower():
                    return 1.0 / rank
        
        return 0.0  # No relevant document found
    
    def check_hallucination(self, response: str, sources: List[Dict]) -> bool:
        """
        Check if response contains hallucinated information.
        
        Returns:
            True if hallucination detected, False otherwise
        """
        hallucination_flags = []
        
        # Check 1: Mentions articles outside valid range
        mentioned_articles = re.findall(r'\bArticle\s+(\d+[A-Z]?)', response, re.IGNORECASE)
        for article_str in mentioned_articles:
            article_num = int(re.search(r'\d+', article_str).group())
            if article_num not in self.valid_articles:
                hallucination_flags.append(f"Invalid article: {article_num}")
        
        # Check 2: Fabricated case law (has "v." but no PDF citation)
        if " v. " in response or " vs. " in response.lower():
            # Check if response admits no documents
            if "I don't have relevant documents" not in response:
                # Check if sources are empty or generic
                if not sources or all(s.get('document_name') in ['Unknown', 'Constitutional Document'] for s in sources):
                    hallucination_flags.append("Potential fabricated case law")
        
        # Check 3: Claims specific page numbers but sources don't match
        page_mentions = re.findall(r'Page\s+(\d+)', response, re.IGNORECASE)
        if page_mentions and sources:
            source_pages = [str(s.get('page_number', '')) for s in sources]
            for page in page_mentions:
                if page not in source_pages and page != 'N/A':
                    hallucination_flags.append(f"Unverified page reference: {page}")
        
        return len(hallucination_flags) > 0, hallucination_flags
    
    def evaluate_single_query(self, query: str, expected_articles: List[int], 
                             query_type: str = "constitutional") -> Dict[str, Any]:
        """
        Evaluate a single query and return all metrics.
        
        Args:
            query: The user question
            expected_articles: List of article numbers that should be retrieved
            query_type: Type of query (default "constitutional")
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"📊 Evaluating: {query}")
        print(f"Expected articles: {expected_articles}")
        print(f"{'='*60}")
        
        # Measure latency
        start_time = time.time()
        
        # Get raw retrieved documents BEFORE they're sent to LLM
        # We need to intercept at retrieval stage
        try:
            # Call the retrieval part of your pipeline
            enhanced_query = self.rag_system._enhance_constitutional_query(
                query, 
                self.rag_system._analyze_legal_query(query, query_type)
            )
            
            # Get retrieved documents
            retrieved_texts = self.rag_system.retrievers['texts'].invoke(enhanced_query)
            
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            retrieved_texts = []
        
        # Get full response
        try:
            result = self.rag_system.ultimate_query_processor(
                question=query,
                prompt_template_key=query_type
            )
            response_text = result.get('response', '')
            sources = result.get('sources', [])
            confidence = result.get('confidence', 0.0)
        except Exception as e:
            print(f"❌ Query processing error: {e}")
            response_text = ""
            sources = []
            confidence = 0.0
        
        latency = time.time() - start_time
        
        # Calculate metrics
        precision_5 = self.calculate_precision_at_k(retrieved_texts, expected_articles, k=5)
        recall_10 = self.calculate_recall_at_k(retrieved_texts, expected_articles, k=10)
        mrr = self.calculate_reciprocal_rank(retrieved_texts, expected_articles)
        
        # Check for hallucinations
        is_hallucinated, hallucination_details = self.check_hallucination(response_text, sources)
        
        # Check if answer is correct (manual review needed, but we can auto-check some things)
        has_pdf_citation = any(
            s.get('document_name', '').endswith('.pdf') 
            for s in sources
        )
        
        metrics = {
            'query': query,
            'expected_articles': expected_articles,
            'latency_seconds': round(latency, 2),
            'precision_at_5': round(precision_5, 3),
            'recall_at_10': round(recall_10, 3),
            'reciprocal_rank': round(mrr, 3),
            'confidence': round(confidence, 2),
            'docs_retrieved': len(retrieved_texts),
            'has_pdf_citation': has_pdf_citation,
            'is_hallucinated': is_hallucinated,
            'hallucination_details': hallucination_details,
            'response_length': len(response_text),
            'num_sources': len(sources),
        }
        
        # Print results
        print(f"\n📈 Metrics:")
        print(f"   ⏱️  Latency: {metrics['latency_seconds']}s")
        print(f"   🎯 Precision@5: {metrics['precision_at_5']:.1%}")
        print(f"   🔍 Recall@10: {metrics['recall_at_10']:.1%}")
        print(f"   📊 MRR: {metrics['reciprocal_rank']:.3f}")
        print(f"   🔥 Confidence: {metrics['confidence']:.0%}")
        print(f"   📄 PDF Citation: {'✅' if has_pdf_citation else '❌'}")
        print(f"   ⚠️  Hallucination: {'❌ YES' if is_hallucinated else '✅ NO'}")
        
        self.test_results.append(metrics)
        return metrics
    
    def run_benchmark(self, test_set: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Run full benchmark on a test set.
        
        Args:
            test_set: List of dicts with 'query' and 'expected_articles' keys
        
        Returns:
            Aggregate metrics across all queries
        """
        print(f"\n🚀 Running VirLaw Benchmark on {len(test_set)} queries...")
        print(f"{'='*80}\n")
        
        self.test_results = []
        
        for test_case in test_set:
            self.evaluate_single_query(
                query=test_case['query'],
                expected_articles=test_case['expected_articles'],
                query_type=test_case.get('type', 'constitutional')
            )
            time.sleep(0.5)  # Rate limiting
        
        # Calculate aggregate metrics
        aggregate = {
            'avg_precision_at_5': np.mean([r['precision_at_5'] for r in self.test_results]),
            'avg_recall_at_10': np.mean([r['recall_at_10'] for r in self.test_results]),
            'mean_reciprocal_rank': np.mean([r['reciprocal_rank'] for r in self.test_results]),
            'avg_latency': np.mean([r['latency_seconds'] for r in self.test_results]),
            'avg_confidence': np.mean([r['confidence'] for r in self.test_results]),
            'hallucination_rate': np.mean([r['is_hallucinated'] for r in self.test_results]),
            'pdf_citation_rate': np.mean([r['has_pdf_citation'] for r in self.test_results]),
        }
        
        # Print final report
        print(f"\n{'='*80}")
        print(f"📊 VIRLAW BENCHMARK RESULTS")
        print(f"{'='*80}")
        print(f"Precision@5:      {aggregate['avg_precision_at_5']:.1%}  (Target: 96%)")
        print(f"Recall@10:        {aggregate['avg_recall_at_10']:.1%}  (Target: 94%)")
        print(f"MRR:              {aggregate['mean_reciprocal_rank']:.3f} (Target: 0.93)")
        print(f"Avg Latency:      {aggregate['avg_latency']:.2f}s (Target: 7.1s)")
        print(f"Avg Confidence:   {aggregate['avg_confidence']:.0%}")
        print(f"Hallucination %:  {aggregate['hallucination_rate']:.1%} (Target: 0%)")
        print(f"PDF Citation %:   {aggregate['pdf_citation_rate']:.1%} (Target: 100%)")
        print(f"{'='*80}\n")
        
        # Save detailed results
        with open('virlaw_benchmark_results.json', 'w') as f:
            json.dump({
                'aggregate_metrics': aggregate,
                'detailed_results': self.test_results,
                'test_set': test_set
            }, f, indent=2)
        
        print("💾 Detailed results saved to: virlaw_benchmark_results.json")
        
        return aggregate


# Helper function to generate comparison table
def generate_comparison_table(virlaw_metrics: Dict[str, float]) -> str:
    """Generate markdown table comparing VirLaw to baselines"""
    
    table = """
| Metric | Groq Baseline | Single-source RAG | VirLaw |
|--------|---------------|-------------------|--------|
| Accuracy (%) | 68 | 78 | TBD |
| Hallucination Rate (%) | 21 | 12 | {hall:.0f} |
| Precision@5 (%) | 70 | 82 | {prec:.0f} |
| Recall@10 (%) | 65 | 80 | {rec:.0f} |
| Mean Reciprocal Rank | 0.62 | 0.74 | {mrr:.2f} |
| Expert Trust Rating | Medium | High | {trust} |
| Avg. Response Latency (s) | 5.2 | 4.8 | {lat:.1f} |
""".format(
        hall=virlaw_metrics['hallucination_rate'] * 100,
        prec=virlaw_metrics['avg_precision_at_5'] * 100,
        rec=virlaw_metrics['avg_recall_at_10'] * 100,
        mrr=virlaw_metrics['mean_reciprocal_rank'],
        trust=determine_trust_rating(virlaw_metrics),
        lat=virlaw_metrics['avg_latency']
    )
    
    return table

def determine_trust_rating(metrics: Dict[str, float]) -> str:
    """Determine expert trust rating based on metrics"""
    if metrics['hallucination_rate'] < 0.05 and metrics['avg_precision_at_5'] > 0.90:
        return "Very High"
    elif metrics['hallucination_rate'] < 0.15 and metrics['avg_precision_at_5'] > 0.80:
        return "High"
    elif metrics['hallucination_rate'] < 0.30 and metrics['avg_precision_at_5'] > 0.70:
        return "Medium"
    else:
        return "Low"
