"""
VirLaw Complete Benchmark - PRODUCTION VERSION
Fixed type-safe checking for articles_mentioned field
"""

import requests
import time
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

# ==================== RATE LIMITING ====================

REQUESTS_PER_MINUTE = 15
SECONDS_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE
SAFETY_BUFFER = 0.5

class RateLimiter:
    def __init__(self, requests_per_minute: int = 15):
        self.min_interval = (60.0 / requests_per_minute) + SAFETY_BUFFER
        self.last_request_time = 0
        self.request_count = 0
        self.start_time = time.time()
    
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            print(f"   ⏳ Rate limiting: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_stats(self):
        elapsed = time.time() - self.start_time
        requests_per_min = (self.request_count / elapsed) * 60 if elapsed > 0 else 0
        return {
            'total_requests': self.request_count,
            'elapsed_seconds': elapsed,
            'requests_per_minute': requests_per_min
        }

# ==================== TEST SET ====================

TEST_SET = [
    {"id": 1, "query": "What does Article 12 define as 'the State'?", "expected_articles": [12], "expected_keywords": ["State", "Government", "Parliament", "Legislature"], "correct_answer_contains": ["Government", "Parliament", "Legislature"]},
    {"id": 2, "query": "What does Article 13 say about laws inconsistent with fundamental rights?", "expected_articles": [13], "expected_keywords": ["void", "inconsistent", "fundamental rights"], "correct_answer_contains": ["void", "inconsistent", "Part III"]},
    {"id": 3, "query": "What does Article 14 guarantee?", "expected_articles": [14], "expected_keywords": ["equality", "law", "equal protection"], "correct_answer_contains": ["equality", "law", "protection"]},
    {"id": 4, "query": "What does Article 15 prohibit?", "expected_articles": [15], "expected_keywords": ["discrimination", "religion", "race", "caste", "sex"], "correct_answer_contains": ["discrimination", "religion", "caste"]},
    {"id": 5, "query": "What special provisions does Article 15(3) allow?", "expected_articles": [15], "expected_keywords": ["women", "children", "special provision"], "correct_answer_contains": ["women", "children"]},
    {"id": 6, "query": "What does Article 16 guarantee in public employment?", "expected_articles": [16], "expected_keywords": ["equality", "opportunity", "employment", "State"], "correct_answer_contains": ["equality", "opportunity", "employment"]},
    {"id": 7, "query": "What does Article 17 abolish?", "expected_articles": [17], "expected_keywords": ["untouchability", "abolish", "offence"], "correct_answer_contains": ["untouchability", "abolished"]},
    {"id": 8, "query": "What does Article 18 prohibit?", "expected_articles": [18], "expected_keywords": ["titles", "prohibit", "military", "academic"], "correct_answer_contains": ["titles", "State"]},
    {"id": 9, "query": "What freedoms does Article 19(1)(a) protect?", "expected_articles": [19], "expected_keywords": ["freedom", "speech", "expression"], "correct_answer_contains": ["freedom", "speech", "expression"]},
    {"id": 10, "query": "What does Article 19(1)(b) guarantee?", "expected_articles": [19], "expected_keywords": ["assemble", "peaceably", "arms"], "correct_answer_contains": ["assemble", "peaceably"]},
    {"id": 11, "query": "What does Article 19(1)(c) protect?", "expected_articles": [19], "expected_keywords": ["association", "union", "form"], "correct_answer_contains": ["association", "union"]},
    {"id": 12, "query": "What does Article 19(1)(d) allow?", "expected_articles": [19], "expected_keywords": ["move", "freely", "territory", "India"], "correct_answer_contains": ["move", "territory"]},
    {"id": 13, "query": "What does Article 19(1)(e) protect?", "expected_articles": [19], "expected_keywords": ["reside", "settle", "India"], "correct_answer_contains": ["reside", "settle"]},
    {"id": 14, "query": "What does Article 19(1)(g) guarantee?", "expected_articles": [19], "expected_keywords": ["profession", "occupation", "trade", "business"], "correct_answer_contains": ["profession", "occupation"]},
    {"id": 15, "query": "What restrictions does Article 19(2) allow on speech?", "expected_articles": [19], "expected_keywords": ["reasonable restrictions", "sovereignty", "public order", "defamation"], "correct_answer_contains": ["restrictions", "sovereignty", "order"]},
    {"id": 16, "query": "What restrictions can Article 19(5) impose on freedom of movement?", "expected_articles": [19], "expected_keywords": ["reasonable restrictions", "interests", "Scheduled Tribes"], "correct_answer_contains": ["restrictions", "Scheduled Tribes"]},
    {"id": 17, "query": "What protections does Article 20 provide?", "expected_articles": [20], "expected_keywords": ["ex post facto", "double jeopardy", "self-incrimination"], "correct_answer_contains": ["conviction", "offences"]},
    {"id": 18, "query": "What does Article 21 protect?", "expected_articles": [21], "expected_keywords": ["life", "personal liberty", "procedure"], "correct_answer_contains": ["life", "liberty", "procedure"]},
    {"id": 19, "query": "What does Article 21A guarantee?", "expected_articles": [21], "expected_keywords": ["education", "children", "six", "fourteen"], "correct_answer_contains": ["education", "children"]},
    {"id": 20, "query": "What rights does Article 22(1) provide to arrested persons?", "expected_articles": [22], "expected_keywords": ["arrested", "grounds", "legal practitioner"], "correct_answer_contains": ["grounds", "legal practitioner"]},
    {"id": 21, "query": "What does Article 22(2) require?", "expected_articles": [22], "expected_keywords": ["magistrate", "twenty-four hours", "arrested"], "correct_answer_contains": ["magistrate", "twenty-four hours"]},
    {"id": 22, "query": "What does Article 22(4) require for preventive detention?", "expected_articles": [22], "expected_keywords": ["preventive detention", "Advisory Board", "three months"], "correct_answer_contains": ["Advisory Board", "three months"]},
    {"id": 23, "query": "What does Article 23 prohibit?", "expected_articles": [23], "expected_keywords": ["traffic", "human beings", "forced labour", "begar"], "correct_answer_contains": ["traffic", "forced labour"]},
    {"id": 24, "query": "What does Article 24 prohibit?", "expected_articles": [24], "expected_keywords": ["children", "factory", "mine", "fourteen"], "correct_answer_contains": ["children", "factory"]},
    {"id": 25, "query": "What does Article 25 guarantee?", "expected_articles": [25], "expected_keywords": ["religious freedom", "conscience", "practice", "propagate"], "correct_answer_contains": ["freedom", "conscience", "practice"]},
    {"id": 26, "query": "What does Article 25(1) subject religious freedom to?", "expected_articles": [25], "expected_keywords": ["public order", "morality", "health"], "correct_answer_contains": ["public order", "morality", "health"]},
    {"id": 27, "query": "What does Article 26(a) allow?", "expected_articles": [26], "expected_keywords": ["establish", "maintain", "institutions", "religious", "charitable"], "correct_answer_contains": ["establish", "maintain", "institutions"]},
    {"id": 28, "query": "What does Article 26(b) grant?", "expected_articles": [26], "expected_keywords": ["manage", "affairs", "religion"], "correct_answer_contains": ["manage", "affairs", "religion"]},
    {"id": 29, "query": "What does Article 27 prohibit?", "expected_articles": [27], "expected_keywords": ["taxes", "promotion", "religion"], "correct_answer_contains": ["taxes", "religion"]},
    {"id": 30, "query": "What does Article 28 prohibit in educational institutions?", "expected_articles": [28], "expected_keywords": ["religious instruction", "State funds", "educational institution"], "correct_answer_contains": ["religious instruction", "State funds"]},
    {"id": 31, "query": "What does Article 29(1) protect?", "expected_articles": [29], "expected_keywords": ["language", "script", "culture", "conserve"], "correct_answer_contains": ["language", "culture", "conserve"]},
    {"id": 32, "query": "What does Article 29(2) prohibit?", "expected_articles": [29], "expected_keywords": ["admission", "educational institution", "discrimination"], "correct_answer_contains": ["admission", "educational", "discrimination"]},
    {"id": 33, "query": "What does Article 30(1) grant to minorities?", "expected_articles": [30], "expected_keywords": ["minorities", "establish", "administer", "educational institutions"], "correct_answer_contains": ["minorities", "establish", "educational"]},
    {"id": 34, "query": "What does Article 30(2) prohibit in granting aid?", "expected_articles": [30], "expected_keywords": ["aid", "discrimination", "minority", "educational institution"], "correct_answer_contains": ["aid", "discrimination", "minority"]},
    {"id": 35, "query": "What does Article 32 provide?", "expected_articles": [32], "expected_keywords": ["remedy", "Supreme Court", "enforcement", "fundamental rights"], "correct_answer_contains": ["remedy", "Supreme Court", "enforcement"]},
    {"id": 36, "query": "What writs can Article 32 issue?", "expected_articles": [32], "expected_keywords": ["habeas corpus", "mandamus", "prohibition", "certiorari", "quo warranto"], "correct_answer_contains": ["habeas corpus", "mandamus"]},
    {"id": 37, "query": "What does Article 33 empower Parliament to do?", "expected_articles": [33], "expected_keywords": ["Armed Forces", "modify", "rights", "Parliament"], "correct_answer_contains": ["Armed Forces", "modify", "rights"]},
    {"id": 38, "query": "What does Article 34 allow during martial law?", "expected_articles": [34], "expected_keywords": ["martial law", "indemnify", "acts"], "correct_answer_contains": ["martial law", "indemnify"]},
    {"id": 39, "query": "What does Article 35 empower Parliament to do?", "expected_articles": [35], "expected_keywords": ["Parliament", "legislation", "fundamental rights"], "correct_answer_contains": ["Parliament", "laws", "fundamental rights"]},
    {"id": 40, "query": "Which articles guarantee right to equality?", "expected_articles": [14, 15, 16, 17, 18], "expected_keywords": ["equality", "discrimination", "Article 14", "Article 15"], "correct_answer_contains": ["Article 14", "Article 15", "equality"]},
    {"id": 41, "query": "Which articles protect freedom of speech and movement?", "expected_articles": [19], "expected_keywords": ["speech", "expression", "movement", "Article 19"], "correct_answer_contains": ["Article 19", "freedom"]},
    {"id": 42, "query": "Which articles protect religious freedom?", "expected_articles": [25, 26, 27, 28], "expected_keywords": ["religious freedom", "Article 25", "Article 26"], "correct_answer_contains": ["Article 25", "Article 26", "religious"]},
    {"id": 43, "query": "What are cultural and educational rights?", "expected_articles": [29, 30], "expected_keywords": ["minorities", "educational", "cultural", "Article 29", "Article 30"], "correct_answer_contains": ["Article 29", "Article 30"]},
    {"id": 44, "query": "What protections exist against arrest and detention?", "expected_articles": [20, 21, 22], "expected_keywords": ["arrest", "detention", "Article 20", "Article 21", "Article 22"], "correct_answer_contains": ["Article 20", "Article 21", "Article 22"]},
    {"id": 45, "query": "Which articles prohibit exploitation?", "expected_articles": [23, 24], "expected_keywords": ["exploitation", "forced labour", "child labour", "Article 23", "Article 24"], "correct_answer_contains": ["Article 23", "Article 24"]},
    {"id": 46, "query": "What are the reasonable restrictions on freedom of speech under Article 19(2)?", "expected_articles": [19], "expected_keywords": ["reasonable restrictions", "sovereignty", "integrity", "security", "public order"], "correct_answer_contains": ["restrictions", "sovereignty", "order"]},
    {"id": 47, "query": "What special provisions exist for Scheduled Castes and Scheduled Tribes?", "expected_articles": [15, 16, 19], "expected_keywords": ["Scheduled Castes", "Scheduled Tribes", "advancement"], "correct_answer_contains": ["Scheduled Castes", "Scheduled Tribes"]},
    {"id": 48, "query": "What are the differences between Article 25 and Article 26?", "expected_articles": [25, 26], "expected_keywords": ["individual", "denomination", "practice", "manage", "institutions"], "correct_answer_contains": ["Article 25", "Article 26"]},
    {"id": 49, "query": "What procedural safeguards does Article 22 provide for detained persons?", "expected_articles": [22], "expected_keywords": ["grounds", "legal practitioner", "magistrate", "Advisory Board"], "correct_answer_contains": ["legal practitioner", "magistrate", "Advisory Board"]},
    {"id": 50, "query": "What remedies are available under Article 32 for violation of fundamental rights?", "expected_articles": [32], "expected_keywords": ["Supreme Court", "writs", "habeas corpus", "mandamus", "enforcement"], "correct_answer_contains": ["Supreme Court", "writs", "enforcement"]},
]

VALID_ARTICLES = list(range(12, 36))

ARTICLE_PAGE_MAP = {
    12: [1], 13: [1], 14: [1,2,6], 15: [2,6], 16: [2,6], 17: [2,6], 18: [2,6],
    19: [3,4,5], 20: [6,12], 21: [6,7,12], 22: [7], 23: [8], 24: [8],
    25: [9], 26: [9], 27: [9], 28: [9], 29: [9,10], 30: [10],
    31: [11], 32: [11,12], 33: [14], 34: [14], 35: [14]
}

# ==================== TYPE-SAFE METRIC CALCULATORS ====================

def calculate_accuracy(test_case: Dict, response_text: str) -> bool:
    response_lower = response_text.lower()
    keyword_matches = sum(1 for keyword in test_case['expected_keywords'] if keyword.lower() in response_lower)
    answer_matches = sum(1 for phrase in test_case['correct_answer_contains'] if phrase.lower() in response_lower)
    keyword_score = keyword_matches / len(test_case['expected_keywords'])
    answer_score = answer_matches / len(test_case['correct_answer_contains'])
    return (keyword_score >= 0.5) or (answer_score >= 0.4)

def check_hallucination(response_text: str, sources: List[Dict]) -> Tuple[bool, List[str]]:
    issues = []
    mentioned_articles = re.findall(r'\bArticle\s+(\d+)', response_text, re.IGNORECASE)
    for article_str in mentioned_articles:
        if int(article_str) not in VALID_ARTICLES:
            issues.append(f"Invalid Article {article_str}")
    if (" v. " in response_text or " vs. " in response_text.lower()):
        if not any(s.get('document_name', '').endswith('.pdf') for s in sources):
            issues.append("Potential fabricated case law")
    if "I don't have relevant documents" in response_text and len(response_text) > 500:
        issues.append("Claims no documents but provides detailed answer")
    return len(issues) > 0, issues

def calculate_precision_at_5(test_case: Dict, retrieved_docs_metadata: List[Dict]) -> float:
    """✅ TYPE-SAFE: Handle articles_mentioned as list, string, or None"""
    if not retrieved_docs_metadata:
        return 0.0
    
    top_5 = retrieved_docs_metadata[:5]
    relevant_count = 0
    
    for doc in top_5:
        is_relevant = False
        
        # ✅ TYPE-SAFE: Check articles_mentioned (handle list/string/None)
        articles_mentioned = doc.get('articles_mentioned')
        
        if articles_mentioned is not None:
            # Convert to list if needed
            if isinstance(articles_mentioned, str):
                try:
                    articles_mentioned = json.loads(articles_mentioned)
                except:
                    articles_mentioned = []
            
            if isinstance(articles_mentioned, list) and len(articles_mentioned) > 0:
                for article in test_case['expected_articles']:
                    if article in articles_mentioned:
                        is_relevant = True
                        break
        
        # Priority 2: Check full_content
        if not is_relevant:
            full_content = doc.get('full_content', '')
            if full_content:
                for article in test_case['expected_articles']:
                    if f"Article {article}" in full_content or f"article {article}" in full_content.lower():
                        is_relevant = True
                        break
        
        # Priority 3: Check truncated content
        if not is_relevant:
            content = doc.get('content', '')
            for article in test_case['expected_articles']:
                if f"Article {article}" in content or f"article {article}" in content.lower():
                    is_relevant = True
                    break
        
        # Priority 4: Check page number
        if not is_relevant:
            doc_name = doc.get('document_name', '').lower()
            page = doc.get('page_number', '')
            
            if ('fundamental' in doc_name or 'constitution' in doc_name) and page and page != 'N/A':
                try:
                    page_int = int(page)
                    for article in test_case['expected_articles']:
                        if article in ARTICLE_PAGE_MAP and page_int in ARTICLE_PAGE_MAP[article]:
                            is_relevant = True
                            break
                except:
                    pass
        
        if is_relevant:
            relevant_count += 1
    
    return relevant_count / 5.0

def calculate_recall_at_10(test_case: Dict, retrieved_docs_metadata: List[Dict]) -> float:
    """✅ TYPE-SAFE"""
    if not retrieved_docs_metadata:
        return 0.0
    
    top_10 = retrieved_docs_metadata[:10]
    found_articles = set()
    
    for doc in top_10:
        # TYPE-SAFE: Handle articles_mentioned
        articles_mentioned = doc.get('articles_mentioned')
        
        if articles_mentioned is not None:
            if isinstance(articles_mentioned, str):
                try:
                    articles_mentioned = json.loads(articles_mentioned)
                except:
                    articles_mentioned = []
            
            if isinstance(articles_mentioned, list) and len(articles_mentioned) > 0:
                for article in test_case['expected_articles']:
                    if article in articles_mentioned:
                        found_articles.add(article)
        
        # Fallback to full_content
        if len(found_articles) < len(test_case['expected_articles']):
            full_content = doc.get('full_content', '')
            if full_content:
                for article in test_case['expected_articles']:
                    if f"Article {article}" in full_content or f"article {article}" in full_content.lower():
                        found_articles.add(article)
        
        # Fallback to content
        if len(found_articles) < len(test_case['expected_articles']):
            content = doc.get('content', '')
            for article in test_case['expected_articles']:
                if f"Article {article}" in content or f"article {article}" in content.lower():
                    found_articles.add(article)
        
        # Fallback to page
        if len(found_articles) < len(test_case['expected_articles']):
            doc_name = doc.get('document_name', '').lower()
            page = doc.get('page_number', '')
            
            if ('fundamental' in doc_name or 'constitution' in doc_name) and page and page != 'N/A':
                try:
                    page_int = int(page)
                    for article in test_case['expected_articles']:
                        if article in ARTICLE_PAGE_MAP and page_int in ARTICLE_PAGE_MAP[article]:
                            found_articles.add(article)
                except:
                    pass
    
    recall = len(found_articles) / len(test_case['expected_articles']) if test_case['expected_articles'] else 0.0
    return recall

def calculate_reciprocal_rank(test_case: Dict, retrieved_docs_metadata: List[Dict]) -> float:
    """✅ TYPE-SAFE"""
    if not retrieved_docs_metadata:
        return 0.0
    
    for rank, doc in enumerate(retrieved_docs_metadata, start=1):
        # TYPE-SAFE: Handle articles_mentioned
        articles_mentioned = doc.get('articles_mentioned')
        
        if articles_mentioned is not None:
            if isinstance(articles_mentioned, str):
                try:
                    articles_mentioned = json.loads(articles_mentioned)
                except:
                    articles_mentioned = []
            
            if isinstance(articles_mentioned, list) and len(articles_mentioned) > 0:
                for article in test_case['expected_articles']:
                    if article in articles_mentioned:
                        return 1.0 / rank
        
        # Fallback to full_content
        full_content = doc.get('full_content', '')
        if full_content:
            for article in test_case['expected_articles']:
                if f"Article {article}" in full_content or f"article {article}" in full_content.lower():
                    return 1.0 / rank
        
        # Fallback to content
        content = doc.get('content', '')
        for article in test_case['expected_articles']:
            if f"Article {article}" in content or f"article {article}" in content.lower():
                return 1.0 / rank
        
        # Fallback to page
        doc_name = doc.get('document_name', '').lower()
        page = doc.get('page_number', '')
        
        if ('fundamental' in doc_name or 'constitution' in doc_name) and page and page != 'N/A':
            try:
                page_int = int(page)
                for article in test_case['expected_articles']:
                    if article in ARTICLE_PAGE_MAP and page_int in ARTICLE_PAGE_MAP[article]:
                        return 1.0 / rank
            except:
                pass
    
    return 0.0

# ==================== MAIN BENCHMARK ====================

def run_complete_benchmark():
    print("\n" + "="*80)
    print("🚀 VirLaw Complete Benchmark - TYPE-SAFE VERSION")
    print("="*80 + "\n")
    
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
        print("✅ Flask server is running\n")
    except:
        print("❌ Flask server not running!")
        return
    
    rate_limiter = RateLimiter(requests_per_minute=REQUESTS_PER_MINUTE)
    
    results = []
    benchmark_start = time.time()
    
    for test_case in TEST_SET:
        print(f"\n📊 Query {test_case['id']}/50: {test_case['query'][:50]}...")
        
        rate_limiter.wait_if_needed()
        query_start = time.time()
        
        try:
            response = requests.post(f"{BASE_URL}/ultimate-query", json={"question": test_case['query']}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - query_start
            response_text = data.get('response', '')
            sources = data.get('sources', [])
            confidence = data.get('confidence', 0.0)
            
            is_accurate = calculate_accuracy(test_case, response_text)
            is_hallucinated, hallucination_issues = check_hallucination(response_text, sources)
            precision_5 = calculate_precision_at_5(test_case, sources)
            recall_10 = calculate_recall_at_10(test_case, sources)
            mrr = calculate_reciprocal_rank(test_case, sources)
            has_pdf_citation = any(s.get('document_name', '').endswith('.pdf') for s in sources)
            
            result = {
                'query_id': test_case['id'],
                'query': test_case['query'],
                'latency_seconds': round(latency, 2),
                'is_accurate': is_accurate,
                'is_hallucinated': is_hallucinated,
                'hallucination_issues': hallucination_issues,
                'precision_at_5': round(precision_5, 3),
                'recall_at_10': round(recall_10, 3),
                'reciprocal_rank': round(mrr, 3),
                'confidence': round(confidence, 2),
                'has_pdf_citation': has_pdf_citation,
                'num_sources': len(sources),
            }
            
            print(f"   P@5: {precision_5:.0%} | R@10: {recall_10:.0%} | MRR: {mrr:.3f}")
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({'query_id': test_case['id'], 'query': test_case['query'], 'error': str(e)})
    
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        print("\n❌ No successful queries!")
        return
    
    accuracy = np.mean([r['is_accurate'] for r in valid_results]) * 100
    hallucination_rate = np.mean([r['is_hallucinated'] for r in valid_results]) * 100
    precision_5 = np.mean([r['precision_at_5'] for r in valid_results]) * 100
    recall_10 = np.mean([r['recall_at_10'] for r in valid_results]) * 100
    mrr = np.mean([r['reciprocal_rank'] for r in valid_results])
    
    if hallucination_rate == 0 and accuracy >= 95:
        trust_rating = "Very High"
    elif hallucination_rate <= 10 and accuracy >= 85:
        trust_rating = "High"
    elif hallucination_rate <= 20 and accuracy >= 70:
        trust_rating = "Medium"
    else:
        trust_rating = "Low"
    
    avg_latency = np.mean([r['latency_seconds'] for r in valid_results])
    
    total_time = time.time() - benchmark_start
    print(f"\n\n{'='*80}")
    print(f"📊 VIRLAW BENCHMARK RESULTS")
    print(f"{'='*80}\n")
    print(f"✅ Accuracy:        {accuracy:.1f}%")
    print(f"❌ Hallucination:   {hallucination_rate:.1f}%")
    print(f"🎯 Precision@5:     {precision_5:.1f}%")
    print(f"📈 Recall@10:       {recall_10:.1f}%")
    print(f"🏆 MRR:             {mrr:.3f}")
    print(f"⭐ Trust Rating:    {trust_rating}")
    print(f"⏱️  Avg Latency:     {avg_latency:.2f}s")
    print(f"⏱️  Total Runtime:   {total_time/60:.1f} min")
    print("="*80 + "\n")
    
    output = {
        'summary_metrics': {
            'accuracy_percent': round(accuracy, 1),
            'hallucination_rate_percent': round(hallucination_rate, 1),
            'precision_at_5_percent': round(precision_5, 1),
            'recall_at_10_percent': round(recall_10, 1),
            'mean_reciprocal_rank': round(mrr, 3),
            'expert_trust_rating': trust_rating,
            'avg_response_latency_seconds': round(avg_latency, 2),
        },
        'benchmark_info': {
            'total_queries': len(TEST_SET),
            'successful_queries': len(valid_results),
            'total_runtime_minutes': round(total_time / 60, 2),
            'actual_requests_per_minute': round(rate_limiter.get_stats()['requests_per_minute'], 2)
        },
        'detailed_results': results
    }
    
    with open('virlaw_benchmark_results_FINAL.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Results saved to: virlaw_benchmark_results_FINAL.json\n")
    
    return output

if __name__ == "__main__":
    run_complete_benchmark()
