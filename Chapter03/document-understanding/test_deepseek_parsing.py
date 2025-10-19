"""
DeepSeek R1 Document Understanding Test
Parses the test financial statement and extracts structured data
Uses direct API calls to DeepSeek R1 (not OpenAI client)
"""

import json
import time
import requests
import os
from pathlib import Path

# DeepSeek R1 API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-reasoner"  # DeepSeek R1

def load_document():
    """Load the test financial statement"""
    doc_path = Path(__file__).parent / "test_financial_statement.md"
    with open(doc_path, 'r') as f:
        return f.read()

def load_ground_truth():
    """Load ground truth data for comparison"""
    truth_path = Path(__file__).parent / "ground_truth_data.json"
    with open(truth_path, 'r') as f:
        return json.loads(f.read())

def extract_with_deepseek(document_text):
    """
    Use DeepSeek R1 to extract structured data from the financial document
    Uses direct API calls (not OpenAI client)
    """
    
    if not DEEPSEEK_API_KEY:
        return {
            "success": False,
            "error": "DEEPSEEK_API_KEY environment variable not set"
        }
    
    prompt = f"""Analyze this financial report and extract ALL key information in structured JSON format.

The document is a Q3 2024 financial report. Extract the following with EXACT values from the document:

1. **Document Metadata**: company_name, report_type, quarter, year, report_date, document_id, nasdaq_symbol
2. **Financial Highlights**: total_revenue, net_income, operating_margin, cash_and_equivalents, total_assets (with values and units)
3. **Balance Sheet - Current Assets** (Q3 2024, Q2 2024, Q3 2023):
   - cash_and_cash_equivalents
   - short_term_investments
   - accounts_receivable_net
   - inventory
   - prepaid_expenses
   - total_current_assets
4. **Balance Sheet - Non-Current Assets** (Q3 2024):
   - property_plant_equipment
   - accumulated_depreciation
   - net_ppe
   - intangible_assets
   - goodwill
   - long_term_investments
   - total_non_current_assets
5. **Balance Sheet - Liabilities** (Q3 2024):
   - All current liabilities items
   - All non-current liabilities items
   - Total liabilities
6. **Balance Sheet - Equity** (Q3 2024):
   - common_stock
   - shares_authorized
   - shares_outstanding
   - additional_paid_in_capital
   - retained_earnings
   - accumulated_other_comprehensive_loss
   - total_shareholders_equity
7. **Income Statement** (Q3 2024, with Q2 2024 and Q3 2023 comparisons):
   - All revenue items
   - All expense items
   - Operating metrics (margins, EPS, etc.)
8. **Revenue by Geographic Region** (Q3 2024) - ALL regions with revenue, %, growth, margin
9. **Revenue by Product Line** (Q3 2024) - ALL product lines with revenue, %, growth, margin
10. **Key Performance Indicators** (Q3 2024) - ALL 8 KPIs with exact values
11. **Management Team**: CEO and CFO names and titles
12. **Q4 2024 Guidance**: Revenue range, growth range, margin range, EPS range
13. **Contact Information**: email, phone, website

CRITICAL REQUIREMENTS:
- Extract EXACT numerical values (do not round)
- Preserve ALL decimal places
- Include negative numbers with minus sign
- Extract ALL table rows and columns
- Do not miss any data points
- Return valid JSON format

Document to analyze:
{document_text}

Return ONLY the JSON object with extracted data. Use this exact structure:
{{
  "document_metadata": {{}},
  "key_financial_highlights": {{}},
  "balance_sheet_current_assets": {{
    "q3_2024": {{}},
    "q2_2024": {{}},
    "q3_2023": {{}}
  }},
  "balance_sheet_non_current_assets": {{"q3_2024": {{}}}},
  "balance_sheet_totals": {{}},
  "balance_sheet_liabilities": {{"q3_2024": {{}}}},
  "balance_sheet_equity": {{"q3_2024": {{}}}},
  "income_statement": {{
    "q3_2024": {{}},
    "q2_2024": {{}},
    "q3_2023": {{}}
  }},
  "revenue_by_geographic_region_q3_2024": [],
  "revenue_by_product_line_q3_2024": [],
  "key_performance_indicators_q3_2024": {{}},
  "management_team": {{}},
  "q4_2024_guidance": {{}},
  "contact_information": {{}}
}}"""

    start_time = time.time()
    
    try:
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial document analysis expert. Extract data with 100% accuracy, preserving exact values."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,  # Deterministic output
            "max_tokens": 16000
        }
        
        # Make direct API call to DeepSeek R1
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=180  # 3 minute timeout
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API returned status {response.status_code}: {response.text}",
                "processing_time_ms": processing_time
            }
        
        # Parse API response
        api_response = response.json()
        
        # Extract content from response
        response_text = api_response["choices"][0]["message"]["content"]
        
        # Try to parse JSON
        # Sometimes the response might include markdown code blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        extracted_data = json.loads(response_text)
        
        # Get token usage from API response
        token_usage = {
            "prompt_tokens": api_response["usage"]["prompt_tokens"],
            "completion_tokens": api_response["usage"]["completion_tokens"],
            "total_tokens": api_response["usage"]["total_tokens"]
        }
        
        return {
            "success": True,
            "extracted_data": extracted_data,
            "processing_time_ms": processing_time,
            "token_usage": token_usage,
            "raw_response": response_text,
            "model_used": DEEPSEEK_MODEL
        }
        
    except requests.exceptions.RequestException as e:
        processing_time = (time.time() - start_time) * 1000
        return {
            "success": False,
            "error": f"API request failed: {str(e)}",
            "processing_time_ms": processing_time
        }
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return {
            "success": False,
            "error": str(e),
            "processing_time_ms": processing_time
        }

def calculate_accuracy_metrics(extracted, ground_truth):
    """
    Calculate accuracy metrics by comparing extracted data with ground truth
    """
    
    def safe_compare(extracted_val, truth_val):
        """Compare values with type handling"""
        if extracted_val is None or truth_val is None:
            return extracted_val == truth_val
        
        # Handle numeric comparisons with tolerance
        if isinstance(truth_val, (int, float)) and isinstance(extracted_val, (int, float)):
            return abs(extracted_val - truth_val) < 0.01
        
        # Handle string comparisons (case-insensitive)
        if isinstance(truth_val, str) and isinstance(extracted_val, str):
            return extracted_val.lower().strip() == truth_val.lower().strip()
        
        return extracted_val == truth_val
    
    def count_nested_fields(data):
        """Count total fields in nested structure"""
        if isinstance(data, dict):
            return sum(count_nested_fields(v) for v in data.values()) if data else 1
        elif isinstance(data, list):
            return sum(count_nested_fields(item) for item in data) if data else 1
        else:
            return 1
    
    def compare_nested(extracted_dict, truth_dict, path=""):
        """Recursively compare nested structures"""
        correct = 0
        total = 0
        errors = []
        
        for key, truth_val in truth_dict.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in extracted_dict:
                total += count_nested_fields(truth_val)
                errors.append({
                    "field": current_path,
                    "error_type": "missing_field",
                    "expected": truth_val,
                    "extracted": None
                })
                continue
            
            extracted_val = extracted_dict[key]
            
            if isinstance(truth_val, dict) and isinstance(extracted_val, dict):
                nested_correct, nested_total, nested_errors = compare_nested(extracted_val, truth_val, current_path)
                correct += nested_correct
                total += nested_total
                errors.extend(nested_errors)
            elif isinstance(truth_val, list) and isinstance(extracted_val, list):
                list_correct, list_total, list_errors = compare_lists(extracted_val, truth_val, current_path)
                correct += list_correct
                total += list_total
                errors.extend(list_errors)
            else:
                total += 1
                if safe_compare(extracted_val, truth_val):
                    correct += 1
                else:
                    errors.append({
                        "field": current_path,
                        "error_type": "incorrect_value",
                        "expected": truth_val,
                        "extracted": extracted_val
                    })
        
        return correct, total, errors
    
    def compare_lists(extracted_list, truth_list, path=""):
        """Compare lists of objects"""
        correct = 0
        total = 0
        errors = []
        
        for i, truth_item in enumerate(truth_list):
            if i >= len(extracted_list):
                total += count_nested_fields(truth_item)
                errors.append({
                    "field": f"{path}[{i}]",
                    "error_type": "missing_list_item",
                    "expected": truth_item,
                    "extracted": None
                })
                continue
            
            extracted_item = extracted_list[i]
            
            if isinstance(truth_item, dict):
                item_correct, item_total, item_errors = compare_nested(extracted_item, truth_item, f"{path}[{i}]")
                correct += item_correct
                total += item_total
                errors.extend(item_errors)
            else:
                total += 1
                if safe_compare(extracted_item, truth_item):
                    correct += 1
                else:
                    errors.append({
                        "field": f"{path}[{i}]",
                        "error_type": "incorrect_value",
                        "expected": truth_item,
                        "extracted": extracted_item
                    })
        
        return correct, total, errors
    
    # Perform comparison
    correct_fields, total_fields, extraction_errors = compare_nested(extracted, ground_truth)
    
    # Calculate metrics
    field_accuracy = (correct_fields / total_fields * 100) if total_fields > 0 else 0
    error_rate = ((total_fields - correct_fields) / total_fields * 100) if total_fields > 0 else 0
    
    return {
        "field_extraction_accuracy": round(field_accuracy, 2),
        "correct_fields": correct_fields,
        "total_fields": total_fields,
        "error_rate": round(error_rate, 2),
        "errors": extraction_errors[:20]  # Show first 20 errors
    }

def main():
    """Main test execution"""
    print("=" * 80)
    print("DeepSeek R1 Document Understanding Test")
    print("Using Direct API Calls (Not OpenAI Client)")
    print("=" * 80)
    
    # Load document and ground truth
    print("\n1. Loading test document and ground truth...")
    document = load_document()
    ground_truth = load_ground_truth()
    print(f"   Document loaded: {len(document)} characters")
    print(f"   Ground truth loaded: {ground_truth['total_expected_fields']} expected fields")
    
    # Extract data with DeepSeek R1
    print("\n2. Extracting data with DeepSeek R1...")
    print(f"   Model: {DEEPSEEK_MODEL}")
    print(f"   API: {DEEPSEEK_API_URL}")
    result = extract_with_deepseek(document)
    
    if not result["success"]:
        print(f"   ERROR: {result['error']}")
        return
    
    print(f"   ✓ Extraction completed in {result['processing_time_ms']:.0f}ms")
    print(f"   ✓ Tokens used: {result['token_usage']['total_tokens']:,}")
    
    # Calculate accuracy metrics
    print("\n3. Calculating accuracy metrics...")
    metrics = calculate_accuracy_metrics(result["extracted_data"], ground_truth)
    
    print(f"\n   Field Extraction Accuracy: {metrics['field_extraction_accuracy']}%")
    print(f"   Correct Fields: {metrics['correct_fields']}/{metrics['total_fields']}")
    print(f"   Error Rate: {metrics['error_rate']}%")
    
    # Show some errors if any
    if metrics['errors']:
        print(f"\n   First {min(5, len(metrics['errors']))} errors:")
        for error in metrics['errors'][:5]:
            print(f"      - {error['field']}: {error['error_type']}")
            print(f"        Expected: {error['expected']}")
            print(f"        Extracted: {error['extracted']}")
    
    # Save results
    output_path = Path(__file__).parent / "deepseek_test_results.json"
    output_data = {
        "test_info": {
            "tool": "DeepSeek R1",
            "model": result.get("model_used", DEEPSEEK_MODEL),
            "api_method": "Direct API Call (requests library)",
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "performance": {
            "processing_time_ms": result["processing_time_ms"],
            "token_usage": result["token_usage"]
        },
        "accuracy_metrics": metrics,
        "extracted_data": result["extracted_data"]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

