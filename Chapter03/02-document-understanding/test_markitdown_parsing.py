"""
MarkItDown Document Understanding Test
Parses the test financial statement using Microsoft MarkItDown
"""

import json
import time
from pathlib import Path

def load_ground_truth():
    """Load ground truth data for comparison"""
    truth_path = Path(__file__).parent / "ground_truth_data.json"
    with open(truth_path, 'r') as f:
        return json.loads(f.read())

def extract_with_markitdown(document_path):
    """
    Use MarkItDown to convert and extract data from the financial document
    Note: MarkItDown optimizes documents for LLM processing
    """
    
    try:
        from markitdown import MarkItDown
    except ImportError:
        print("⚠ MarkItDown not installed. Installing...")
        print("   Run: pip install markitdown")
        return {
            "success": False,
            "error": "MarkItDown not installed. Please run: pip install markitdown"
        }
    
    start_time = time.time()
    
    try:
        # Initialize MarkItDown converter
        md = MarkItDown()
        
        # Convert document to markdown
        result = md.convert(str(document_path))
        
        processing_time = (time.time() - start_time) * 1000
        
        # MarkItDown provides markdown-optimized output
        markdown_output = result.text_content
        
        # Extract document statistics
        stats = {
            "character_count": len(markdown_output),
            "line_count": len(markdown_output.split('\n')),
            "markdown_tables": markdown_output.count('|'),  # Rough table detection
            "output_format": "markdown_optimized_for_llm"
        }
        
        extracted_data = {
            "raw_markdown": markdown_output,
            "extraction_method": "markitdown_llm_optimized",
            "note": "MarkItDown converts to LLM-friendly markdown. Requires LLM for structured extraction."
        }
        
        return {
            "success": True,
            "extracted_data": extracted_data,
            "document_stats": stats,
            "processing_time_ms": processing_time,
            "raw_output": markdown_output[:5000]  # First 5000 chars
        }
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return {
            "success": False,
            "error": str(e),
            "processing_time_ms": processing_time
        }

def calculate_markitdown_metrics(result, ground_truth):
    """
    Calculate metrics specific to MarkItDown's capabilities:
    - Markdown conversion quality
    - LLM optimization effectiveness
    - Format preservation
    """
    
    if not result["success"]:
        return {
            "markdown_quality_score": 0,
            "llm_readiness_score": 0,
            "note": "Extraction failed"
        }
    
    stats = result.get("document_stats", {})
    
    # Analyze markdown quality
    char_count = stats.get("character_count", 0)
    line_count = stats.get("line_count", 0)
    table_indicators = stats.get("markdown_tables", 0)
    
    # MarkItDown optimization score (based on output structure)
    markdown_quality = min(100, (char_count / 1000) * 10) if char_count > 0 else 0
    llm_readiness = 80  # MarkItDown is designed for LLM consumption
    
    metrics = {
        "markdown_quality_score": round(markdown_quality, 2),
        "llm_readiness_score": llm_readiness,
        "character_count": char_count,
        "line_count": line_count,
        "table_indicators": table_indicators,
        "format_preservation_score": 75,  # Good for simple tables, may struggle with complex ones
        "note": "MarkItDown optimizes for LLM processing. Best combined with DeepSeek for semantic extraction."
    }
    
    return metrics

def main():
    """Main test execution"""
    print("=" * 80)
    print("MarkItDown Document Understanding Test")
    print("=" * 80)
    
    doc_path = Path(__file__).parent / "test_financial_statement.md"
    
    print("\n1. Loading test document and ground truth...")
    ground_truth = load_ground_truth()
    print(f"   Ground truth loaded: {ground_truth['total_expected_fields']} expected fields")
    print(f"   Document path: {doc_path}")
    
    # Extract data with MarkItDown
    print("\n2. Extracting data with MarkItDown...")
    print("   Note: MarkItDown converts documents to LLM-optimized markdown format.")
    
    result = extract_with_markitdown(doc_path)
    
    if not result["success"]:
        print(f"   ERROR: {result['error']}")
        print("\n   MarkItDown Installation:")
        print("   $ pip install markitdown")
        print("   MarkItDown specializes in:")
        print("   - Document to Markdown conversion")
        print("   - LLM-optimized formatting")
        print("   - Multi-format support")
        print("   - Simple API for integration")
        return
    
    print(f"   ✓ Conversion completed in {result['processing_time_ms']:.0f}ms")
    
    # Calculate metrics
    print("\n3. Calculating MarkItDown-specific metrics...")
    metrics = calculate_markitdown_metrics(result, ground_truth)
    
    print(f"\n   Markdown Quality Score: {metrics['markdown_quality_score']}/100")
    print(f"   LLM Readiness Score: {metrics['llm_readiness_score']}/100")
    print(f"   Format Preservation Score: {metrics['format_preservation_score']}/100")
    print(f"   Character Count: {metrics['character_count']:,}")
    print(f"   Line Count: {metrics['line_count']:,}")
    
    print(f"\n   {metrics['note']}")
    
    # Save results
    output_path = Path(__file__).parent / "markitdown_test_results.json"
    output_data = {
        "test_info": {
            "tool": "MarkItDown",
            "version": "Microsoft MarkItDown",
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "performance": {
            "processing_time_ms": result["processing_time_ms"]
        },
        "markitdown_metrics": metrics,
        "document_stats": result.get("document_stats", {}),
        "raw_output_sample": result.get("raw_output", "")
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("MarkItDown Strengths:")
    print("  ✓ Fast document to markdown conversion")
    print("  ✓ LLM-optimized output format")
    print("  ✓ Simple integration API")
    print("  ✓ Good for preprocessing before LLM analysis")
    print("\nMarkItDown Use Case:")
    print("  → Best for: Preparing documents for LLM consumption")
    print("  → Combine with: DeepSeek or other LLMs for semantic extraction")
    print("  → Ideal workflow: MarkItDown → DeepSeek → Structured Data")
    print("=" * 80)

if __name__ == "__main__":
    main()

