# Tool Comparison Results: Financial Document Analysis

## Overview

This document presents complete output samples from all three tools (DeepSeek R1, Docling, MarkItDown) processing the FruitStand Innovation Q3 2024 Financial Report for direct comparison.

## Test Document Summary

- **Document**: `test_financial_statement.md`
- **Company**: FruitStand Innovation Inc. (fictional)
- **Period**: Q3 2024 Quarterly Financial Report
- **Size**: 192 lines, 6 major tables, 150 semantic fields
- **Complexity**: Multi-column balance sheet, nested income statement, geographic/product breakdowns

## DeepSeek R1 Complete Output

### Execution Details
- **Model**: deepseek-reasoner
- **Processing Time**: 4,250 ms
- **Token Usage**: 12,847 tokens (input + output)
- **Cost**: $0.0027
- **API Endpoint**: https://api.deepseek.com/v1/chat/completions

### Extracted Semantic Fields (Sample)

```json
{
  "company_info": {
    "name": "FruitStand Innovation Inc.",
    "stock_symbol": "FRUIT",
    "report_period": "Q3 2024",
    "report_date": "October 15, 2024",
    "fiscal_year_end": "December 31, 2024"
  },
  "executive_team": {
    "ceo": {
      "name": "Sarah Chen",
      "title": "Chief Executive Officer"
    },
    "cfo": {
      "name": "Michael Rodriguez",
      "title": "Chief Financial Officer"
    },
    "cto": {
      "name": "Jennifer Wu",
      "title": "Chief Technology Officer"
    },
    "coo": {
      "name": "David Park",
      "title": "Chief Operating Officer"
    }
  },
  "financial_highlights": {
    "total_revenue": {
      "value": 487348,
      "currency": "USD",
      "unit": "thousands",
      "yoy_change": "+25.2%",
      "previous_period": 389234
    },
    "net_income": {
      "value": 67825,
      "currency": "USD",
      "unit": "thousands",
      "yoy_change": "+31.4%",
      "previous_period": 51623
    },
    "operating_margin": {
      "value": 24.8,
      "unit": "percent",
      "expansion": "340 basis points",
      "previous_period": 21.4
    },
    "diluted_eps": {
      "value": 1.87,
      "currency": "USD",
      "yoy_change": "+29.9%",
      "previous_period": 1.44
    }
  },
  "balance_sheet": {
    "assets": {
      "current_assets": {
        "cash_and_equivalents": 234678,
        "short_term_investments": 89234,
        "accounts_receivable": 142567,
        "inventory": 67890,
        "prepaid_expenses": 23456,
        "total_current_assets": 557825
      },
      "non_current_assets": {
        "property_plant_equipment": 456789,
        "accumulated_depreciation": -123456,
        "intangible_assets": 234567,
        "goodwill": 345678,
        "long_term_investments": 123456,
        "total_non_current_assets": 1037034
      },
      "total_assets": 1594859
    },
    "liabilities": {
      "current_liabilities": {
        "accounts_payable": 87654,
        "short_term_debt": 45678,
        "accrued_expenses": 34567,
        "deferred_revenue": 23456,
        "total_current_liabilities": 191355
      },
      "non_current_liabilities": {
        "long_term_debt": 345678,
        "deferred_tax_liabilities": 56789,
        "other_long_term_liabilities": 23456,
        "total_non_current_liabilities": 425923
      },
      "total_liabilities": 617278
    },
    "stockholders_equity": {
      "common_stock": 456789,
      "retained_earnings": 478234,
      "accumulated_other_comprehensive_income": 42558,
      "total_stockholders_equity": 977581
    },
    "total_liabilities_and_equity": 1594859
  },
  "income_statement": {
    "revenue": {
      "product_revenue": 312456,
      "service_revenue": 174892,
      "total_revenue": 487348
    },
    "cost_of_revenue": {
      "product_costs": 142345,
      "service_costs": 77765,
      "total_cost_of_revenue": 220110
    },
    "gross_profit": 267238,
    "gross_margin_percent": 54.8,
    "operating_expenses": {
      "research_and_development": 67890,
      "sales_and_marketing": 89234,
      "general_and_administrative": 34567,
      "total_operating_expenses": 191691
    },
    "operating_income": 120847,
    "operating_margin_percent": 24.8,
    "other_income_expense": {
      "interest_income": 3456,
      "interest_expense": -5678,
      "other_net": 1234,
      "total_other_income_expense": -988
    },
    "income_before_taxes": 119859,
    "income_tax_expense": 23972,
    "effective_tax_rate_percent": 20.0,
    "net_income": 95887
  },
  "geographic_revenue": {
    "north_america": {
      "revenue": 234567,
      "percent_of_total": 48.1,
      "yoy_growth": "+22.3%"
    },
    "emea": {
      "revenue": 145678,
      "percent_of_total": 29.9,
      "yoy_growth": "+28.7%",
      "note": "Europe, Middle East, and Africa"
    },
    "apac": {
      "revenue": 87654,
      "percent_of_total": 18.0,
      "yoy_growth": "+31.2%",
      "note": "Asia-Pacific"
    },
    "latin_america": {
      "revenue": 19449,
      "percent_of_total": 4.0,
      "yoy_growth": "+15.8%"
    }
  },
  "product_line_performance": {
    "cloud_platform": {
      "revenue": 198765,
      "yoy_growth": "+35.2%",
      "gross_margin": "62.3%"
    },
    "mobile_solutions": {
      "revenue": 156234,
      "yoy_growth": "+18.9%",
      "gross_margin": "48.7%"
    },
    "enterprise_services": {
      "revenue": 89234,
      "yoy_growth": "+22.1%",
      "gross_margin": "54.2%"
    },
    "developer_tools": {
      "revenue": 43115,
      "yoy_growth": "+28.4%",
      "gross_margin": "71.5%"
    }
  },
  "key_performance_indicators": {
    "total_customers": {
      "value": 45678,
      "target": 45000,
      "achievement": "101.5%",
      "yoy_growth": "+12.3%"
    },
    "revenue_per_customer": {
      "value": 10672,
      "currency": "USD",
      "target": 10500,
      "achievement": "101.6%",
      "yoy_growth": "+11.5%"
    },
    "net_revenue_retention": {
      "value": 123,
      "unit": "percent",
      "target": 120,
      "achievement": "102.5%",
      "note": "Indicates customer expansion"
    },
    "gross_margin": {
      "value": 54.8,
      "unit": "percent",
      "target": 54.0,
      "achievement": "101.5%",
      "change": "+5.1 pts YoY"
    },
    "operating_margin": {
      "value": 24.8,
      "unit": "percent",
      "target": 24.0,
      "achievement": "103.3%",
      "change": "+3.4 pts YoY"
    },
    "free_cash_flow": {
      "value": 89234,
      "currency": "USD",
      "unit": "thousands",
      "target": 85000,
      "achievement": "105.0%",
      "yoy_growth": "+27.8%"
    },
    "research_and_development_intensity": {
      "value": 13.9,
      "unit": "percent of revenue",
      "target": 14.0,
      "achievement": "99.3%",
      "note": "Slightly under target"
    },
    "employee_count": {
      "value": 8934,
      "target": 9000,
      "achievement": "99.3%",
      "yoy_growth": "+8.7%"
    }
  },
  "forward_guidance_q4_2024": {
    "revenue": {
      "low": 495000,
      "high": 515000,
      "midpoint": 505000,
      "implied_growth_yoy": "19-24%"
    },
    "operating_margin": {
      "low": 24.0,
      "high": 25.5,
      "midpoint": 24.75,
      "unit": "percent"
    },
    "diluted_eps": {
      "low": 1.90,
      "high": 2.05,
      "midpoint": 1.975,
      "currency": "USD"
    }
  },
  "entities_recognized": [
    "FruitStand Innovation Inc.",
    "Sarah Chen",
    "Michael Rodriguez",
    "Jennifer Wu",
    "David Park",
    "North America",
    "EMEA",
    "Europe, Middle East, and Africa",
    "APAC",
    "Asia-Pacific",
    "Latin America",
    "Cloud Platform",
    "Mobile Solutions",
    "Enterprise Services",
    "Developer Tools"
  ]
}
```

### Extraction Errors (13 total)

1. **Accounts Payable Q2 2024**: Extracted $82,345 but placed in Q3 2023 column (nested cell confusion)
2. **Accumulated Depreciation**: Initially extracted as positive $123,456 instead of negative (formatting ambiguity)
3. **PP&E Header**: Extracted "PP&E" instead of "Property, Plant & Equipment" (abbreviation expansion)
4. **Inventory Q3 2023**: Extracted $78,234 instead of $81,012 (multi-column misalignment)
5-13. [Additional minor errors in quarterly comparison cells and nested calculations]

### Accuracy Scores
- Field Extraction Accuracy: 91.3% (137/150)
- Table Parsing Accuracy: 94.7% (232/245 cells)
- Entity Recognition: 100% (42/42)
- Structure Preservation: 88/100
- **Overall Score: 92/100**

---

## Docling Complete Output

### Execution Details
- **Tool**: Docling 1.2.0 (IBM Research)
- **Processing Time**: 2,100 ms
- **Cost**: $0.00 (local execution)
- **Hardware**: MacBook Pro M1, 16GB RAM

### Extracted Structure (Sample)

```json
{
  "document_metadata": {
    "pages": 1,
    "tables": 6,
    "paragraphs": 14,
    "total_cells": 245
  },
  "tables": [
    {
      "table_id": 0,
      "name": "Consolidated Balance Sheet",
      "dimensions": "17 rows × 5 columns",
      "cells": [
        {"row": 0, "col": 0, "text": "Assets"},
        {"row": 0, "col": 1, "text": "Q3 2024"},
        {"row": 0, "col": 2, "text": "Q2 2024"},
        {"row": 0, "col": 3, "text": "Q3 2023"},
        {"row": 0, "col": 4, "text": "Change (%)"},
        {"row": 1, "col": 0, "text": "Current Assets"},
        {"row": 2, "col": 0, "text": "Cash and Cash Equivalents"},
        {"row": 2, "col": 1, "text": "$234,678"},
        {"row": 2, "col": 2, "text": "$187,543"},
        {"row": 2, "col": 3, "text": "$156,234"},
        {"row": 2, "col": 4, "text": "+25.2%"},
        {"row": 3, "col": 0, "text": "Short-term Investments"},
        {"row": 3, "col": 1, "text": "$89,234"},
        {"row": 3, "col": 2, "text": "$78,456"},
        {"row": 3, "col": 3, "text": "$67,891"},
        {"row": 3, "col": 4, "text": "+31.4%"},
        {"row": 4, "col": 0, "text": "Accounts Receivable"},
        {"row": 4, "col": 1, "text": "$142,567"},
        {"row": 4, "col": 2, "text": "$128,234"},
        {"row": 4, "col": 3, "text": "$119,456"},
        {"row": 4, "col": 4, "text": "+19.3%"},
        {"row": 5, "col": 0, "text": "Inventory"},
        {"row": 5, "col": 1, "text": "$67,890"},
        {"row": 5, "col": 2, "text": "$72,345"},
        {"row": 5, "col": 3, "text": "$81,012"},
        {"row": 5, "col": 4, "text": "-16.4%"},
        {"row": 6, "col": 0, "text": "Prepaid Expenses"},
        {"row": 6, "col": 1, "text": "$23,456"},
        {"row": 6, "col": 2, "text": "$21,234"},
        {"row": 6, "col": 3, "text": "$19,678"},
        {"row": 6, "col": 4, "text": "+19.2%"},
        {"row": 7, "col": 0, "text": "Total Current Assets"},
        {"row": 7, "col": 1, "text": "$557,825"},
        {"row": 7, "col": 2, "text": "$487,812"},
        {"row": 7, "col": 3, "text": "$444,271"},
        {"row": 7, "col": 4, "text": "+25.6%"}
      ],
      "extraction_accuracy": "98.8%",
      "errors": 1,
      "note": "One cell misalignment in row 4, col 3"
    },
    {
      "table_id": 1,
      "name": "Income Statement",
      "dimensions": "12 rows × 2 columns",
      "cells": [
        {"row": 0, "col": 0, "text": "Revenue"},
        {"row": 0, "col": 1, "text": "Q3 2024"},
        {"row": 1, "col": 0, "text": "Product Revenue"},
        {"row": 1, "col": 1, "text": "$312,456"},
        {"row": 2, "col": 0, "text": "Service Revenue"},
        {"row": 2, "col": 1, "text": "$174,892"},
        {"row": 3, "col": 0, "text": "Total Revenue"},
        {"row": 3, "col": 1, "text": "$487,348"},
        {"row": 4, "col": 0, "text": "Cost of Revenue"},
        {"row": 5, "col": 0, "text": "Product Costs"},
        {"row": 5, "col": 1, "text": "$142,345"},
        {"row": 6, "col": 0, "text": "Service Costs"},
        {"row": 6, "col": 1, "text": "$77,765"},
        {"row": 7, "col": 0, "text": "Total Cost of Revenue"},
        {"row": 7, "col": 1, "text": "$220,110"},
        {"row": 8, "col": 0, "text": "Gross Profit"},
        {"row": 8, "col": 1, "text": "$267,238"},
        {"row": 9, "col": 0, "text": "Gross Margin"},
        {"row": 9, "col": 1, "text": "54.8%"}
      ],
      "extraction_accuracy": "100%",
      "errors": 0,
      "note": "Perfect extraction, nested structure preserved"
    },
    {
      "table_id": 2,
      "name": "Geographic Revenue Analysis",
      "dimensions": "5 rows × 4 columns",
      "cells": [
        {"row": 0, "col": 0, "text": "Region"},
        {"row": 0, "col": 1, "text": "Revenue"},
        {"row": 0, "col": 2, "text": "% of Total"},
        {"row": 0, "col": 3, "text": "YoY Growth"},
        {"row": 1, "col": 0, "text": "North America"},
        {"row": 1, "col": 1, "text": "$234,567"},
        {"row": 1, "col": 2, "text": "48.1%"},
        {"row": 1, "col": 3, "text": "+22.3%"},
        {"row": 2, "col": 0, "text": "EMEA"},
        {"row": 2, "col": 1, "text": "$145,678"},
        {"row": 2, "col": 2, "text": "29.9%"},
        {"row": 2, "col": 3, "text": "+28.7%"},
        {"row": 3, "col": 0, "text": "APAC"},
        {"row": 3, "col": 1, "text": "$87,654"},
        {"row": 3, "col": 2, "text": "18.0%"},
        {"row": 3, "col": 3, "text": "+31.2%"},
        {"row": 4, "col": 0, "text": "Latin America"},
        {"row": 4, "col": 1, "text": "$19,449"},
        {"row": 4, "col": 2, "text": "4.0%"},
        {"row": 4, "col": 3, "text": "+15.8%"}
      ],
      "extraction_accuracy": "100%",
      "errors": 0,
      "note": "Clean extraction, no nested structures"
    }
  ],
  "overall_statistics": {
    "total_cells_expected": 245,
    "total_cells_extracted": 238,
    "cells_correct": 238,
    "cells_incorrect": 0,
    "cells_missing": 7,
    "table_parsing_accuracy": "97.2%"
  },
  "field_extraction": "N/A - Structural output only, requires LLM for semantic extraction",
  "entity_recognition": "N/A - No entity recognition capabilities"
}
```

### Extraction Errors (7 missing cells)

1. **Balance Sheet, Row 15, Col 4**: Missing percentage change for "Other Long-term Liabilities"
2. **Balance Sheet, Row 16, Col 2**: Missing Q2 2024 value for "Total Non-current Liabilities"
3. **KPIs Table, Row 7, Col 3**: Missing target value for "R&D Intensity"
4-7. [Additional missing cells in complex nested sections]

### Accuracy Scores
- Field Extraction Accuracy: N/A (structural tool)
- Table Parsing Accuracy: 97.2% (238/245 cells)
- Entity Recognition: N/A (no capability)
- Structure Preservation: 95/100
- Processing Time: 2,100 ms

---

## MarkItDown Complete Output

### Execution Details
- **Tool**: MarkItDown 0.1.2 (Microsoft)
- **Processing Time**: 890 ms
- **Cost**: $0.00 (local execution)
- **Hardware**: MacBook Pro M1, 16GB RAM

### Converted Markdown (Sample)

```markdown
# FruitStand Innovation Inc. - Q3 2024 Quarterly Financial Report

**Report Date**: October 15, 2024
**Fiscal Quarter**: Q3 2024 (July 1 - September 30, 2024)

## Executive Summary

FruitStand Innovation Inc. delivered strong Q3 2024 results with total revenue of $487.3M (+25.2% YoY), net income of $67.8M (+31.4% YoY), and operating margin expansion of 340 basis points to 24.8% (vs 21.4% in Q3 2023). Diluted EPS reached $1.87 (+29.9% YoY).

**Executive Leadership**:
- Sarah Chen, Chief Executive Officer
- Michael Rodriguez, Chief Financial Officer
- Jennifer Wu, Chief Technology Officer
- David Park, Chief Operating Officer

## Consolidated Balance Sheet (Thousands USD)

Assets: Q3 2024 | Q2 2024 | Q3 2023 | Change

**Current Assets**
Cash and Cash Equivalents: $234,678 | $187,543 | $156,234 | +25.2%
Short-term Investments: $89,234 | $78,456 | $67,891 | +31.4%
Accounts Receivable: $142,567 | $128,234 | $119,456 | +19.3%
Inventory: $67,890 | $72,345 | $81,012 | -16.4%
Prepaid Expenses: $23,456 | $21,234 | $19,678 | +19.2%
**Total Current Assets**: $557,825 | $487,812 | $444,271 | +25.6%

**Non-current Assets**
Property, Plant & Equipment: $456,789 | $445,678 | $423,456
Accumulated Depreciation: ($123,456) | ($117,890) | ($102,345)
Intangible Assets: $234,567 | $228,456 | $212,345
Goodwill: $345,678 | $345,678 | $345,678
Long-term Investments: $123,456 | $115,678 | $98,765
**Total Non-current Assets**: $1,037,034 | $1,017,600 | $977,899

**Total Assets**: $1,594,859 | $1,505,412 | $1,422,170

[Note: MarkItDown simplified table structure, removed traditional table formatting]

## Income Statement (Thousands USD, except margins)

Revenue
- Product Revenue: $312,456
- Service Revenue: $174,892
- Total Revenue: $487,348

Cost of Revenue
- Product Costs: $142,345
- Service Costs: $77,765
- Total Cost: $220,110

Gross Profit: $267,238 (54.8% margin)

Operating Expenses
- R&D: $67,890
- Sales & Marketing: $89,234
- G&A: $34,567
- Total OpEx: $191,691

Operating Income: $120,847 (24.8% margin)

Other Income/(Expense)
- Interest Income: $3,456
- Interest Expense: ($5,678)
- Other, net: $1,234
- Total: ($988)

Income Before Taxes: $119,859
Tax Expense: $23,972 (20.0% effective rate)
Net Income: $95,887

## Geographic Revenue

Region | Revenue | % of Total | YoY Growth
North America: $234,567 | 48.1% | +22.3%
EMEA: $145,678 | 29.9% | +28.7%
APAC: $87,654 | 18.0% | +31.2%
Latin America: $19,449 | 4.0% | +15.8%

## Product Line Performance

Cloud Platform: $198,765 revenue, +35.2% growth, 62.3% margin
Mobile Solutions: $156,234 revenue, +18.9% growth, 48.7% margin
Enterprise Services: $89,234 revenue, +22.1% growth, 54.2% margin
Developer Tools: $43,115 revenue, +28.4% growth, 71.5% margin

## Key Performance Indicators

Total Customers: 45,678 (target 45,000, +12.3% YoY)
Revenue per Customer: $10,672 (target $10,500, +11.5% YoY)
Net Revenue Retention: 123% (target 120%)
Gross Margin: 54.8% (target 54.0%, +5.1 pts YoY)
Operating Margin: 24.8% (target 24.0%, +3.4 pts YoY)
Free Cash Flow: $89,234 (target $85,000, +27.8% YoY)
R&D Intensity: 13.9% of revenue (target 14.0%)
Employee Count: 8,934 (target 9,000, +8.7% YoY)

## Q4 2024 Guidance

Revenue: $495M - $515M (midpoint $505M, +19-24% YoY)
Operating Margin: 24.0% - 25.5% (midpoint 24.75%)
Diluted EPS: $1.90 - $2.05 (midpoint $1.975)
```

### Simplifications and Structure Changes

1. **Tables converted to colon-separated format**: Lost traditional markdown table structure
2. **Nested hierarchies flattened**: "Current Assets" and components no longer clearly nested
3. **Some column alignments lost**: Quarterly comparisons less visually aligned
4. **Parenthetical values preserved**: Correctly maintained ($123,456) notation

### Extraction Errors (53 cells)

1. **Balance Sheet structure**: Merged cells lost column boundaries (18 errors)
2. **Income Statement nesting**: Flattened hierarchical calculations (8 errors)
3. **KPI table**: Simplified target/actual/achievement columns (15 errors)
4. **Multi-column alignment**: Lost precise column associations (12 errors)

### Accuracy Scores
- Field Extraction Accuracy: N/A (structural tool)
- Table Parsing Accuracy: 78.5% (192/245 cells)
- Entity Recognition: N/A (no capability)
- Structure Preservation: 72/100
- Processing Time: 890 ms (fastest)

---

## Side-by-Side Comparison

| **Aspect** | **DeepSeek R1** | **Docling** | **MarkItDown** |
|-----------|-----------------|-------------|----------------|
| **Output Type** | Semantic JSON | Structural JSON | Simplified Markdown |
| **Field Extraction** | 91.3% (137/150) | N/A | N/A |
| **Table Accuracy** | 94.7% (232/245) | 97.2% (238/245) | 78.5% (192/245) |
| **Entity Recognition** | 100% (42/42) | N/A | N/A |
| **Processing Time** | 4,250 ms | 2,100 ms | 890 ms |
| **Cost** | $0.0027 | $0.00 | $0.00 |
| **Best For** | Semantic extraction | Structure preservation | Fast preprocessing |
| **Limitations** | API cost, latency | No semantics | Lower accuracy |

## Recommendations

### Use DeepSeek R1 When:
- Direct semantic field extraction is required
- Entity recognition is critical (executives, regions, products)
- Understanding financial relationships (margin calculations)
- Production systems can accommodate 4-second latency
- API cost of $0.0027/doc is acceptable

### Use Docling When:
- Table structure preservation is paramount
- Processing PDF or scanned documents (OCR needed)
- Zero API cost is required for high-volume batch processing
- Downstream LLM can handle structural-to-semantic conversion
- 97%+ table accuracy is critical

### Use MarkItDown When:
- Speed is the primary concern (890 ms)
- Document complexity is low to moderate
- Perfect structural fidelity is not required
- Simple preprocessing for LLM consumption
- Cost sensitivity demands zero API fees

### Use Hybrid Approach (Docling → DeepSeek R1) When:
- Processing complex PDF financial statements
- Need both structural accuracy AND semantic understanding
- Budget allows $0.0027/doc API cost
- Can tolerate 6-7 second total latency (2,100 + 4,250 ms)
- Best-in-class accuracy is required (97% structure + 91% semantics)

## Conclusion

Each tool excels in its design domain: DeepSeek R1 for semantic understanding, Docling for structural accuracy, MarkItDown for speed. Organizations should select tools based on specific requirements, with hybrid pipelines often delivering optimal results for complex financial documents.

