# Document Understanding Evaluation Framework

## Quantitative Metrics Definition

### 1. Field Extraction Accuracy
- **Definition**: Percentage of correctly extracted key-value pairs from structured documents
- **Formula**: (Correct Extractions / Total Expected Fields) × 100
- **Examples**: Company name, date, amounts, addresses, policy numbers

### 2. Table Parsing Accuracy
- **Row Accuracy**: Percentage of correctly extracted table rows
- **Column Accuracy**: Percentage of correctly preserved column structure
- **Cell Value Accuracy**: Percentage of correctly extracted cell values
- **Formula**: (Correct Cells / Total Cells) × 100

### 3. Entity Recognition Accuracy
- **Named Entities**: Accuracy in identifying organizations, persons, dates, monetary values
- **Formula**: (Correct Entities / Total Expected Entities) × 100

### 4. Structure Preservation Score
- **Headings**: Correct identification of document headings and hierarchy
- **Lists**: Preservation of bullet points and numbered lists
- **Formatting**: Retention of bold, italic, and special formatting
- **Formula**: (Preserved Structural Elements / Total Structural Elements) × 100

### 5. Performance Metrics
- **Processing Time**: Time taken to process document (milliseconds)
- **Cost per Document**: API call costs or computational resources
- **Token Usage**: For LLM-based solutions, number of tokens consumed

### 6. Error Rate Metrics
- **False Positives**: Incorrectly extracted information
- **False Negatives**: Missed information
- **Hallucinations**: Generated information not present in document
- **Formula**: (Errors / Total Extractions) × 100

## Test Document Specifications

### Test Document 1: Financial Statement
- **Complexity**: High
- **Elements**: 
  - Multi-column layout
  - Complex financial tables (5+ columns, 10+ rows)
  - Nested headings
  - Currency symbols and calculations
- **Ground Truth**: Pre-defined set of 50 key data points

### Test Document 2: Insurance Policy Document
- **Complexity**: Medium-High
- **Elements**:
  - Legal tables
  - Multi-level lists
  - Cross-references
  - Policy numbers and dates
- **Ground Truth**: Pre-defined set of 40 key data points

### Test Document 3: Multi-page Technical Report
- **Complexity**: High
- **Elements**:
  - Multiple sections with hierarchical headings
  - Technical diagrams (if multi-modal)
  - Data tables with merged cells
  - Footnotes and references
- **Ground Truth**: Pre-defined set of 60 key data points

## Evaluation Criteria

### Accuracy Thresholds
- **Excellent**: 95-100%
- **Good**: 85-94%
- **Acceptable**: 75-84%
- **Poor**: Below 75%

### Comparison Matrix Template

| Metric | DeepSeek R1 | Docling | MarkItDown |
|--------|-------------|---------|------------|
| Field Extraction Accuracy | % | % | % |
| Table Parsing Accuracy | % | % | % |
| Entity Recognition Accuracy | % | % | % |
| Structure Preservation | % | % | % |
| Avg Processing Time (ms) | ms | ms | ms |
| Cost per Document | $ | $ | $ |
| Overall Error Rate | % | % | % |
| **Overall Score** | /100 | /100 | /100 |

## Scoring Methodology

1. **Weighted Average**:
   - Field Extraction: 30%
   - Table Parsing: 25%
   - Entity Recognition: 20%
   - Structure Preservation: 15%
   - Performance: 10%

2. **Bonus Points**:
   - Multi-modal capabilities: +5 points
   - Real-time streaming: +3 points
   - Self-correction ability: +2 points

3. **Penalty Points**:
   - Hallucinations: -5 points per instance
   - Major structural errors: -3 points per instance
   - Security concerns: -10 points

## Testing Procedure

1. **Document Preparation**:
   - Create ground truth datasets
   - Annotate expected outputs
   - Define edge cases

2. **Tool Setup**:
   - Configure DeepSeek R1 with optimal prompts
   - Install and configure Docling
   - Install and configure MarkItDown

3. **Execution**:
   - Process each document with all three tools
   - Record timestamps and token usage
   - Collect raw outputs

4. **Analysis**:
   - Compare outputs against ground truth
   - Calculate all defined metrics
   - Document error patterns

5. **Reporting**:
   - Create comparison tables
   - Generate visualization charts
   - Write analysis commentary

