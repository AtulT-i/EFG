# Adding Government Scheme Documents

To use this Government Scheme Eligibility Assistant, you need to add PDF documents containing information about government schemes.

## Steps to Add Documents

1. **Place PDF files in the `data/raw/` directory**

   ```bash
   cp /path/to/your/scheme-document.pdf data/raw/
   ```

2. **Run the ingestion pipeline**

   ```bash
   python main.py ingest
   ```

## Document Requirements

- **Format**: PDF files only (Phase 1)
- **Content**: Official government scheme documents with eligibility criteria, benefits, application processes, etc.
- **Naming**: Use descriptive names (e.g., `pm-kisan-scheme.pdf`, `ayushman-bharat.pdf`)

## Example Scheme Documents to Include

You can find government scheme documents from:

- Official Government portals (e.g., india.gov.in)
- Ministry websites
- Scheme-specific portals
- Official scheme brochures and guidelines

## Recommended Schemes to Add

Some popular Indian government schemes:

1. **PM-KISAN** (Pradhan Mantri Kisan Samman Nidhi)
2. **Ayushman Bharat** (Health Insurance)
3. **PMAY** (Pradhan Mantri Awas Yojana - Housing)
4. **MUDRA** (Micro Units Development & Refinance Agency)
5. **Atal Pension Yojana**
6. **PM Matru Vandana Yojana**
7. **Sukanya Samriddhi Yojana**
8. **Stand Up India**
9. **PM Svanidhi** (Street Vendor's Atmanirbhar Nidhi)
10. **PM Fasal Bima Yojana** (Crop Insurance)

## Document Structure

For best results, ensure your PDF documents include:

- Scheme name and overview
- Eligibility criteria (age, income, location, etc.)
- Benefits provided
- Application process
- Required documents
- Contact information

## Testing

After ingesting documents, test with sample queries:

```bash
# Interactive chat
python main.py chat

# Single query
python main.py query "What is the eligibility for PM-KISAN?"
```

## Troubleshooting

If documents are not being processed:

1. Check that PDFs are in `data/raw/` directory
2. Ensure PDFs contain readable text (not scanned images)
3. Check that your OpenAI API key is set in `.env`
4. Review console output for specific errors

For scanned PDFs, you may need to use OCR tools to extract text first.
