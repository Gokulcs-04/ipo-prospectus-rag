import json
from document import Document
from semantic_chunker import chunk_document

# 1. Load the raw extracted data directly from the filename
with open("ellenbarrie_rhp.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    
# 2. Let Pydantic automatically validate and build the objects
ipo_document = Document(**raw_data)

# 3. Pass the Document object into your chunker
chunks = chunk_document(ipo_document)

print(f"Successfully processed {ipo_document.company_name}!")
print(f"Generated {len(chunks)} semantic chunks from {ipo_document.page_count} pages.\n")

# 4. Preview how it handled the complex tables
print("--- Sample Table Chunk Preview ---")
table_chunks = [c for c in chunks if "Table title" in c.content]
if table_chunks:
    print(f"Chunk ID: {table_chunks[0].chunk_id}")
    print(f"Pages: {table_chunks[0].pages}")
    print(table_chunks[0].content[:500] + "...\n")