
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Open the PDF
doc = pymupdf.open("sample.pdf")

# Step 2: Extract text from every page
all_text = ""

for page_num, page in enumerate(doc):
    text = page.get_text()
    all_text += text + "\n"

doc.close()

# Step 3: Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# Step 4: Split the extracted text
chunks = text_splitter.split_text(all_text)

# Step 5: Display the results
print("Total characters:", len(all_text))
print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])

print("\nCharacters in first chunk:", len(chunks[0]))

print("\nSecond chunk:")
print(chunks[1])

print("\nCharacters in second chunk:", len(chunks[1]))
