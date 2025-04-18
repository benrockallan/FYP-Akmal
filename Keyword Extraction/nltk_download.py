import nltk

# List of essential NLTK resources
resources = [
    ('punkt', 'tokenizers/punkt'),
    ('stopwords', 'corpora/stopwords'),
    ('wordnet', 'corpora/wordnet'),
    ('omw-1.4', 'corpora/omw-1.4')  # Optional, for extended lemmatization support
]

for resource_name, resource_path in resources:
    try:
        nltk.data.find(resource_path)
        print(f"✅ '{resource_name}' already exists.")
    except LookupError:
        print(f"⏳ Downloading '{resource_name}'...")
        nltk.download(resource_name)

# Verification
print("\n🔍 Verifying downloads...")
for resource_name, resource_path in resources:
    try:
        nltk.data.find(resource_path)
        print(f"✅ '{resource_name}' successfully downloaded and verified.")
    except LookupError:
        print(f"❌ '{resource_name}' download failed. Please try manually.")
