import re
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize spaCy for advanced NLP (optional)
nlp = spacy.load("en_core_web_sm")

def extract_features(email_df):
    """
    Extract phishing indicators from email data
    Returns DataFrame with features and labels
    """
    features = []
    
    for index, row in email_df.iterrows():
        email = row['raw_email']
        label = row['label']
        
        # Header analysis
        from_header = re.search(r'From:.*?<([^>]+)>', email)
        reply_to = re.search(r'Reply-To:.*?<([^>]+)>', email)
        
        # Body analysis
        body = get_email_body(email)
        soup = BeautifulSoup(body, 'html.parser')
        links = soup.find_all('a', href=True)
        
        # Feature dictionary
        feat = {
            'label': label,
            'num_links': len(links),
            'suspicious_domain': has_suspicious_domain(from_header.group(1) if from_header else ''),
            'reply_to_mismatch': (from_header and reply_to and 
                                 (from_header.group(1) != reply_to.group(1))),
            'num_urgent_keywords': count_keywords(body, ['urgent', 'immediately', 'verify']),
            'contains_html': bool(soup.find()),
            'link_domain_mismatch': check_link_domain_mismatch(from_header.group(1) if from_header else '', links),
            'text_entropy': calculate_text_entropy(body),
            'body_length': len(body)
        }
        
        features.append(feat)
    
    return pd.DataFrame(features)

def get_email_body(email):
    """Extract clean text body from raw email"""
    body = email.split('\n\n', 1)[-1]  # Split headers/body
    # Remove HTML tags and non-text content
    return BeautifulSoup(body, 'html.parser').get_text()

def has_suspicious_domain(email_address):
    """Check for domain anomalies"""
    domain = email_address.split('@')[-1] if '@' in email_address else ''
    return int(domain.count('.') > 2 or ' ' in domain)

def check_link_domain_mismatch(sender, links):
    """Verify if links match sender domain"""
    if not sender or '@' not in sender:
        return 0
    
    sender_domain = sender.split('@')[-1]
    for link in links:
        link_domain = urlparse(link['href']).netloc
        if sender_domain not in link_domain and link_domain != '':
            return 1
    return 0

def calculate_text_entropy(text):
    """Calculate text complexity score"""
    doc = nlp(text)
    return len([ent for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'GPE']])

def count_keywords(text, keywords):
    """Count security-related keywords"""
    return sum(1 for word in keywords if word.lower() in text.lower())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess email data for phishing detection')
    parser.add_argument('--data_path', type=str, required=True, help='Path to raw email data CSV')
    parser.add_argument('--output', type=str, required=True, help='Output path for processed data')
    
    args = parser.parse_args()
    
    # Load raw data (expected columns: 'raw_email', 'label')
    raw_df = pd.read_csv(args.data_path)
    
    # Process emails and extract features
    processed_df = extract_features(raw_df)
    
    # Save processed data
    processed_df.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output} with {len(processed_df)} entries")
