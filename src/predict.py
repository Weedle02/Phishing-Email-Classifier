import re
import joblib
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from preprocess import (get_email_body, has_suspicious_domain, 
                       check_link_domain_mismatch, calculate_text_entropy,
                       count_keywords)

class PhishingDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.feature_order = [  # Must match training feature order
            'num_links',
            'suspicious_domain',
            'reply_to_mismatch',
            'num_urgent_keywords',
            'contains_html',
            'link_domain_mismatch',
            'text_entropy',
            'body_length'
        ]
    
    def extract_features(self, raw_email):
        """Extract features from raw email text"""
        # Header analysis
        from_header = re.search(r'From:.*?<([^>]+)>', raw_email)
        reply_to = re.search(r'Reply-To:.*?<([^>]+)>', raw_email)
        
        # Body analysis
        body = get_email_body(raw_email)
        soup = BeautifulSoup(body, 'html.parser')
        links = soup.find_all('a', href=True)
        
        return {
            'num_links': len(links),
            'suspicious_domain': has_suspicious_domain(from_header.group(1) if from_header else ''),
            'reply_to_mismatch': int(bool(from_header and reply_to and 
                                       (from_header.group(1) != reply_to.group(1)))),
            'num_urgent_keywords': count_keywords(body, ['urgent', 'immediately', 'verify']),
            'contains_html': int(bool(soup.find())),
            'link_domain_mismatch': check_link_domain_mismatch(
                from_header.group(1) if from_header else '', links
            ),
            'text_熵': calculate_text_entropy(body),
            'body_length': len(body)
        }
    
    def predict(self, raw_email):
        """Predict phishing probability for a single email"""
        features = self.extract_features(raw_email)
        df = pd.DataFrame([features])[self.feature_order]
        proba = self.model.predict_proba(df)[0][1]
        return {
            'phishing': bool(self.model.predict(df)[0]),
            'confidence': round(proba, 4),
            'features': features
        }

def main():
    parser = argparse.ArgumentParser(description='Phishing Email Detector')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pkl file)')
    parser.add_argument('--email', type=str,
                       help='Raw email text for single prediction')
    parser.add_argument('--batch', type=str,
                       help='CSV file containing multiple emails (column: raw_email)')
    parser.add_argument('--output', type=str,
                       help='Output file for batch predictions')
    
    args = parser.parse_args()
    
    detector = PhishingDetector(args.model)
    
    if args.email:
        with open(args.email, 'r') as f:
            email_content = f.read()
        result = detector.predict(email_content)
        print(f"Phishing Detected: {result['phishing']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
    
    if args.batch:
        df = pd.read_csv(args.batch)
        results = []
        for _, row in df.iterrows():
            res = detector.predict(row['raw_email'])
            res['email_id'] = row.get('id', _)
            results.append(res)
        
        output_df = pd.DataFrame(results)
        if args.output:
            output_df.to_csv(args.output, index=False)
            print(f"Batch predictions saved to {args.output}")
        else:
            print(output_df.to_string(index=False))

if __name__ == "__main__":
    main()
