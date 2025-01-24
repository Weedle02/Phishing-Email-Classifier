import logging
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import Message
from predict import PhishingDetector
import argparse
import time
import os

class PhishingEmailHandler:
    def __init__(self, detector):
        self.detector = detector
        self.logger = logging.getLogger("phishing-server")
        self.phishing_log = "logs/phishing_attempts.log"
        
        # Create logs directory if not exists
        os.makedirs("logs", exist_ok=True)

    async def handle_DATA(self, server, session, envelope):
        start_time = time.time()
        
        # Decode email content
        email_content = envelope.content.decode('utf-8', errors='replace')
        peer = session.peer[0]
        
        try:
            # Perform phishing detection
            result = self.detector.predict(email_content)
            
            # Log the result
            log_entry = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'peer': peer,
                'mail_from': envelope.mail_from,
                'rcpt_tos': envelope.rcpt_tos,
                'phishing': result['phishing'],
                'confidence': result['confidence'],
                'decision_time': round(time.time() - start_time, 4)
            }
            
            if result['phishing']:
                self.logger.warning(f"Phishing detected from {peer}: {log_entry}")
                # Save full phishing email for analysis
                with open(self.phishing_log, "a") as f:
                    f.write(f"\n\n{'-'*40}\n")
                    f.write(f"[{log_entry['timestamp']}] Phishing attempt from {peer}\n")
                    f.write(f"From: {envelope.mail_from}\n")
                    f.write(f"To: {', '.join(envelope.rcpt_tos)}\n")
                    f.write(f"Confidence: {result['confidence']*100:.2f}%\n")
                    f.write(email_content)
            else:
                self.logger.info(f"Legitimate email from {peer}: {envelope.mail_from}")

            return '250 OK'
        except Exception as e:
            self.logger.error(f"Error processing email from {peer}: {str(e)}")
            return '451 Requested action aborted: local error in processing'

def start_server(host='localhost', port=8025, model_path='models/phishing_model.pkl'):
    # Initialize phishing detector
    try:
        detector = PhishingDetector(model_path)
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        return

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/server.log"),
            logging.StreamHandler()
        ]
    )

    # Create SMTP controller
    handler = PhishingEmailHandler(detector)
    controller = Controller(
        handler,
        hostname=host,
        port=port,
        decode_data=True
    )

    try:
        logging.info(f"Starting phishing detection server on {host}:{port}")
        controller.start()
        while True:
            time.sleep(300)  # Keep server alive
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        controller.stop()
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        controller.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Detection SMTP Server")
    parser.add_argument('--host', type=str, default='localhost',
                       help="Host interface to bind to")
    parser.add_argument('--port', type=int, default=8025,
                       help="Port to listen on")
    parser.add_argument('--model', type=str, default='models/phishing_model.pkl',
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    start_server(
        host=args.host,
        port=args.port,
        model_path=args.model
    )
