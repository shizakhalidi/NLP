import os
import logging
import joblib
import argparse
import torch
from scripts.data_loader import load_datasets
from scripts.preprocessor import prepare_dataset
from scripts.tfidf_features import create_tfidf_features, create_separate_tfidf_features
from scripts.ngrams_features import create_ngram_tfidf_features
from scripts.train_logreg import train_evaluate_logreg
from scripts.train_bilstm import train_evaluate_bilstm
from scripts.train_transformer import train_evaluate_modernbert

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train clickbait classifier')
    # Main selection argument
    parser.add_argument('--model', type=str, default='logreg', 
                        choices=['logreg', 'bilstm', 'modernbert', 'all'],
                        help='Model to train (default: logreg)')
    
    # TF-IDF feature options (for traditional models)
    parser.add_argument('--features', type=str, default='tfidf',
                        choices=['tfidf', 'separate', 'ngrams'],
                        help='Feature type for LogReg (default: tfidf)')
    parser.add_argument('--max-features', type=int, default=10000,
                        help='Maximum features to extract (default: 10000)')
    
    # ModernBERT options
    parser.add_argument('--modernbert-model', type=str, default='answerdotai/ModernBERT-base',
                        help='ModernBERT model to use (default: answerdotai/ModernBERT-base)')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum sequence length for ModernBERT (default: 128)')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate for ModernBERT (default: 5e-5)')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of epochs for ModernBERT (default: 3)')
    
    # Training speed option
    parser.add_argument('--quick', action='store_true',
                        help='Use quick training settings for faster results')
    
    args = parser.parse_args()
    
    # Define paths
    train_path = "data/train.jsonl"
    val_path = "data/validation.jsonl"
    
    # Load and preprocess datasets
    train_df, val_df = load_datasets(train_path, val_path)
    train_df = prepare_dataset(train_df)
    val_df = prepare_dataset(val_df)
    
    # Train LogReg model if specified
    if args.model in ['logreg', 'all']:
        logger.info("=== Training Logistic Regression Model ===")
        
        # Create features based on selected feature type
        logger.info(f"Creating features: {args.features} with max_features={args.max_features}")
        
        if args.features == 'ngrams':
            X_train, X_val, vectorizer = create_ngram_tfidf_features(
                train_df, val_df, max_features=args.max_features
            )
            model_path = f'models/classifier/logreg_ngrams_{args.max_features}.pkl'
        elif args.features == 'separate':
            X_train, X_val = create_separate_tfidf_features(
                train_df, val_df, max_features=args.max_features
            )
            model_path = f'models/classifier/logreg_separate_{args.max_features}.pkl'
        else:  # default tfidf
            X_train, X_val = create_tfidf_features(
                train_df, val_df, max_features=args.max_features
            )
            model_path = f'models/classifier/logreg_tfidf_{args.max_features}.pkl'
        
        # Get labels
        y_train = train_df['label']
        y_val = val_df['label']
        
        # Train and evaluate
        logger.info("Training logistic regression...")
        model = train_evaluate_logreg(X_train, y_train, X_val, y_val)
        
        # Save model
        os.makedirs('models/classifier', exist_ok=True)
        logger.info(f"Saving model to {model_path}...")
        joblib.dump(model, model_path)
        
        logger.info("Logistic Regression training completed successfully")
    
    # Train BiLSTM model if specified
    if args.model in ['bilstm', 'all']:
        logger.info("\n=== Training BiLSTM Model ===")
        
        model_dir = 'models/bilstm'
        os.makedirs(model_dir, exist_ok=True)
        
        # Use simpler parameters if quick mode is enabled
        if args.quick:
            bilstm_params = {
                'max_features': 10000,
                'max_len': 50,
                'embed_dim': 100,
                'lstm_units': 64,
                'batch_size': 64,
                'epochs': 5,
                'model_dir': model_dir
            }
            logger.info("Using quick training settings")
        else:
            # Default parameters for better results
            bilstm_params = {
                'max_features': args.max_features * 2,  # More vocab for deep learning
                'max_len': 100,
                'embed_dim': 300,
                'lstm_units': 128,
                'batch_size': 32,
                'epochs': 15,
                'model_dir': model_dir
            }
        
        # Train and evaluate BiLSTM
        model, word_to_idx = train_evaluate_bilstm(
            train_df=train_df,
            val_df=val_df,
            **bilstm_params
        )
        
        logger.info("BiLSTM training completed successfully")
    
    # Train ModernBERT model if specified
    if args.model in ['modernbert', 'all']:
        logger.info("\n=== Training ModernBERT Model ===")
        
        model_dir = 'models/modernbert'
        os.makedirs(model_dir, exist_ok=True)
        
        # Configure ModernBERT parameters
        modernbert_params = {
            'model_name': args.modernbert_model,
            'output_dir': model_dir,
            'max_length': args.max_length,
            'batch_size': 16,  # Reduced batch size for transformer models
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'fp16': torch.cuda.is_available(),  # Use mixed precision if CUDA is available
            'quick_mode': args.quick  # Use quick mode if specified
        }
        
        # Train and evaluate ModernBERT
        model, tokenizer = train_evaluate_modernbert(
            train_df=train_df,
            val_df=val_df,
            **modernbert_params
        )
        
        logger.info("ModernBERT training completed successfully")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()