#!/usr/bin/env python3
"""
Terminal-based Evaluation Dashboard for Hate Speech Detection Models
Displays model performance metrics in a formatted terminal output
"""

import os
import sys
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init()

class ModelMetrics:
    """Store and display model performance metrics"""
    
    def __init__(self):
        # Model performance data from README.md
        self.models_data = {
            'FastText (Baseline)': {
                'recall': 50.0,
                'precision': 68.0,
                'f1_score': 58.0,
                'accuracy': 75.0,
                'description': 'Linear baseline model'
            },
            'BERTweet': {
                'recall': 62.195,
                'precision': 65.778,
                'f1_score': 63.936,
                'accuracy': 75.0,
                'description': 'BERTweet base model (135M params)'
            },
            'BERTweet_large': {
                'recall': 62.0,
                'precision': 77.0,
                'f1_score': 69.0,
                'accuracy': 81.0,
                'description': 'BERTweet large model (355M params)'
            },
            'DistilBERT': {
                'recall': 66.0,
                'precision': 60.0,
                'f1_score': 62.0,
                'accuracy': 77.0,
                'description': 'Lighter BERT variant, 10 epochs'
            },
            'BERT Base': {
                'recall': 62.0,
                'precision': 74.0,
                'f1_score': 68.0,
                'accuracy': 77.0,
                'description': 'BERT-base-uncased, 3 epochs'
            },
            'DistilRoBERTa': {
                'recall': 62.0,
                'precision': 75.0,
                'f1_score': 68.0,
                'accuracy': 81.0,
                'description': 'Best performing model, optimized hyperparameters'
            }
        }
        
        # Dataset information
        self.dataset_info = {
            'Train Data Size': 31652,
            'Test Data Size': 3957,
            'Validation Data Size': 3956,
            'Datasets': ['Davidson et al.', 'Berkeley'],
            'Classes': ['Hate Speech', 'Not Hate Speech']
        }

    def print_header(self):
        """Print dashboard header"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}{'HATE SPEECH DETECTION MODEL EVALUATION DASHBOARD':^80}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

    def print_dataset_info(self):
        """Print dataset information"""
        print(f"{Fore.YELLOW}DATASET INFORMATION{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'-'*20}{Style.RESET_ALL}")
        
        for key, value in self.dataset_info.items():
            if isinstance(value, list):
                print(f"{key}: {', '.join(map(str, value))}")
            else:
                print(f"{key}: {value:,}")
        print()

    def print_model_comparison_table(self):
        """Print comprehensive model comparison table"""
        print(f"{Fore.GREEN}MODEL PERFORMANCE COMPARISON{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'-'*30}{Style.RESET_ALL}")
        
        # Prepare data for tabulate
        table_data = []
        headers = ['Model', 'Recall', 'Precision', 'F1 Score', 'Accuracy']
        
        for model_name, metrics in self.models_data.items():
            table_data.append([
                model_name,
                f"{metrics['recall']:.2f}%",
                f"{metrics['precision']:.2f}%",
                f"{metrics['f1_score']:.2f}%",
                f"{metrics['accuracy']:.2f}%"
            ])
        
        # Sort by F1 score (descending)
        table_data.sort(key=lambda x: float(x[3].rstrip('%')), reverse=True)
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()

    def print_top_models(self):
        """Print top performing models"""
        print(f"{Fore.BLUE}TOP PERFORMING MODELS{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'-'*22}{Style.RESET_ALL}")
        
        # Sort models by F1 score
        sorted_models = sorted(
            self.models_data.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )
        
        for i, (model_name, metrics) in enumerate(sorted_models[:3], 1):
            print(f"{i}. {model_name}")
            print(f"   F1 Score: {Fore.GREEN}{metrics['f1_score']:.2f}%{Style.RESET_ALL}")
            print(f"   Accuracy: {metrics['accuracy']:.2f}%")
            print(f"   {metrics['description']}")
            print()

    def print_detailed_metrics(self):
        """Print detailed metrics for each model"""
        print(f"{Fore.MAGENTA}DETAILED MODEL METRICS{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'-'*24}{Style.RESET_ALL}")
        
        for model_name, metrics in self.models_data.items():
            print(f"\n{Fore.CYAN}{model_name}{Style.RESET_ALL}")
            print(f"{'─' * len(model_name)}")
            print(f"Recall:     {metrics['recall']:.2f}%")
            print(f"Precision:  {metrics['precision']:.2f}%")
            print(f"F1 Score:   {metrics['f1_score']:.2f}%")
            print(f"Accuracy:   {metrics['accuracy']:.2f}%")
            print(f"Description: {metrics['description']}")

    def print_best_model_highlight(self):
        """Highlight the best performing model"""
        best_model = max(self.models_data.items(), key=lambda x: x[1]['f1_score'])
        model_name, metrics = best_model
        
        print(f"\n{Fore.RED}{'='*60}")
        print(f"{Fore.RED}{'BEST PERFORMING MODEL':^60}")
        print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}{model_name}{Style.RESET_ALL}")
        print(f"F1 Score: {Fore.GREEN}{metrics['f1_score']:.2f}%{Style.RESET_ALL}")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Recall: {metrics['recall']:.2f}%")
        print(f"Precision: {metrics['precision']:.2f}%")
        print(f"\n{metrics['description']}")
        print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}\n")

    def print_training_summary(self):
        """Print training summary information"""
        print(f"{Fore.YELLOW}TRAINING SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'-'*16}{Style.RESET_ALL}")
        print("• All models trained on combined Davidson + Berkeley datasets")
        print("• Data split: 70% train, 15% validation, 15% test")
        print("• Evaluation metric: F1-score (primary), Accuracy (secondary)")
        print("• Best model: DistilRoBERTa with optimized hyperparameters")
        print("• Hyperparameters: Batch Size=64, Learning Rate=5e-5, Epochs=1")
        print()

    def run_dashboard(self):
        """Run the complete dashboard"""
        self.print_header()
        self.print_dataset_info()
        self.print_model_comparison_table()
        self.print_top_models()
        self.print_best_model_highlight()
        self.print_training_summary()
        self.print_detailed_metrics()

def main():
    """Main function to run the dashboard"""
    try:
        dashboard = ModelMetrics()
        dashboard.run_dashboard()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Dashboard terminated by user.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"{Fore.RED}Error running dashboard: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()
