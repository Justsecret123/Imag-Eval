
import argparse
import sys
from dotenv import load_dotenv

def main():
    """
    Entry point for the IMAG-EVAL command-line interface.

    The function parses user-provided commands and dispatches the
    corresponding benchmark workflow. Supported commands include prompt
    generation for a single configuration, prompt generation for all
    available configurations, and evaluation of generated images.

    The appropriate IMAG-EVAL pipeline is initialized through the
    `ExperimentRunner` class and executed according to the selected
    command-line arguments.

    Returns:
        int: Exit status code. Returns `0` on successful execution and
        `1` if an error occurs or no valid command is provided.
    """
    
    parser = argparse.ArgumentParser(description="T2I Evaluation Framework")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic prompts for a specific model')
    generate_parser.add_argument('--config', required=True, help='Path to prompt generation config')

    # 'Generate all' command
    subparsers.add_parser('generate_all', help='Generate synthetic prompts for\
                                                all the yaml files.')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate generated images')
    evaluate_parser.add_argument('--config', required=True, help='Path to evaluation config')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        from experiments.experiment_runner import ExperimentRunner
        runner = ExperimentRunner()

        if args.command == 'generate':
            runner.run_prompt_generation(args.config)
        elif args.command== 'generate_all':
            runner.run_prompt_generation_all()
        elif args.command == 'evaluate':
            runner.run_evaluation(args.config)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    load_dotenv()
    sys.exit(main())
