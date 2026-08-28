from pathlib import Path
import yaml
import traceback
import json
import os
from prompt_generation.prompt_generation import MetaPromptGeneration, PromptGeneration
from datetime import datetime

class ExperimentRunner:
    """
    Orchestrate IMAG-EVAL benchmark generation and evaluation workflows.

    This class provides the main entry points for creating benchmark
    prompts, running large-scale prompt generation experiments, and
    evaluating generated outputs. It handles configuration loading,
    component initialization, experiment execution, and result
    serialization.

    The workflow is designed to support reproducible IMAG-EVAL benchmark
    creation through YAML configuration files and modular metadata,
    language-model, and evaluation backends.
    """

    def run_prompt_generation(self, config_path):
        """
        Generate benchmark prompts from a configuration file.

        The function initializes the metadata generator and language model
        specified in the configuration, constructs meta-prompts for each
        skill combination and difficulty level, generates synthetic prompts,
        and exports the resulting benchmark collection as a JSON file.

        The generated output includes prompt metadata, underlying scene
        descriptions, skill configurations, and one or more synthetic prompt
        variants suitable for IMAG-EVAL evaluation.

        Args:
            config_path (str): Path to the YAML configuration file defining
                metadata generation, language model settings, skill
                combinations, difficulty levels, robustness options, and
                output parameters.

        Returns:
            None: Generated prompts are written to the output JSON file
            specified in the configuration.
        """

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Metadata initialization
        metadata_config = config['prompt_generation']['metadata_config']
        metadata_config_name = metadata_config.pop('name')

        module_name = f"metadata.{metadata_config_name}_metadata"
        class_name = f"{metadata_config_name.capitalize()}Metadata"

        module = __import__(module_name, fromlist=[class_name])
        metadata_class = getattr(module, class_name)
        metadata = metadata_class(**metadata_config)

        # LLM initialization
        llm_config = config['prompt_generation']['llm_model']
        llm_name = llm_config.pop('name')

        module_name = f"llm_interfaces.{llm_name}_llm"
        class_name = f"{llm_name.capitalize()}LLM"

        module = __import__(module_name, fromlist=[class_name])
        llm_class = getattr(module, class_name)
        llm = llm_class(**llm_config)

        # Initialize prompt generators
        meta_generator = MetaPromptGeneration(metadata)
        prompt_generator = PromptGeneration(llm)

        # Generate prompts for each skill configuration
        all_prompts = []
        samples_per_skill = config['prompt_generation']['samples_per_skill']

        for skill_config in config['prompt_generation']['skills']:
            level = skill_config['level']
            skills = skill_config['skills']
            robustness = skill_config.get('robustness', [])
            k = skill_config.get('k', 1)

            print(f"Generating {samples_per_skill} prompts for {skills} at {level} level")

            for i in range(samples_per_skill):
                # Generate meta prompt based on skills
                meta_prompt = meta_generator.generate_meta_prompt(skills, level)
                scene = meta_generator.scene

                # Generate synthetic prompt using LLM
                synthetic_prompt = prompt_generator.generate_prompt(meta_prompt, robustness, k)

                # Create prompt entry
                prompt_entry = {
                    "id": f"prompt_{i+1:03d}_{level}_{'+'.join(skills)}",
                    "skills": skills,
                    "level": level,
                    "meta_prompt": meta_prompt,
                    "scene": scene,
                    "synthetic_prompts": synthetic_prompt,
                }

                all_prompts.append(prompt_entry)

        # Create output data structure
        output_data = {
            "experiment_id": config['experiment']['name'],
            "timestamp": datetime.now().isoformat(),
            "prompts": all_prompts
        }

        # Save to output file
        output_file = config['output']['file']
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Generated {len(all_prompts)} prompts saved to: {output_file}")
        print("Prompt generation completed")
    
    def run_prompt_generation_all(self):
        """
        Generate benchmark prompts for all configuration files found in the
        prompt-generation directory.

        The function iterates through every YAML configuration file, checks
        whether the corresponding output file already exists, and launches
        prompt generation only for configurations that have not yet been
        processed. Errors are reported and recorded in a log file to allow
        interrupted generation runs to be resumed safely.

        Returns:
            None: Generated prompts are written to their respective output
            files, while execution logs are stored for any failed
            configurations.
        """

        # Set the base directory
        base_dir = Path("config/prompt_generation")
        # Get the file names
        files = [p for p in base_dir.glob("*") if p.is_file()]
        print(f"{len(files)} .yaml files.\n")
        # Loop through the files
        for path in files:
            # Check if the file doesn't exist
            json_path = Path("outputs/prompts") / path.with_suffix(".json").name

            if not json_path.exists():
                # Display the current file
                print(f"\n\nFile : {path}\n\n")
                try:
                    # Generate prompt for that file
                    self.run_prompt_generation(path)
                except Exception:
                    # Display error message 
                    traceback.print_exc()
                    # Write in the logs file
                    with open("prompt_generation/logs.txt", "a+") as log_file:
                        log_file.write(f"\nError for config {path}.\n")
            else: 
                print("\nSkipping the current file.\n")


    def run_evaluation(self, config_path):
        """Evaluate generated images"""
        print(f"Running evaluation with config: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
