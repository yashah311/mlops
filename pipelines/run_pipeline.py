import os
import yaml
import datetime
import sagemaker
import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession

# For evaluation validation and conditional evaluation gates
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep

def create_and_run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "pipeline_config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    target_region = config["aws"]["region"]
    print(f"Initializing SageMaker Session explicitly targeting region: {target_region}")
    
    boto_session = boto3.Session(region_name=target_region)
    sagemaker_client = boto_session.client("sagemaker", region_name=target_region)
    custom_bucket = config["aws"].get("default_bucket")
    
    session = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=custom_bucket if custom_bucket and "YOUR_EXISTING_S3" not in custom_bucket else None
    )
    
    role = config["aws"].get("sagemaker_role_arn")
    print(f"Orchestrating workflow utilizing IAM Target Role: {role}")
    model_version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=config["infrastructure"]["training_instance"]
    )
    
    # -----------------------------------------------------------------
    # STEP 1: Model Training Definition
    # -----------------------------------------------------------------
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=os.path.join(base_dir, "src"),
        role=role,
        instance_type=instance_type_param,
        framework_version=config["infrastructure"]["framework_version"],
        sagemaker_session=session
    )
    
    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator
    )
    
    # -----------------------------------------------------------------
    # STEP 2: Model Evaluation (The Quality Gate 🧪)
    # -----------------------------------------------------------------
    eval_processor = SKLearnProcessor(
        framework_version=config["infrastructure"]["framework_version"],
        instance_type=config["infrastructure"]["inference_instance"],
        instance_count=1, # 🚀 THE DEFINITIVE STRUCTURAL FIX: Eliminates the null definition crash
        role=role,
        sagemaker_session=session
    )
    
    # Map the output JSON report so the pipeline engine can parse it natively
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        code=os.path.join(base_dir, "src", "evaluate.py"),
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="evaluation",
                destination=f"s3://{session.default_bucket()}/{config['aws']['pipeline_name']}/evaluation",
                source="/opt/ml/processing/evaluation"
            )
        ],
        property_files=[evaluation_report]
    )
    
    # -----------------------------------------------------------------
    # STEP 3: Model Package Definition Setup
    # -----------------------------------------------------------------
    model_instance = SKLearnModel(
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        source_dir=os.path.join(base_dir, "src"), 
        framework_version=config["infrastructure"]["framework_version"],
        sagemaker_session=session,
        env={
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": training_step.properties.ModelArtifacts.S3ModelArtifacts
        }
    )
    
    register_step = ModelStep(
        name="RegisterModelStep",
        step_args=model_instance.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=[config["infrastructure"]["inference_instance"]],
            transform_instances=[config["infrastructure"]["inference_instance"]],
            model_package_group_name=config["aws"]["model_package_group_name"],
            approval_status="Approved"
        )
    )
    
    # -----------------------------------------------------------------
    # STEP 4: Conditional Routing Evaluation Gate 🧭
    # -----------------------------------------------------------------
    # Extract the numerical value out of the structured JSON layout property file
    accuracy_expression = sagemaker.workflow.functions.JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="classification_metrics.accuracy.value"
    )
    
    # Enforce an absolute rule: Model must hit at least 80% accuracy to proceed
    condition_gate = ConditionGreaterThanOrEqualTo(
        left=accuracy_expression,
        right=0.80
    )
    
    condition_step = ConditionStep(
        name="CheckAccuracyGate",
        conditions=[condition_gate],
        if_steps=[register_step],  # Only registers if accuracy >= 80%
        else_steps=[]              # Stops executing if condition fails
    )
    
    # -----------------------------------------------------------------
    # Pipeline Assembly
    # -----------------------------------------------------------------
    pipeline = Pipeline(
        name=config["aws"]["pipeline_name"],
        parameters=[instance_type_param],
        steps=[training_step, evaluation_step, condition_step],
        sagemaker_session=session
    )
    
    print("Uploading and synchronizing conditional pipeline schema with AWS...")
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    
    # 🚀 PRODUCTION OPTIMIZATION: Fire-and-forget strategy
    print(f"\n🚀 Conditional Pipeline execution triggered successfully!")
    print(f"=========================================================================")
    print(f"Execution ARN: {execution.arn}")
    print(f"=========================================================================")
    print("✨ Handed off to AWS SageMaker Engine.")
    print("🎯 Track execution graph logs visually inside your AWS SageMaker Console UI.")
    
    # Removed execution.wait() to prevent GitHub Actions from blocking on runtime logs.
    print("✅ CI/CD Orchestration stage completed successfully.")

if __name__ == "__main__":
    create_and_run_pipeline()
