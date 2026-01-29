"""
Agent Evaluation Pipeline

Evaluates agent performance across three key dimensions:
1. Calculator selection: Validates correct medical calculator identification from final answer
2. Data extraction: Verifies accurate patient data retrieval from database queries
3. Calculation accuracy: Checks correctness of computed calculator values
"""
import os
import logging
import json
import asyncio
import argparse
import jsonlines
import pandas as pd

from tqdm import tqdm
from rich import print
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.llm import MyOpenAILLM



class Args:
    """
    Configuration class to handle command-line arguments and system settings.
    """
    def parseargs(self):
        parser = argparse.ArgumentParser(description="M1-Env Agentic Hospital Simulation")

        parser.add_argument('--task_path', type=str, default="../benchmark/tasks.jsonl")
        parser.add_argument('--evaled_model', type=str, default="claude-opus-4-5")
        
        parser.add_argument("--max_workers", type=int, default=16)
        parser.add_argument("--stats_only", action="store_true",
                            help="If set, the script will only read the output file to compute statistical metrics without performing API evaluation.")

        # Model API Configurations
        parser.add_argument('--model_name', type=str, default="DeepSeek-V3.1")
        parser.add_argument("--base_url", type=str, default="")
        parser.add_argument("--api_key", type=str, default="")
        
        self.pargs = parser.parse_args()
        
        # Map parsed arguments to class attributes
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)
            
    def __init__(self) -> None:
        self.parseargs()
        
        self.input_path = f"../outputs/{self.evaled_model}_results.jsonl"
        self.output_path = f"../outputs/{self.evaled_model}_results_{self.model_name}_evaled.jsonl"


args = Args()  



logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
    filename=f"./logs/{args.evaled_model}_results_evaluation.log",
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class Prompts:
    def __init__(self) -> None:
        
        self.extract_calcs_from_final_answer = """
You are a medical information extraction expert. Please identify and extract all medical calculators and their related values from the following text, outputting in JSON format.

Task Requirements:
1. Identify all medical calculators or scoring scales mentioned in the text
2. Extract each calculator's name, value, unit (if any), and clinical significance for the patient
3. If the text contains calculation processes or parameters, extract them as well

Output Format (JSON only, no additional content):
[
    {
      "name": "calculator_name",
      "full_name": "full name (if abbreviated)",
      "value": "numeric_value",
      "unit": "unit",
      "parameters": {
        "parameter_name": "parameter_value"
      },
      "interpretation": "clinical interpretation or significance",
      "context": "usage scenario in the original text"
    }
]

Important Notes:
- If the value is a range, preserve the complete range (e.g., "3-5")
- If any field information is unclear, mark it as "not specified"
- Maintain extraction accuracy; do not speculate or add information not present in the original text

Text to analyze:
{INSERT_CASE_HERE}
""".strip()

        self.compare_clacs_list = """
You are a medical calculator evaluation expert. Please compare the intern doctor's calculator evaluation results with the ground truth (gt) and output structured JSON format results.

Task Description:
1. Compare whether each calculator's name matches (names may have slight differences, such as abbreviations, capitalization, etc., please identify intelligently)
2. Compare whether the calculation results of matched calculators are consistent
3. Count the number of calculators correctly used by the doctor and the number of correct results

Input Format:
- gt (ground truth): List containing correct calculator names and results
- doctor_results (doctor's results): List containing calculator names selected by the doctor and calculation results

Please output strictly according to the following JSON format, do not output any other content:
{
  "comparison_details": [
    {
      "gt_calculator_name": "Calculator name in ground truth",
      "doctor_calculator_name": "Calculator name used by doctor (if match found)",
      "name_matched": true/false,
      "gt_result": "Ground truth result",
      "doctor_result": "Doctor's calculation result (if available)",
      "result_matched": true/false,
      "notes": "Remarks (e.g., names are similar but not exactly the same)"
    }
  ],
  "summary": {
    "total_gt_calculators": Total number of calculators,
    "correctly_identified_calculators": Number of correctly identified calculators,
    "correctly_calculated_results": Number of correct results,
    "missing_calculators": ["List of unidentified calculator names"],
    "extra_calculators": ["List of calculators additionally used by doctor"]
  }
}

Ground Truth:
{INSERT_GT_HERE}

Doctor Result:
{INSERT_RESULT_HERE}
""".strip()

        self.extract_data_from_sql_action = """
You are a specialized assistant for analyzing database query execution steps. Please analyze the following execution steps and extract database query-related information.

Execution steps:
{INSERT_CONTENT_HERE}

Please analyze these steps and extract the following information:
For patient with ID {INSERT_ID_HERE}, considering only the fields specified in calculator_fields: {INSERT_FIELD_HERE}

1. Query Intent: Which fields from calculator_fields are being attempted to extract from the database?
2. Query Results: Among the attempted calculator_fields, which extractions succeeded and which failed?
   - Successfully extracted fields with their corresponding values
   - Failed extractions with detailed reasons for failure

Please output in JSON format only, without any additional explanation or text. Use the following structure:
{
  "query_intent": {
    "target_data": ["List of fields being extracted that exist in calculator_fields"],
    "target_table": "Target table name",
    "query_condition": "Description of query conditions"
  },
  "query_result": {
    "success": {
        "field_name": "value",
        ...
    },
    "failure": {
      "field_name": "failure_reason",
      ...
    }
  },
  "execution_status": "success/failure"
}

Note: Only include fields that are present in the provided calculator_fields list. Ignore any other fields that may appear in the query but are not in calculator_fields.
""".strip()

        self.sql_data_compare = """
Compare the machine-extracted SQL data (sql_data) with the ground truth (gt) and output the comparison results in JSON format.

Task:
1. Identify which gt fields are covered in sql_data
2. For covered fields, determine if values match using the following flexible criteria:
   - Numerical values: Consider them matching if the core numeric value is the same, regardless of units, formatting, or additional descriptors (e.g., "0.08" matches "0.08 ng/mL")
   - Text values: Consider them matching if:
     - The gt_value is contained within or is a substring of the sql_value
     - The sql_value semantic meaning includes or encompasses the gt_value
     - For medical findings: if the sql_value contains the key finding mentioned in gt_value, even if sql_value provides more detail
   - Partial matches: If the essential information from gt_value is present in sql_value, consider it a match
3. Calculate overall extraction accuracy based on these flexible matching criteria

Important: Focus on whether the essential information is captured correctly, not on exact string matching. Medical data often varies in level of detail between sources.

Output JSON format only:
{
  "summary": {
    "gt_total": <number of fields in gt>,
    "extracted": <number of gt fields found in sql_data>,
    "correct": <number of fields with matching values>,
    "incorrect": <number of fields with wrong values>,
    "missing": <number of gt fields not found in sql_data>
  },
  "field_details": [
    {
      "field": "<field name>",
      "gt_value": "<value in gt>",
      "sql_value": "<value in sql_data or null if missing>",
      "match": true/false/null
    }
  ],
  "missing_fields": [<list of gt fields not extracted>],
  "incorrect_fields": [<list of fields with wrong values>]
}

Ground Truth (gt):
{INSERT_GT_HERE}

SQL Data (sql_data):
{INSERT_RESULT_HERE}
""".strip()


async def process_task_async(item: dict) -> None:
    """
    Process a single data item through the evaluation pipeline.
    
    Args:
        item (dict): The input dictionary containing agent history and results.
    """

    result = {
        "task_id": item["task_id"],
        "patient_id": None,
        "process_success": item["process_success"],
        "final_calculators": [],
        "calculators_compare_result": {},
        "data_from_sql": [],
        "sql_data_compare_result": {},
        "tools_times": {
            "run_sql": 0,
            "run_python": 0,
            "search": 0,
            "fetch": 0
        }
    }
    
    # Initialize the LLM wrapper
    llm = MyOpenAILLM(args.model_name, args.base_url, args.api_key)
    prompts = Prompts()
    
    # Load specific task details from the ground truth file
    task_item = {}
    with jsonlines.open(args.task_path) as reader:
        for obj in reader:
            if obj["task_id"] == item["task_id"]:
                task_item = obj
                break
    # Raise error if the task ID is not found in the ground truth file
    if not task_item:
        raise KeyError(f"Task ID {item['task_id']} not found in task_path")
    
    result["patient_id"] = task_item["patient_id"]
    
    # --- Step 1: Extract final_calculators from the agent's answer ---
    try:
        final_answer = item.get("final_answer", "") or ""
        final_answer = str(final_answer)
        ans = await llm.generate(prompts.extract_calcs_from_final_answer
                           .replace("INSERT_CASE_HERE", final_answer))
        cleaned_ans = ans.strip().replace("```json", "").replace("```", "").strip()
        ans_dict = json.loads(cleaned_ans)
        result["final_calculators"] = ans_dict
        logger.info(f"Task {obj['task_id']} --- Extracting Final Calculators:\n\n{ans_dict}")
    except Exception as e:
        result["final_calculators"] = []
        logger.info(f"Error extracting final calculators: {e}")
    
    # --- Step 2: Compare extracted calculators with Ground Truth ---
    try:
        gt_calc = [
            {
                "name": c["name"], 
                "final_answer": "" if not c["final_answer"] else c["final_answer"]
            } 
            for c in task_item["calculator_answers"]
        ]
        doc_result = [{"name":c["name"], "full_name":c["full_name"], "value":c["value"]
                        , "unit":c["unit"]} for c in result["final_calculators"]]
        ans = await llm.generate(prompts.compare_clacs_list
                            .replace("INSERT_GT_HERE", json.dumps(gt_calc, ensure_ascii=False))
                            .replace("INSERT_RESULT_HERE", json.dumps(doc_result, ensure_ascii=False)))
        cleaned_ans = ans.strip().replace("```json", "").replace("```", "").strip()
        ans_dict = json.loads(cleaned_ans)
        result["calculators_compare_result"] = ans_dict
        logger.info(f"Task {obj['task_id']} --- Comparing Calculators:\n\n{ans_dict}")
    except Exception as e:
        result["calculators_compare_result"] = {}
        logger.info(f"Error comparing calculators: {e}")
    
    
    # --- Step 3: Extract data from sql (Analyze SQL actions in history) ---
    cal_field_data = ""
    # Aggregate all required fields from ground truth calculators
    for c in task_item["calculator_answers"]:
        fields = ""
        for cal in c["inputs"]:
            fields += f"'field': {cal['field']};  "
        cal_field_data += "\n" + fields
    
    # Iterate through history to find SQL actions (ignoring the final answer step)
    action_list = item["history_list"][:-1]
    for i, step in enumerate(action_list):
        
        try:
            # Filter for SQL tools
            if "sql" in step["action"]["tool"]:
                pass
            else:
                continue
        
            ans = await llm.generate(prompts.extract_data_from_sql_action
                            .replace("INSERT_ID_HERE", result["patient_id"])
                            .replace("INSERT_CONTENT_HERE", json.dumps(step, ensure_ascii=False))
                            .replace("INSERT_FIELD_HERE", str(cal_field_data))
                            )
            cleaned_ans = ans.strip().replace("```json", "").replace("```", "").strip()
            ans_dict = json.loads(cleaned_ans)
            ans_dict["step"] = i+1
            result["data_from_sql"].append(ans_dict)
            logger.info(f"Task {obj['task_id']} --- Analyzing SQL:\n\n{ans_dict}")
        except Exception as e:
            logger.info(f"Error analyzing SQL step {i}: {e}")
            
    # --- Step 4: Compare SQL data extracted with Ground Truth ---
    try:
        sql_data = ""
        for sql in result["data_from_sql"]:
            sql_data += "\n" + str(sql["query_result"]["success"])
        gt_data = ""
        for c in task_item["calculator_answers"]:
            for calc_input in c["inputs"]:
                field = {
                    "field": calc_input["field"],
                    "value": calc_input["value"]
                }
                gt_data += " " + str(field)
            gt_data += "\n"
        ans = await llm.generate(prompts.sql_data_compare
                            .replace("INSERT_GT_HERE", json.dumps(gt_data, ensure_ascii=False))
                            .replace("INSERT_RESULT_HERE", json.dumps(sql_data, ensure_ascii=False)))
        cleaned_ans = ans.strip().replace("```json", "").replace("```", "").strip()
        ans_dict = json.loads(cleaned_ans)
        result["sql_data_compare_result"] = ans_dict
        logger.info(f"Task {obj['task_id']} --- Comparing SQL:\n\n{ans_dict}")
    except Exception as e:
        result["sql_data_compare_result"] = {}
        logger.info(f"Error comparing SQL data: {e}")
    
    # --- Step 5: Calculate tool usage frequency (tools_times) ---
    action_list = item["history_list"][:-1]
    tools_list = [a.get("action", {}).get("tool") for a in action_list]
    counts = Counter(tools_list)
    result["tools_times"]["run_sql"] = counts["run_read_only_sql"]
    result["tools_times"]["run_python"] = counts["run_python"]
    result["tools_times"]["search"] = counts["search"]
    result["tools_times"]["fetch"] = counts["fetch"]

    await llm.close()

    with jsonlines.open(args.output_path, "a") as writer:
        writer.write(result)
    
    return


def main(item: dict) -> None:
    """Thread worker entry point for concurrent task processing"""
    asyncio.run(process_task_async(item))


def extract_domain(scenario: str) -> str:
    """
    Extract domain from scenario string.

    Example scenario: "Clinical Medicine / Special Populations & General Tools / Clinical Pharmacology & Nursing"
    Returns: "Special Populations & General Tools"

    Args:
        scenario: The full scenario path string

    Returns:
        The domain (second level of the path)
    """
    if not scenario:
        return "Unknown"

    parts = [p.strip() for p in scenario.split('/')]

    if len(parts) >= 2:
        return parts[1]
    else:
        return "Unknown"


def analyze_evaluation_results_by_scenario():
    """
    Analyzes evaluation results grouped by scenario.
    Similar to metric.py functionality but integrated into evaluation.py.

    Outputs:
    1. Detailed CSV with domain-specific metrics
    2. Summary CSV with overall averages
    3. Rich formatted tables to console
    """
    detail_output_path = "./evaluation_by_scenario_detail.csv"
    summary_output_path = "./evaluation_by_scenario_summary.csv"

    # Check if files exist
    if not os.path.exists(args.output_path):
        print(f"Error: File not found at {args.output_path}")
        return

    if not os.path.exists(args.task_path):
        print(f"Error: Task file not found at {args.task_path}")
        return

    # Load task items for scenario mapping
    task_items = {}
    with jsonlines.open(args.task_path) as reader:
        for item in reader:
            task_items[item["task_id"]] = item

    # Domain-wise aggregation
    domain_metrics = defaultdict(lambda: {
        "count": 0,
        "calc_selection_acc": [],
        "calc_value_acc": [],
        "sql_extraction_acc": []
    })

    # Process evaluation results
    with jsonlines.open(args.output_path) as reader:
        for obj in reader:
            task_id = obj.get("task_id")

            if task_id not in task_items:
                print(f"Warning: Task {task_id} not found in task file, skipping...")
                continue

            task_item = task_items[task_id]

            # Extract domain from task category
            scenario = task_item.get("task_category", "")
            domain = extract_domain(scenario)

            # Calculate metrics for this task
            # Calculator Statistics
            total_gt_calcs = len(task_item.get("calculator_answers", []))

            correct_id_calcs = 0
            correct_val_calcs = 0

            try:
                for res in obj.get("calculators_compare_result", {}).get("comparison_details", []):
                    if res.get("name_matched") == True:
                        correct_id_calcs += 1
                    if res.get("result_matched") == True:
                        correct_val_calcs += 1
            except Exception:
                pass

            calc_selection_acc = (correct_id_calcs / total_gt_calcs) if total_gt_calcs > 0 else 0.0
            calc_value_acc = (correct_val_calcs / total_gt_calcs) if total_gt_calcs > 0 else 0.0

            # SQL Data Statistics
            sql_res = obj.get('sql_data_compare_result', {})
            sql_summary = sql_res.get('summary', {})

            total_gt_fields = sql_summary.get('gt_total', 0)
            sql_fully_matched = sql_summary.get('correct', 0)

            sql_accuracy = (sql_fully_matched / total_gt_fields) if total_gt_fields > 0 else 0.0

            # Aggregate by domain
            domain_metrics[domain]["count"] += 1
            domain_metrics[domain]["calc_selection_acc"].append(calc_selection_acc)
            domain_metrics[domain]["calc_value_acc"].append(calc_value_acc)
            domain_metrics[domain]["sql_extraction_acc"].append(sql_accuracy)

    # Calculate averages per domain
    results = {}
    all_calc_sel = []
    all_calc_val = []
    all_sql = []

    for domain, data in domain_metrics.items():
        if data["count"] > 0:
            results[domain] = {
                "count": data["count"],
                "calc_selection_acc": sum(data["calc_selection_acc"]) / len(data["calc_selection_acc"]),
                "calc_value_acc": sum(data["calc_value_acc"]) / len(data["calc_value_acc"]),
                "sql_extraction_acc": sum(data["sql_extraction_acc"]) / len(data["sql_extraction_acc"])
            }

            # Collect for overall average
            all_calc_sel.extend(data["calc_selection_acc"])
            all_calc_val.extend(data["calc_value_acc"])
            all_sql.extend(data["sql_extraction_acc"])

    # Calculate overall average (Avg)
    if all_calc_sel:
        results["Avg"] = {
            "count": len(all_calc_sel),
            "calc_selection_acc": sum(all_calc_sel) / len(all_calc_sel),
            "calc_value_acc": sum(all_calc_val) / len(all_calc_val),
            "sql_extraction_acc": sum(all_sql) / len(all_sql)
        }

    # Sort domains
    domains = sorted([d for d in results.keys() if d != "Avg"])
    domains.append("Avg")

    # Print results to console
    print("="*60)
    print("        EVALUATION RESULTS BY SCENARIO SUMMARY        ")
    print("="*60)
    print(f"Total Domains: {len([d for d in domains if d != 'Avg'])}")
    print("-" * 60)

    for domain in domains:
        if domain in results:
            data = results[domain]
            if domain == "Avg":
                print("-" * 60)
                print("Overall Average (Avg):")
            else:
                print(f"\nDomain: {domain}")
            print(f"  - Count:                    {data['count']}")
            print(f"  - Calc Selection Acc:       {data['calc_selection_acc']:.2%}")
            print(f"  - Calc Value Acc:           {data['calc_value_acc']:.2%}")
            print(f"  - SQL Extraction Acc:       {data['sql_extraction_acc']:.2%}")
            if domain != "Avg":
                print("-" * 60)

    print("="*60)
    
    
    # Export to CSV files
    # CSV: Domain-specific details
    detail_rows = []
    for domain in domains:
        if domain in results:
            data = results[domain]
            detail_rows.append({
                "Domain": domain,
                "Count": data["count"],
                "Calc_Selection_Acc": f"{data['calc_selection_acc']:.4f}",
                "Calc_Value_Acc": f"{data['calc_value_acc']:.4f}",
                "SQL_Extraction_Acc": f"{data['sql_extraction_acc']:.4f}"
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(detail_output_path, index=False)
    print(f"\n[Detail by Domain] Saved to: {detail_output_path}")


def analyze_evaluation_results():
    """
    Reads the evaluation JSONL file, calculates statistics for:
    1. Process Success Rate
    2. Calculator Selection Accuracy
    3. Calculator Calculation Accuracy
    4. SQL Data Extraction Accuracy
    5. Waste/Redundancy Rates
    6. Tool Usage Statistics
    
    Outputs a CSV file and prints a global summary.
    """
    detail_output_path = "./evaluation_detail.csv"
    
    data_rows = []
    
    # Check if file exists
    if not os.path.exists(args.output_path):
        print(f"Error: File not found at {args.output_path}")
        return

    with jsonlines.open(args.output_path) as reader:
        for obj in reader:
            task_item = {}
            with jsonlines.open(args.task_path) as reader:
                for item in reader:
                    if item["task_id"] == obj["task_id"]:
                        task_item = item
                        break
            # Raise error if the task ID is not found in the ground truth file
            if not task_item:
                raise KeyError(f"Task ID {obj['task_id']} not found in task_path")

            # Calculator Statistics
            total_gt_calcs = len(task_item["calculator_answers"])
            
            correct_id_calcs = 0
            correct_val_calcs = 0
            try:
                for res in obj["calculators_compare_result"]["comparison_details"]:
                    if res["name_matched"] == True:
                        correct_id_calcs += 1
                    if res["result_matched"] == True:
                        correct_val_calcs += 1
            except Exception as e:
                print(obj)
                raise e

            extra_calcs_list = obj["calculators_compare_result"]["summary"]["extra_calculators"]
            extra_calcs_count = len(extra_calcs_list) if isinstance(extra_calcs_list, list) else 0
            missing_calcs_list = obj["calculators_compare_result"]["summary"]["missing_calculators"]

            calc_selection_acc = (correct_id_calcs / total_gt_calcs) if total_gt_calcs > 0 else 0.0
            calc_value_acc = (correct_val_calcs / total_gt_calcs) if total_gt_calcs > 0 else 0.0

            # SQL Data Statistics

            sql_res = obj.get('sql_data_compare_result', {})
            sql_summary = sql_res.get('summary', {})

            total_gt_fields = sql_summary.get('gt_total', 0)
            sql_fully_matched = sql_summary.get('correct', 0)

            sql_accuracy = (sql_fully_matched / total_gt_fields) if total_gt_fields > 0 else 0.0

            # Tool Usage Statistics
            tools = obj.get('tools_times', {})
            run_sql = tools.get('run_sql', 0)
            run_python = tools.get('run_python', 0)
            search = tools.get('search', 0)
            fetch = tools.get('fetch', 0)
            
            total_steps = run_sql + run_python + search + fetch + 1

            # Compile metrics into row dictionary
            row = {
                "task_id": obj["task_id"],
                "success_rate": obj["process_success"],

                "total_gt_calcs": total_gt_calcs,
                "correct_id_calcs": correct_id_calcs,
                "calc_selection_acc": round(calc_selection_acc, 4),
                "correct_val_calcs": correct_val_calcs,
                "calc_value_acc": round(calc_value_acc, 4),
                "extra_calcs_count": extra_calcs_count,

                "total_gt_fields": total_gt_fields,
                "sql_fully_matched": sql_fully_matched,
                "sql_extraction_acc": round(sql_accuracy, 4),

                "tool_sql": run_sql,
                "tool_python": run_python,
                "tool_search": search,
                "tool_fetch": fetch,

                "total_steps": total_steps
            }
            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    
    if df.empty:
        print("No valid data rows found to analyze.")
        return

    # Aggregate statistics across all tasks
    summary = {
        "Total_Tasks": len(df),
        
        "Avg_Success_Rate": df['success_rate'].mean(),
        
        "Avg_Calc_Selection_Acc": df['calc_selection_acc'].mean(),
        "Avg_Calc_Value_Acc": df['calc_value_acc'].mean(),
        
        "Avg_SQL_Extraction_Acc": df['sql_extraction_acc'].mean(),
        
        "Avg_Extra_Calcs_Waste": df['extra_calcs_count'].mean(),
        
        "Avg_SQL_Calls": df['tool_sql'].mean(),
        "Avg_Python_Calls": df['tool_python'].mean(),
        "Avg_Search_Calls": df['tool_search'].mean(),
        "Avg_Fetch_Calls": df['tool_fetch'].mean(),
        
        "Avg_Steps": df['total_steps'].mean()
    }

    # Print summary to console
    print("="*50)
    print("           EVALUATION STATISTICS SUMMARY           ")
    print("="*50)
    print(f"Total Tasks Processed: {summary['Total_Tasks']}")
    print("-" * 30)
    print(f"1. Process Success Rate:           {summary['Avg_Success_Rate']:.2%}")
    print(f"2. Calculator Selection Accuracy:  {summary['Avg_Calc_Selection_Acc']:.2%}")
    print(f"3. Calculator Value Accuracy:      {summary['Avg_Calc_Value_Acc']:.2%}")
    print(f"4. SQL Data Extraction Accuracy:   {summary['Avg_SQL_Extraction_Acc']:.2%}")
    print("-" * 30)
    print(f"5. Waste Metrics (Avg per task):")
    print(f"   - Extra Calculators Found:      {summary['Avg_Extra_Calcs_Waste']:.2f}")
    print("-" * 30)
    print(f"6. Tool Usage (Avg steps per task):")
    print(f"   - SQL:    {summary['Avg_SQL_Calls']:.2f}")
    print(f"   - Python: {summary['Avg_Python_Calls']:.2f}")
    print(f"   - Search: {summary['Avg_Search_Calls']:.2f}")
    print(f"   - Fetch:  {summary['Avg_Fetch_Calls']:.2f}")
    print("-" * 30)
    print(f"7. Total Steps:                    {summary['Avg_Steps']:.2f}")
    print("="*50)

    # Save detailed and summary results
    df.to_csv(detail_output_path, index=False)
    print(f"\n[Detail] Detailed statistics saved to: {detail_output_path}")


if __name__ == "__main__":

    if args.stats_only:
        analyze_evaluation_results()
        print("\n\n")
        analyze_evaluation_results_by_scenario()
        exit()

    with jsonlines.open(args.input_path) as reader:
        data = [obj for obj in reader]

    print(f"Total data size: {len(data)}")

    # Resume from previously processed tasks
    processed_ids = []
    if os.path.exists(args.output_path):
        with jsonlines.open(args.output_path, "r") as reader:
            processed_ids = [obj["task_id"] for obj in reader]

    rest_data = [obj for obj in data if obj["task_id"] not in processed_ids]
    print(f"Remaining items to process: {len(rest_data)}")

    # Process tasks concurrently with thread pool
    with ThreadPoolExecutor(max_workers = args.max_workers) as executor:
        futures = [executor.submit(main, d) for d in rest_data]

        for future in tqdm(as_completed(futures), total=len(rest_data)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing data item: {e}")
                

