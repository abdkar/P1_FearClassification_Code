#!/usr/bin/env python3
"""
Export metrics from SQLite database to MLflow file format
"""

import sqlite3
import os
import json
from pathlib import Path

def export_db_to_mlflow():
    """Export metrics from SQLite database to MLflow file format"""
    
    db_path = "/home/amir/P1_FearClassification_Code/mlruns/mlflow.db"
    mlruns_path = "/home/amir/P1_FearClassification_Code/mlruns"
    
    print(f"🔄 Exporting database metrics to MLflow file format...")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all experiments
    cursor.execute("SELECT experiment_id, name FROM experiments")
    experiments = cursor.fetchall()
    
    for exp_id, exp_name in experiments:
        print(f"📁 Processing experiment {exp_id}: {exp_name}")
        
        # Get all runs for this experiment
        cursor.execute("SELECT run_uuid, name FROM runs WHERE experiment_id = ?", (exp_id,))
        runs = cursor.fetchall()
        
        for run_uuid, run_name in runs:
            print(f"  🏃 Processing run {run_uuid}: {run_name}")
            
            run_dir = os.path.join(mlruns_path, str(exp_id), run_uuid)
            os.makedirs(run_dir, exist_ok=True)
            
            # Export metrics
            cursor.execute("""
                SELECT key, value, timestamp, step 
                FROM metrics 
                WHERE run_uuid = ? 
                ORDER BY key, step, timestamp
            """, (run_uuid,))
            
            metrics = cursor.fetchall()
            metrics_dir = os.path.join(run_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Group metrics by key
            metric_groups = {}
            for key, value, timestamp, step in metrics:
                if key not in metric_groups:
                    metric_groups[key] = []
                metric_groups[key].append((value, timestamp, step))
            
            # Write each metric to its own file
            for metric_key, values in metric_groups.items():
                metric_file = os.path.join(metrics_dir, metric_key)
                with open(metric_file, 'w') as f:
                    for value, timestamp, step in values:
                        f.write(f"{timestamp} {value} {step}\n")
                
                print(f"    📊 Exported {len(values)} values for metric '{metric_key}'")
            
            # Export parameters
            cursor.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_uuid,))
            params = cursor.fetchall()
            
            if params:
                params_dir = os.path.join(run_dir, "params")
                os.makedirs(params_dir, exist_ok=True)
                
                for key, value in params:
                    param_file = os.path.join(params_dir, key)
                    with open(param_file, 'w') as f:
                        f.write(str(value))
                
                print(f"    📝 Exported {len(params)} parameters")
            
            # Export tags
            cursor.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_uuid,))
            tags = cursor.fetchall()
            
            if tags:
                tags_dir = os.path.join(run_dir, "tags")
                os.makedirs(tags_dir, exist_ok=True)
                
                for key, value in tags:
                    tag_file = os.path.join(tags_dir, key)
                    with open(tag_file, 'w') as f:
                        f.write(str(value) if value else "")
                
                print(f"    🏷️  Exported {len(tags)} tags")
    
    conn.close()
    print("✅ Export completed! MLflow UI should now show all the data.")

if __name__ == "__main__":
    export_db_to_mlflow()
